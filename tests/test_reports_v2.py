"""Public v2 report behavior, including unregistered Benchmark names."""

from __future__ import annotations

import csv
import json
import re
from urllib.parse import unquote

import pytest

from luban_meter.cli import main
from luban_meter.core.models import BenchmarkResult
from luban_meter.core.registry import BenchmarkRegistry
from luban_meter.reporting.data import from_payload, load_report, lookup
from luban_meter.reporting.render import write_report
from luban_meter.result.report_spec import bar, line, table
from luban_meter.result.schema import RESULT_SCHEMA, SUITE_SCHEMA
from luban_meter.suite.loader import SuiteLoader
from luban_meter.utils.json_io import to_jsonable
from tests.test_suite import BENCHMARK_SOURCE, RESULT_SOURCE


def result(metrics=None, *, run_id="demo", report=None):
    return to_jsonable(
        BenchmarkResult(
            schema_version=RESULT_SCHEMA,
            run_id=run_id,
            status="success",
            module="custom",
            benchmark="never-registered",
            config="test.yaml",
            metrics=metrics or {},
            metadata={"report": report} if report else {},
        )
    )


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def suite(outputs):
    return {
        "schema_version": SUITE_SCHEMA,
        "suite_id": "suite-1",
        "name": "demo",
        "status": "partial_failed",
        "metadata": {},
        "tasks": [
            {
                "name": f"task-{i}",
                "module": item["module"],
                "benchmark": item["benchmark"],
                "run_id": item["run_id"],
                "status": item["status"],
                "result": "unavailable/result.json",
                "output": item,
            }
            for i, item in enumerate(outputs)
        ],
    }


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_unknown_benchmark_auto_extracts_without_registration(tmp_path):
    metrics = {
        "plain": 8,
        "zero": 0,
        "numeric_unit": {"unit": 3},
        "typed": {"value": 1.2, "unit": "ms"},
        "a/b~c": {"p50": None, "p99": 2, "count": 0, "unit": "s"},
        "cases": [{"size": 4, "speed": {"value": 5, "unit": "op/s"}}],
        "analysis": "Do not include prose analysis in numeric reports",
    }
    source = save(tmp_path / "含 空格/result.json", result(metrics))
    original = source.read_bytes()
    path, _ = write_report(load_report(source), tmp_path / "report")
    content = path.read_text()
    assert "1.2" in content and "op/s" in content
    assert "Do not include" not in content
    rows = {r["metric"]: r for r in csv_rows(path.parent / "results.csv")}
    assert rows["/zero"]["value"] == "0"
    assert rows["/numeric_unit/unit"]["value"] == "3"
    assert rows["/numeric_unit/unit"]["unit"] == ""
    assert rows["/a~1b~0c/p50"]["value"] == ""
    assert rows["/a~1b~0c/count"]["unit"] == "count"
    assert rows["/cases/0/speed/value"]["value"] == "5"
    assert source.read_bytes() == original
    assert lookup(metrics, "/a~1b~0c/p99") == (2, "s")
    reordered = json.loads(original)
    reordered["metrics"] = dict(reversed(reordered["metrics"].items()))
    csv_before = (path.parent / "results.csv").read_bytes()
    write_report(from_payload(reordered, source), path.parent)
    assert path.read_text() == content
    assert (path.parent / "results.csv").read_bytes() == csv_before


def test_summary_is_bounded_but_csv_preserves_metrics(tmp_path):
    source = save(
        tmp_path / "result.json",
        result({f"measure-{i}": i for i in range(55)}),
    )
    report = load_report(source)
    path, _ = write_report(report, tmp_path / "out")
    assert "前 30 个指标" in path.read_text()
    assert "/measure-54" not in path.read_text()
    assert len(csv_rows(path.parent / "results.csv")) == 55
    original = path.read_bytes()
    write_report(report, tmp_path / "out")
    assert path.read_bytes() == original


def test_declared_table_and_images_follow_each_other(tmp_path):
    from PIL import Image

    data = result(
        {
            "cases": [
                {
                    "size": size,
                    "length": 128,
                    "speed": {"value": value, "unit": "op/s"},
                }
                for size, value in ((8, 80), (1, 10), (4, None))
            ],
            "subjects": {"数学": {"value": 0.8, "unit": "ratio"}},
            "unselected": 999,
        },
        report={
            "tables": [
                table(
                    "吞吐",
                    "/cases",
                    {"/size": "Size", "/speed": "Speed"},
                    charts=[line("/size", "/speed", ["/length"])],
                ),
                table(
                    "效果",
                    "/subjects",
                    {"/key": "Subject", "/value": "Score"},
                    mapping=True,
                    charts=[bar("/value", "/key")],
                ),
            ]
        },
    )
    source = save(tmp_path / "result.json", data)
    path, _ = write_report(load_report(source), tmp_path / "out")
    content = path.read_text()
    assert "999" not in content
    assert (
        content.index("### 吞吐")
        < content.index("![")
        < content.index("### 效果")
    )
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
    assert len(images) == 2
    for image in images:
        with Image.open(path.parent / unquote(image)) as png:
            assert png.width >= 1000
    assert any(
        r["metric"] == "/unselected"
        for r in csv_rows(path.parent / "results.csv")
    )


def test_suite_is_self_contained_and_keeps_failures(tmp_path):
    failed = result(run_id="failed")
    failed.update(
        status="failed", error={"type": "TestFailure", "message": "offline"}
    )
    payload = suite([result({"speed": 4}), failed])
    payload["tasks"].append(
        {
            "name": "skipped",
            "module": "custom",
            "benchmark": "any",
            "status": "skipped",
        }
    )
    source = save(tmp_path / "moved/suite_result.json", payload)
    path, _ = write_report(load_report(source), tmp_path / "out")
    content = path.read_text()
    assert "offline" in content and "skipped" in content
    assert content.index("## task-0") < content.index("## task-1")
    assert len(csv_rows(path.parent / "results.csv")) == 3
    payload["tasks"][0]["output"]["benchmark"] = "wrong"
    with pytest.raises(ValueError, match="身份不一致"):
        from_payload(payload, source)


def test_multiple_suites_generate_independent_reports(tmp_path, capsys):
    first = save(
        tmp_path / "first/suite_result.json", suite([result({"a": 1})])
    )
    second = save(
        tmp_path / "second/suite_result.json", suite([result({"b": 9})])
    )
    assert (
        main(
            [
                "report",
                "--input",
                str(first),
                "--input",
                str(second),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    paths = sorted((tmp_path / "out").glob("*/report.md"))
    assert len(paths) == 2
    values = [
        set(r["value"] for r in csv_rows(p.parent / "results.csv"))
        for p in paths
    ]
    assert {frozenset(v) for v in values} == {
        frozenset({"1"}),
        frozenset({"9"}),
    }
    assert not list((tmp_path / "out").rglob("comparison*"))
    assert "未比较" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["compare", "--baseline", str(first)])


def test_v1_rejected_and_bad_input_does_not_hide_valid_report(tmp_path):
    legacy = result()
    legacy["schema_version"] = "luban-meter.result/v1"
    first = save(tmp_path / "v1.json", legacy)
    second = save(tmp_path / "v2.json", result({"a": 1}))
    assert (
        main(
            [
                "report",
                "--input",
                str(first),
                "--input",
                str(second),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    assert len(list((tmp_path / "out").glob("*/report.md"))) == 1
    invalid = result()
    invalid["metrics"] = []
    with pytest.raises(ValueError, match="metrics"):
        from_payload(invalid, second)


def test_multiple_inputs_in_same_directory_do_not_overwrite(tmp_path):
    paths = [
        save(tmp_path / f"suite-{value}.json", suite([result({"x": value})]))
        for value in (1, 9)
    ]
    assert (
        main(["report", "--input", str(paths[0]), "--input", str(paths[1])])
        == 0
    )
    reports = list((tmp_path / "report").glob("*/results.csv"))
    assert len(reports) == 2
    assert {csv_rows(path)[0]["value"] for path in reports} == {"1", "9"}


def test_invalid_optional_layout_falls_back_to_values(tmp_path):
    data = result({"latency": 3}, report={"tables": [{"title": "bad"}]})
    source = save(tmp_path / "result.json", data)
    path, _ = write_report(load_report(source), tmp_path / "out")
    assert "报告声明无效" in path.read_text()
    assert "/latency" in path.read_text()


def test_json_console_and_report_failure_preserve_result(
    tmp_path, capsys, monkeypatch
):
    def run(_self, request):
        print("hardware monitor stdout")
        data = result({"speed": 4}, run_id=request.run_id)
        save(request.output_dir / request.run_id / "result.json", data)
        return BenchmarkResult(**data)

    monkeypatch.setattr("luban_meter.cli.CoreEngine.run", run)
    args = [
        "run",
        "--module",
        "custom",
        "--benchmark",
        "test",
        "--config",
        "test.yaml",
        "--format",
        "json",
        "--output",
        str(tmp_path),
    ]
    assert main(args) == 0
    streams = capsys.readouterr()
    assert json.loads(streams.out)["schema_version"] == RESULT_SCHEMA
    assert "hardware monitor stdout" in streams.err

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr("luban_meter.cli.write_report", fail)
    assert main(args) == 0
    streams = capsys.readouterr()
    assert json.loads(streams.out)["metrics"] == {"speed": 4}
    assert "disk full" in streams.err


def test_saved_monitoring_charts_are_combined_using_artifact_directory(
    tmp_path,
):
    from PIL import Image

    directory = tmp_path / "run/raw/artifacts"
    directory.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(directory / "gpu.png")
    Image.new("RGB", (20, 20), "blue").save(directory / "cpu.png")
    data = result({"score": 1})
    data["artifacts"]["directory"] = str(directory)
    data["environment"]["device_monitoring"] = {
        "charts": ["gpu.png", "cpu.png", "absent.png", "../private.png"]
    }
    source = save(tmp_path / "run/result.json", data)
    path, _ = write_report(load_report(source), tmp_path / "out")
    images = list((path.parent / "figures").glob("*.png"))
    assert len(images) == 1
    content = path.read_text()
    assert content.index("### 硬件环境与监控") < content.index("### 指标摘要")
    assert content.count("![") == 1
    with Image.open(images[0]) as png:
        assert png.width > 1000


def test_cli_automatically_reports_real_runs_and_suite(
    tmp_path, monkeypatch, capsys
):
    benchmark_root = tmp_path / "benchmark"
    custom = benchmark_root / "inference/custom"
    custom.mkdir(parents=True)
    (custom / "benchmark.py").write_text(BENCHMARK_SOURCE, encoding="utf-8")
    (custom / "result.py").write_text(RESULT_SOURCE, encoding="utf-8")
    config = tmp_path / "test.yaml"
    config.write_text("value: 123\n", encoding="utf-8")
    suites = tmp_path / "definitions"
    suites.mkdir()
    (suites / "demo.yaml").write_text(
        "name: demo\ntasks:\n"
        "  - name: custom\n    module: inference\n"
        "    benchmark: custom\n    config: ../test.yaml\n"
        "  - name: broken\n    module: inference\n"
        "    benchmark: missing\n    config: ../test.yaml\n"
        "  - name: later\n    module: inference\n"
        "    benchmark: custom\n    config: ../test.yaml\n",
        encoding="utf-8",
    )
    registry = BenchmarkRegistry(benchmark_root)
    loader = SuiteLoader(suites)
    monkeypatch.setattr("luban_meter.cli.BenchmarkRegistry", lambda: registry)
    monkeypatch.setattr("luban_meter.cli.SuiteLoader", lambda: loader)
    monkeypatch.delenv("LUBAN_MONITOR_URL", raising=False)
    output = tmp_path / "runs"
    common = ["--output", str(output), "--name", "第一次", "--format", "json"]
    assert (
        main(
            [
                "run",
                "--module",
                "inference",
                "--benchmark",
                "custom",
                "--config",
                str(config),
                *common,
            ]
        )
        == 0
    )
    single = json.loads(capsys.readouterr().out)
    assert single["schema_version"] == RESULT_SCHEMA
    assert single["metrics"] == {"value": 123}
    assert single["metadata"]["display_name"] == "第一次"
    assert (output / single["run_id"] / "report/report.md").is_file()
    assert main(["suite", "--suite", "demo", "--fail-fast", *common]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SUITE_SCHEMA
    assert [task["status"] for task in payload["tasks"]] == [
        "success",
        "failed",
        "skipped",
    ]
    suite_dir = output / payload["suite_id"]
    summary = (suite_dir / "report/report.md").read_text()
    assert summary.index("## custom") < summary.index("## broken")
    assert summary.index("## broken") < summary.index("## later")
    assert "123" in summary and "skipped" in summary
    for task in payload["tasks"][:2]:
        directory = suite_dir / "tasks" / task["run_id"]
        assert (directory / "report/report.md").is_file()
        saved = json.loads((directory / "result.json").read_text())
        assert saved == task["output"]
    assert payload["tasks"][2]["output"] is None
