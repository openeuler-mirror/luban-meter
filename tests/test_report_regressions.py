"""Regressions for failed results and portable, faithful report output."""

import json

import pytest
from matplotlib.figure import Figure

from luban_meter.cli import main
from luban_meter.core.registry import BenchmarkRegistry
from luban_meter.reporting.data import load_report
from luban_meter.reporting.render import write_report
from luban_meter.result.report_spec import bar, line, table
from luban_meter.result.schema import validate_result
from luban_meter.suite.loader import SuiteLoader
from tests.test_reports_v2 import csv_rows, result, save, suite
from tests.test_suite import BENCHMARK_SOURCE, RESULT_SOURCE


@pytest.fixture
def malformed_run(tmp_path, monkeypatch):
    directory = tmp_path / "benchmark/generate/custom"
    directory.mkdir(parents=True)
    (directory / "benchmark.py").write_text(
        BENCHMARK_SOURCE.replace(
            '"status": "success",',
            '"status": payload["parameters"]["status"],\n'
            '        "error": payload["parameters"].get("error"),\n'
            '        "hardware_environment": '
            '{"cpu": {"model": "recorded-cpu"}},',
        ),
        encoding="utf-8",
    )
    (directory / "result.py").write_text(RESULT_SOURCE, encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text("status: failed\nerror: boom\nvalue: 1\n")
    good = tmp_path / "good.yaml"
    good.write_text("status: success\nvalue: 42\n")
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    (definitions / "regression.yaml").write_text(
        "name: regression\ntasks:\n"
        "  - name: bad\n    module: generate\n"
        "    benchmark: custom\n    config: ../bad.yaml\n"
        "  - name: good\n    module: generate\n"
        "    benchmark: custom\n    config: ../good.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "luban_meter.cli.BenchmarkRegistry",
        lambda: BenchmarkRegistry(tmp_path / "benchmark"),
    )
    monkeypatch.setattr(
        "luban_meter.cli.SuiteLoader", lambda: SuiteLoader(definitions)
    )
    monkeypatch.delenv("LUBAN_MONITOR_URL", raising=False)
    return bad, tmp_path / "runs"


def test_invalid_result_is_saved_as_a_diagnostic(malformed_run, capsys):
    config, output = malformed_run
    assert main(
        [
            "run", "--module", "generate", "--benchmark", "custom",
            "--config", str(config), "--output", str(output),
            "--format", "json",
        ]
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    validate_result(payload)
    directory = output / payload["run_id"]
    assert payload["status"] == "failed"
    assert payload["metadata"]["failure_stage"] == "validate_result"
    assert "error" in payload["error"]["message"]
    assert json.loads((directory / "result.json").read_text()) == payload
    assert (directory / "raw/raw_result.json").is_file()
    assert payload["artifacts"]["raw_result"] == str(
        directory / "raw/raw_result.json"
    )
    assert payload["environment"]["hardware_environment"]["cpu"] == {
        "model": "recorded-cpu"
    }
    assert (directory / "report/report.md").is_file()


def test_unreadable_raw_result_keeps_the_original_failure(
    malformed_run, capsys
):
    config, output = malformed_run
    entry = config.parent / "benchmark/generate/custom/benchmark.py"
    entry.write_text(
        BENCHMARK_SOURCE.split("with open(args.request")[0]
        + "with open(args.output, 'wb') as stream:\n"
        + "    stream.write(bytes([255]))\n",
        encoding="utf-8",
    )
    assert main([
        "run", "--module", "generate", "--benchmark", "custom",
        "--config", str(config), "--output", str(output),
        "--format", "json",
    ]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["failure_stage"] == "process_result"
    assert payload["error"]["type"] == "UnicodeDecodeError"
    assert payload["environment"] == {}
    validate_result(payload)
    directory = output / payload["run_id"]
    assert (directory / "raw/raw_result.json").read_bytes() == bytes([255])
    assert (directory / "report/report.md").is_file()


@pytest.mark.parametrize("fail_fast", [False, True])
def test_invalid_child_result_respects_suite_fail_fast(
    malformed_run, capsys, fail_fast
):
    _, output = malformed_run
    args = [
        "suite", "--suite", "regression", "--output", str(output),
        "--format", "json",
    ]
    if fail_fast:
        args.append("--fail-fast")
    assert main(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert [entry["status"] for entry in payload["tasks"]] == [
        "failed", "skipped" if fail_fast else "success",
    ]
    directory = output / payload["suite_id"]
    assert json.loads((directory / "suite_result.json").read_text()) == payload
    markdown = (directory / "report/report.md").read_text()
    assert "## bad" in markdown and "## good" in markdown
    assert "skipped" in markdown if fail_fast else "42" in markdown
    validate_result(payload["tasks"][0]["output"])


@pytest.mark.parametrize("bad_type", [[], ["line"], {}, None, 7, True])
def test_invalid_optional_chart_does_not_block_suite(
    tmp_path, bad_type
):
    declaration = {
        "tables": [table("Latency", "", {"/latency": "Latency"})]
    }
    declaration["tables"][0]["charts"] = [
        {"type": bad_type, "y": "/latency"}
    ]
    source = save(
        tmp_path / "suite_result.json",
        suite([
            result({"throughput": 42}, run_id="good"),
            result({"latency": 3}, run_id="bad", report=declaration),
        ]),
    )
    before = source.read_bytes()
    output = tmp_path / "report"
    assert main([
        "report", "--input", str(source), "--output", str(output)
    ]) == 0
    content = (output / "report.md").read_text()
    assert "报告声明无效" in content
    assert "/latency" in content and "/throughput" in content
    assert {row["value"] for row in csv_rows(output / "results.csv")} == {
        "3", "42",
    }
    assert source.read_bytes() == before


@pytest.mark.parametrize("shape", ["dict", "list", "mapping"])
def test_inherited_units_reach_tables_and_charts(
    tmp_path, monkeypatch, shape
):
    record = {
        "p50": 2,
        "p99": {"value": 8, "unit": "us"},
        "count": 3,
        "zero": 0,
        "missing": None,
        "bare": {"value": 5, "unit": ""},
    }
    value = {"dict": record, "list": [record], "mapping": {"A": record}}
    prefix = "/value" if shape == "mapping" else ""
    columns = {f"{prefix}/{key}": key for key in record}
    if shape == "mapping":
        columns["/key"] = "Group"
    declaration = {
        "tables": [table(
            "Latency", "/latency/summary", columns,
            mapping=shape == "mapping", charts=[bar(f"{prefix}/p50")],
        )]
    }
    source = save(
        tmp_path / "result.json",
        result(
            {"latency": {"unit": "ms", "summary": value[shape]}},
            report=declaration,
        ),
    )
    before = source.read_bytes()
    labels = []
    original_save = Figure.savefig

    def save_figure(figure, *args, **kwargs):
        labels.extend(axis.get_ylabel() for axis in figure.axes)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save_figure)
    path, console = write_report(load_report(source), tmp_path / "report")
    expected = "| 2 ms | 8 us | 3 count | 0 ms | — | 5 |"
    assert expected in path.read_text() and expected in console
    assert "p50 ms" in labels
    if shape == "mapping":
        assert "| A |" in path.read_text()
        assert "A ms" not in path.read_text()
    rows = csv_rows(path.parent / "results.csv")
    p50 = next(row for row in rows if row["metric"].endswith("/p50"))
    assert p50["unit"] == "ms"
    assert source.read_bytes() == before


def test_inference_context_preserves_recorded_values_and_layout(tmp_path):
    data = result(
        {"score": 0.8},
        report={"tables": [table("Score", "", {"/score": "Score"})]},
    )
    data["module"] = "inference"
    data["model"] = {"name": "request-model"}
    data["metadata"].update({
        "model": "actual-model", "dataset": "recorded-dataset",
        "split": "test", "sample_count": 0, "few_shot": 0,
        "eval_mode": "ppl", "prompt_format": "base",
        "prompt_version": "prompt-v2", "temperature": 0.0,
        "max_tokens": None, "stop": [], "scorer_version": "scorer-v3",
    })
    data["parameters"] = {
        "temperature": 0.8, "max_tokens": 999,
        "stop": ["configured-stop"], "few_shot": 7,
        "shuffle": False, "seed": 0, "max_samples": 12,
        "prompt_template": "full-prompt-must-not-appear",
    }
    data["environment"]["hardware_environment"] = {
        "cpu": {"model": "recorded-cpu"}
    }
    source = save(tmp_path / "result.json", data)
    before = source.read_bytes()
    path, console = write_report(load_report(source), tmp_path / "report")
    content = path.read_text()
    assert (
        content.index("### 硬件环境与监控")
        < content.index("### 评测条件")
        < content.index("### Score")
    )
    for text in (content, console):
        for term in (
            "actual-model", "recorded-dataset", "ppl", "base",
            "prompt-v2", "scorer-v3", "未记录", "max_tokens=null",
            "stop=[]", "temperature=0.0", "shuffle=false", "seed=0",
        ):
            assert term in text
        for term in (
            "request-model", "configured-stop", "999",
            "full-prompt-must-not-appear",
        ):
            assert term not in text
        assert "样本数 | 0" in text
        assert "Few-shot 设置 | 0" in text
    assert len(csv_rows(path.parent / "results.csv")) == 1
    assert source.read_bytes() == before


def test_mapping_table_does_not_turn_unit_metadata_into_a_metric(tmp_path):
    source = save(
        tmp_path / "result.json",
        result(
            {"latencies": {"unit": "ms", "A": 0, "B": 4}},
            report={"tables": [table(
                "Latency", "/latencies",
                {"/key": "Name", "/value": "Latency"},
                mapping=True,
            )]},
        ),
    )
    path, _ = write_report(load_report(source), tmp_path / "report")
    assert "| A | 0 ms |" in path.read_text()
    assert "| B | 4 ms |" in path.read_text()
    assert "| unit |" not in path.read_text()


def test_line_chart_preserves_units_on_both_axes(tmp_path, monkeypatch):
    source = save(
        tmp_path / "result.json",
        result(
            {"latency": {"unit": "ms", "rows": [
                {"time": 1, "delay": 2}, {"time": 2, "delay": 3},
            ]}},
            report={"tables": [table(
                "Latency", "/latency/rows",
                {"/time": "Time", "/delay": "Delay"},
                charts=[line("/time", "/delay", [])],
            )]},
        ),
    )
    labels = []
    original_save = Figure.savefig

    def save_figure(figure, *args, **kwargs):
        labels.extend(
            (axis.get_xlabel(), axis.get_ylabel()) for axis in figure.axes
        )
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save_figure)
    path, _ = write_report(load_report(source), tmp_path / "report")
    assert labels == [("Time ms", "Delay ms")]
    assert "| 1 ms | 2 ms |" in path.read_text()


def test_suite_context_uses_each_tasks_own_metadata(tmp_path):
    outputs = []
    for index in (0, 1):
        data = result({"score": index}, run_id=f"run-{index}")
        data["module"] = "inference"
        data["metadata"] = {"model": f"model-{index}"}
        outputs.append(data)
    source = save(tmp_path / "suite_result.json", suite(outputs))
    path, _ = write_report(load_report(source), tmp_path / "report")
    sections = path.read_text().split("## task-")[1:]
    for index, section in enumerate(sections):
        assert f"model-{index}" in section
        assert f"model-{1 - index}" not in section
        assert "未记录" in section
