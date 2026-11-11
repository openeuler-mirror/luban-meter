"""Terminal reports preserve saved statistics and execution semantics."""

import copy
import json
import threading
from pathlib import Path

import pytest

from luban_meter import cli
from luban_meter.benchmarking.benchmark_metrics import SampleAccuracy
from luban_meter.reporting.data import load_result, metric_rows
from luban_meter.reporting.html import report_data
from luban_meter.reporting.terminal import (
    display_width,
    distribution_values,
    render_text,
)
from luban_meter.utils.json_io import write_json_atomic


@pytest.fixture
def result():
    return {
        "kind": "run", "id": "terminal-example", "status": "success",
        "tasks": [{
            "kind": "task", "name": "task_1", "scenario": "online_serving",
            "category": "generation_performance", "status": "success",
            "cases": [{
                "id": "case_1", "status": "success",
                "parameters": {"model": "Qwen3-30B-A3B", "data": "jsonl",
                               "profile": "concurrent", "max_concurrency": 4,
                               "num_warmups": 5, "output_tokens": 16},
                "coverage": {"selected": 50, "complete": 50, "failed": 0,
                             "incomplete": 0, "not_started": 0},
                "window": {"duration_s": 30, "started_at": 100,
                           "ended_at": 130, "request_capacity": 4,
                           "peak_requests": 4},
                "metrics": {"streaming_performance": {
                    "status": "success", "observation_boundary": "client",
                    "values": {
                        "request_view": {
                            "ttft": {"mean": 48.67, "median": 46.356,
                                     "p95": 59.401, "count": 50, "unit": "ms"},
                            "itl": {"mean": 20, "p50": 19, "p95": 24,
                                    "count": 50, "unit": "ms/token",
                                    "weight_sum": 750,
                                    "aggregation": "token_weighted",
                                    "percentile_method": "weighted_cdf"},
                        },
                        "service_view": {
                            "request_throughput": {"value": 1.667,
                                                   "unit": "req/s"},
                            "output_token_throughput": {
                                "value": None, "unit": "token/s",
                                "reason": "missing usage"},
                            "benchmark_duration": {"value": 30, "unit": "s"},
                            "total_requests": {"value": 50},
                            "successful_requests": {"value": 50},
                            "failed_requests": {"value": 0},
                        },
                    },
                }},
            }],
        }],
    }


def test_statistics_match_html_and_do_not_change_facts(result):
    before = copy.deepcopy(result)
    text = render_text(result, width=120, verbose=True)
    html_rows = report_data(result)["rows"]
    assert next(r["value"] for r in html_rows
                if r["metric"].endswith("/ttft/mean")) == 48.67
    assert "48.67" in text and "46.356" in text and "59.401" in text
    assert "weight_sum=750" in text
    assert "token_weighted" in text and "weighted_cdf" in text
    assert "boundary=client" in text
    assert "duration_s=30" in text
    assert "successful_requests" in text and "50" in text
    assert "missing usage" in text
    assert "service_view/output_token_throughput" in text
    assert result == before
    assert list(metric_rows(result["tasks"][0]["cases"][0]["metrics"]))


def test_empty_and_missing_metrics_never_become_zero(result):
    case = result["tasks"][0]["cases"][0]
    case.update(coverage={}, window={}, parameters={}, metrics={})
    text = render_text(result)
    assert "complete=N/A" in text
    assert "duration_s=N/A" in text
    assert "未保存指标" in text
    assert "complete=0" not in text
    assert distribution_values({"count": 0, "mean": 0})[:3] == (
        "N/A", "N/A", "N/A"
    )
    assert distribution_values({"value": 0})[0] == "0"
    assert distribution_values({"mean": float("nan")})[0] == "N/A"
    assert distribution_values({"mean": float("inf")})[0] == "N/A"
    assert distribution_values({"value": 0.00001})[0] != "0"


def test_quality_metrics_keep_unit_and_scoring_denominator(result):
    task = result["tasks"][0]
    task.update(scenario="gsm8k", category="model_service_quality")
    task["cases"][0]["metrics"] = {"numeric_accuracy": {
        "status": "partial_failed", "values": {"task_view": {"gsm8k": {
            "exact_match": {"value": 0.8, "unit": "ratio", "count": 4},
        }}},
    }}
    text = render_text(result, width=40)
    assert "task_view/gsm8k/exact_match" in text
    assert "[ratio] N=4" in text
    assert "value=0.8" in text
    assert "P50" not in text
    assert "request_view" not in text


def test_builtin_top_level_accuracy_reaches_saved_best_summary(
        result, tmp_path):
    task = result["tasks"][0]
    task.update(scenario="custom", category="model_service_quality")
    case = task["cases"][0]
    case["metrics"] = {"sample_accuracy": SampleAccuracy().compute({
        "samples": [
            {"state": "complete", "facts": {"correct": True}},
            {"state": "complete", "facts": {"correct": False}},
        ],
    })}
    case.update(collection_complete=True, metrics_status="complete")
    destination = tmp_path / "result.json"
    write_json_atomic(destination, result)
    saved = load_result(destination)
    before = copy.deepcopy(saved)
    text = render_text(saved, width=120)
    assert "accuracy: 50 %" in text
    assert "最佳得分: N/A" not in text
    assert "[ratio] N=2" in render_text(saved, width=40)
    assert saved == before


def test_multi_case_limit_and_hierarchical_filters(result):
    task = result["tasks"][0]
    case = task["cases"][0]
    task["cases"] = [dict(case, id=f"case_{i}") for i in range(1, 8)]
    result["tasks"].append(dict(task, name="task_2"))
    text = render_text(result)
    assert "匹配 14，展示 5" in text
    assert "省略 9 个 Case" in text
    assert "[task_1/case_6]" not in text
    assert "[task_2/case_7]" in render_text(result, all_cases=True)
    scoped = render_text(result, tasks=["task_2"], cases=["case_1"])
    assert "[task_2/case_1]" in scoped
    assert "[task_1/case_1]" not in scoped
    exact = render_text(result, cases=["task_1/case_1"])
    assert "[task_2/case_1]" not in exact
    assert "任务: 1/2" in exact
    with pytest.raises(ValueError, match="selected tasks"):
        render_text(result, tasks=["task_1"], cases=["task_2/case_1"])
    with pytest.raises(ValueError, match="unknown task"):
        render_text(result, tasks=["wrong"])


@pytest.mark.parametrize("status", ["partial_failed", "interrupted",
                                   "failed", "discarded", "skipped"])
def test_failure_states_and_empty_tasks(result, status):
    result["status"] = status
    task = result["tasks"][0]
    task["status"] = status
    task["cases"][0]["status"] = status
    task["cases"][0]["metrics"] = {
        "streaming_performance": {"status": "failed", "error": "timeout"}
    }
    text = render_text(result)
    assert status in text
    assert "N/A" in text
    if status != "discarded":
        assert "timeout" in text
    task["cases"] = []
    task["errors"] = ["worker failed"]
    assert "worker failed" in render_text(result)
    assert "无 Case 结果" in render_text(result)
    result["tasks"] = []
    assert "无 Case 结果" in render_text(result)


def test_narrow_terminal_wraps_controls_and_long_identifiers(result):
    task = result["tasks"][0]
    task["name"] = "中文长任务" * 20 + "\x1b[2J\r\n"
    for width in (20, 40, 80, 100, 120):
        text = render_text(result, width=width)
        assert "\x1b" not in text and "\r" not in text
        assert all(display_width(line) <= width for line in text.splitlines())


def test_engine_metrics_keep_distinct_observation_boundary(result):
    case = result["tasks"][0]["cases"][0]
    case["metrics"] = {"engine_performance": {
        "observation_boundary": "engine", "values": {"cases": [{
            "request_metrics": {"internal_ttft": {
                "mean": 5, "unit": "ms", "count": 10}},
        }]},
    }}
    text = render_text(result)
    assert "boundary=engine" in text and "internal_ttft" in text
    assert "request_view/ttft" not in text
    assert "吞吐量范围: 正式 Case" not in text


@pytest.mark.parametrize("level", ["debug", "info", "warn", "error"])
def test_text_cli_stdout_and_file_leave_json_unchanged(
    result, tmp_path, capsys, level,
):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    original = path.read_bytes()
    arguments = ["report", "--input", str(path), "--format", "text",
                 "--log-dir", str(tmp_path / "log"), "--log-level", level]
    assert cli.main(arguments) == 0
    assert "LuBan-Meter 终端结果摘要" in capsys.readouterr().out
    output = tmp_path / "text" / "report.txt"
    assert cli.main([*arguments, "--output", str(output)]) == 0
    assert capsys.readouterr().out == ""
    assert output.is_file()
    assert path.read_bytes() == original
    assert not (tmp_path / "report.html").exists()
    assert load_result(path) == result
    assert cli.main([*arguments, "--output", str(path)]) == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize("include_requests", [False, True])
@pytest.mark.parametrize("level", ["info", "error"])
def test_html_report_locations_use_logs(
    result, tmp_path, capsys, include_requests, level,
):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    log_dir = tmp_path / "log"
    arguments = ["report", "--input", str(path),
                 "--log-dir", str(log_dir), "--log-level", level]
    if include_requests:
        arguments.append("--include-requests")
    assert cli.main(arguments) == 0
    report = tmp_path / (
        "report-full.html" if include_requests else "report.html"
    )
    assert report.is_file()
    captured = capsys.readouterr()
    assert captured.out == ""
    log_path, = log_dir.glob("*.log")
    text = log_path.read_text()
    assert text == captured.err
    assert ("HTML report saved:" in text) is (level == "info")
    assert (str(report) in text) is (level == "info")
    assert ("完整报告含" in text) is (include_requests and level == "info")


@pytest.mark.parametrize("extra", [
    ["--format", "text", "--include-requests"],
    ["--format", "text", "--raw-root", "/tmp"],
    ["--task", "task_1"], ["--all"],
    ["--format", "text", "--task", "wrong"],
])
def test_cli_invalid_options_are_diagnosable(result, tmp_path, capsys, extra):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert cli.main(["report", "--input", str(path), *extra]) == 2
    assert "CLI command failed" in capsys.readouterr().err


@pytest.mark.parametrize("status,code", [("success", 0), ("failed", 1),
                                        ("interrupted", 130)])
def test_summary_failure_preserves_run_artifacts_and_exit(
    result, tmp_path, monkeypatch, capsys, status, code,
):
    result["status"] = status

    def run(*args, **kwargs):
        (tmp_path / "result.json").write_text(json.dumps(result))
        return result, tmp_path

    def broken_summary(*args, **kwargs):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(cli, "build_plan", lambda *a, **k: {
        "tasks": [], "case_count": 0,
    })
    monkeypatch.setattr(cli.RunCoordinator, "run", run)
    monkeypatch.setattr(cli, "render_text", broken_summary)
    assert cli.main(["run"]) == code
    assert (tmp_path / "report.html").is_file()
    assert load_result(tmp_path / "result.json")["status"] == status
    assert "summary failed" in capsys.readouterr().err
    assert cli.main(["run", "--no-summary"]) == code
    captured = capsys.readouterr()
    assert "终端摘要" not in captured.out + captured.err


def test_task_result_and_unknown_extensions_are_viewable(result):
    task = dict(result["tasks"][0], kind="task")
    task["cases"][0]["metrics"] = {
        "custom": {"score": {"value": 0.6, "unit": "ratio",
                              "sample_count": 3, "condition": "scored only"}}
    }
    text = render_text(task)
    assert "score" in text and "0.6" in text
    assert "scored only" in text


def test_inherited_units_and_counts_match_html(result):
    result["tasks"][0]["cases"][0]["metrics"] = {
        "extension": {"unit": "ratio", "sample_count": 7,
                      "accuracy": {"value": 0.7}}
    }
    text = render_text(result, width=80)
    assert any(
        "Accuracy" in line and "ratio" in line
        and line.split("|")[-2].strip() == "7"
        for line in text.splitlines()
    )
    row = report_data(result)["rows"][0]
    assert row["unit"] == "ratio" and row["count"] == 7


def test_documented_example_cli(capsys):
    path = (
        Path(__file__).resolve().parents[1] / "examples/terminal-result.json"
    )
    assert cli.main(["report", "--input", str(path), "--format", "text"]) == 0
    output = capsys.readouterr().out
    assert "synthetic-terminal-example" in output
    assert "service_view" in output and "Request Throughput" in output
    assert "2.941" in output
    assert "weight_sum=" not in output


@pytest.mark.parametrize("text", ["invalid JSON", "{}", '{"kind": "run"}'])
def test_invalid_result_failure_is_logged(tmp_path, capsys, text):
    path = tmp_path / "result.json"
    path.write_text(text)
    assert cli.main(["report", "--input", str(path), "--format", "text"]) == 2
    assert "CLI command failed" in capsys.readouterr().err


def test_run_default_prints_summary(result, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_plan", lambda *a, **k: {
        "tasks": [], "case_count": 0,
    })
    monkeypatch.setattr(cli.RunCoordinator, "run", lambda *a, **k: (
        result, tmp_path,
    ))
    assert cli.main(["run"]) == 0
    text = capsys.readouterr().out
    assert "终端结果摘要" in text and "重新查看/Details:" in text
    assert (tmp_path / "report.html").is_file()


@pytest.mark.parametrize("no_summary", [False, True])
@pytest.mark.parametrize("mode", ["ok", "malformed"])
@pytest.mark.parametrize("no_progress", [False, True])
@pytest.mark.parametrize("log_level", ["info", "error"])
def test_real_cli_pipeline_preserves_saved_results(
    tmp_path, monkeypatch, capsys, no_summary, mode, no_progress, log_level,
):
    # Real sockets and worker process, synthetic responses (no GPU/model).
    from tests.test_http_connection_pool import PoolServer

    monkeypatch.setenv("NO_PROXY", "*")
    server = PoolServer()
    server.mode = mode
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"prompt":"hello"}\n' * 3)
    try:
        arguments = [
            "run", "--scenario", "online_serving", "--model", "mock",
            "--backend", "openai_http", "--base-url", server.url,
            "--data", "jsonl", "--dataset-path", str(dataset),
            "--output-tokens", "2", "--num-requests", "3",
            "--num-warmups", "1" if mode == "ok" else "0",
            "--profile", "synchronous",
            "--request-timeout", "3", "--task-timeout", "30",
            "--result-dir", str(tmp_path / "runs"),
            "--log-level", log_level,
        ]
        if no_summary:
            arguments.append("--no-summary")
        if no_progress:
            arguments.append("--no-progress")
        assert cli.main(arguments) == (0 if mode == "ok" else 1)
    finally:
        server.release.set()
        server.stalled.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    captured = capsys.readouterr()
    output = captured.out
    assert "准备完成" not in output
    assert ".prepared:" not in output
    assert '"backend":' not in output
    assert ("当前任务：" in captured.err) is not no_progress
    assert "\x1b" not in captured.err
    assert ("终端结果摘要" in output) is not no_summary
    if no_summary:
        assert output == ""
    path, = (tmp_path / "runs").glob("*/result.json")
    log_path, = (path.parent / "log").glob("*.log")
    log_text = log_path.read_text()
    for message in ("Run plan: tasks=1 cases=1", "Resolved Case:",
                    "Case prepared:", "HTML report saved:"):
        assert (message in log_text) is (log_level == "info")
    assert "Resolved Case:" not in captured.err
    assert "Case prepared:" not in captured.err
    assert "Case plan:" not in captured.err
    if log_level == "info":
        assert "task=task_1 case=case_1 phase=prepare" in log_text
        assert str(path) in log_text and str(log_path) in log_text
    before = path.read_bytes()
    result = load_result(path)
    assert output == ("" if no_summary else
                      render_text(result, result_path=path) + "\n")
    assert result["status"] == ("success" if mode == "ok" else "failed")
    case = result["tasks"][0]["cases"][0]
    assert case["coverage"]["complete" if mode == "ok" else "failed"] == 3
    if mode != "ok" and not no_summary:
        assert "定位异常:" in output
        assert "--case task_1/case_1" in output
    assert "no_summary" not in case["parameters"]
    assert "no_progress" not in case["parameters"]
    raw = Path(case["raw"])
    raw_before = raw.read_bytes()
    assert len(json.loads(raw_before)["requests"]) == 3
    assert (path.parent / "report.html").is_file()
    assert set(case["metrics"]) == {
        "streaming_performance", "streaming_phase_timing",
    }
    metric = case["metrics"]["streaming_performance"]
    case_path = raw.with_name(raw.stem + ".result.json")
    assert json.loads(case_path.read_text())["metrics"][
        "streaming_performance"
    ] == metric
    run_metric = result["metrics"]["token_throughput"]
    for key, node in run_metric["values"]["service_view"].items():
        assert node == metric["values"]["service_view"][key]
    assert run_metric["metadata"]["included_cases"][0]["case"] == case["id"]
    assert metric["values"]["service_view"]["successful_requests"][
        "value"
    ] == (
        3 if mode == "ok" else 0
    )
    assert result["duration_s"] > 0
    if mode == "ok":
        view = metric["values"]["service_view"]
        assert view["total_tokens"]["value"] > 0
        assert view["benchmark_duration"]["value"] == (
            case["window"]["duration_s"]
        )
    assert cli.main(["report", "--input", str(path), "--format", "text",
                     "--all", "--output", str(tmp_path / "report.txt")]) == 0
    assert path.read_bytes() == before
    assert raw.read_bytes() == raw_before
    assert cli.main([
        "report", "--input", str(path), "--format", "text",
        "--details", "requests" if mode == "ok" else "errors",
    ]) == 0
    details_output = capsys.readouterr().out
    assert "已保存请求 3" in details_output
    assert "Request ID" in details_output
    assert "request-0" in details_output
    if mode == "ok":
        assert "raw_index=" not in details_output
    else:
        assert "#1 [task_1/case_1] request" in details_output
    assert path.read_bytes() == before
    assert raw.read_bytes() == raw_before


def test_eighty_columns_have_framed_metric_rows(result):
    text = render_text(result, width=80)
    lines = text.splitlines()
    ttft = next(line for line in lines if line.startswith("| TTFT "))
    assert [cell.strip() for cell in ttft.split("|")[1:-1]] == [
        "TTFT 首Token延迟", "48.67", "46.356", "59.401", "ms", "50",
    ]
    frames = [line for line in lines if line.startswith(("|", "+"))]
    assert frames
    assert all(display_width(line) == 80 for line in frames)
    assert "\x1b" not in text


def test_long_numeric_values_are_not_truncated(result):
    metric = result["tasks"][0]["cases"][0]["metrics"]
    metric["streaming_performance"]["values"]["request_view"]["ttft"][
        "mean"
    ] = 123456789.123456
    text = render_text(result, width=80)
    rows = text.splitlines()
    start = next(i for i, line in enumerate(rows)
                 if line.startswith("| TTFT "))
    assert rows[start].split("|")[2].strip() == "123456789.123"
    assert metric["streaming_performance"]["values"]["request_view"][
        "ttft"
    ]["mean"] == 123456789.123456


@pytest.mark.parametrize("verbose", [False, True])
def test_cli_verbose_filter_and_redirect(result, tmp_path, capsys, verbose):
    path = tmp_path / "result with spaces.json"
    path.write_text(json.dumps(result))
    before = path.read_bytes()
    arguments = ["report", "--input", str(path), "--format", "text",
                 "--case", "task_1/case_1"]
    if verbose:
        arguments.append("--verbose")
    assert cli.main(arguments) == 0
    text = capsys.readouterr().out
    assert ("weight_sum=750" in text) is verbose
    assert ("started_at=100" in text) is verbose
    assert ("吞吐量范围" in text) is verbose
    assert "--verbose --all --case task_1/case_1" in text
    output = tmp_path / "output.txt"
    assert cli.main(arguments + ["--output", str(output)]) == 0
    assert output.read_text() == text
    assert path.read_bytes() == before
    assert cli.main(["report", "--input", str(path), "--verbose"]) == 2


def test_dynamic_metrics_do_not_invent_missing_values(result):
    metric = result["tasks"][0]["cases"][0]["metrics"]["streaming_performance"]
    metric["values"] = {"request_view": {
        "novel_latency": {"mean": 0, "p95": 7, "unit": "ms", "count": 2},
        "p95_only": {"p95": 9, "unit": "ms", "count": 2},
    }}
    text = render_text(result, width=120)
    assert "novel_latency" in text and "p95_only" in text
    assert "最低 TTFT P99: N/A" in text
    assert "最低 TPOT P99: N/A" in text
    assert "service_view" not in text
    rows = [line for line in text.splitlines() if "| novel_latency" in line]
    assert rows[0].split("|")[2].strip() == "0"
    assert "0" in rows[0] and "7" in rows[0]


@pytest.mark.parametrize("scenario", ["ceval", "cmmlu"])
def test_choice_processor_results_are_rendered_dynamically(result, scenario):
    from luban_meter.benchmarking.model_service_quality.common import (
        choice_result,
    )

    # Domain scorer, simulated samples; no service/model was called.
    processed = choice_result.process_choice_result({
        "status": "success", "metrics": {
            "samples": [
            {"status": "success", "reference": "A", "prediction": "A",
             "correct": True, "subject": "math", "input_tokens": 10,
             "output_tokens": 1},
            {"status": "success", "reference": "A", "prediction": "B",
             "correct": False, "subject": "physics", "input_tokens": 10,
             "output_tokens": 1},
            {"status": "service_failed", "reference": "A"},
        ],
    }}, scenario)
    task = result["tasks"][0]
    task.update(scenario=scenario, category="model_service_quality")
    task["cases"][0]["metrics"] = {"choice_accuracy": {
        "status": processed["status"], "values": processed["metrics"],
        "error": processed["error"],
    }}
    text = render_text(result, width=140)
    assert "Accuracy" in text and "0.5" in text
    assert "accuracy_by_subject" not in text
    assert "math" not in text and "physics" not in text
    assert "已隐藏 2 项学科/子类别明细" in text
    detailed = render_text(result, width=140, verbose=True)
    assert "accuracy_by_subject" in detailed
    assert "math" in detailed and "physics" in detailed
    assert "service_failed=1 sample" in " ".join(text.split())
    assert "TTFT" not in text and "service_view" not in text
    html = report_data(result)["rows"]
    row = next(r for r in html if r["metric"].endswith("/accuracy"))
    assert (row["value"], row["unit"], row["count"]) == (0.5, "ratio", 2)


def test_historical_real_gsm8k_result_matches_html():
    path = Path(__file__).parents[1] / "examples/terminal-gsm8k-result.json"
    original = path.read_bytes()
    result = load_result(path)
    for verbose in (False, True):
        text = render_text(result, verbose=verbose, width=140)
        assert "Exact Match" in text and "0.2" in text
        assert "request_view" not in text and "TTFT" not in text
        line = next(line for line in text.splitlines()
                    if "| Exact Match" in line)
        assert line.split("|")[-2].strip() == "5"
    row = next(r for r in report_data(result)["rows"]
               if r["metric"].endswith("/exact_match"))
    assert (row["value"], row["unit"], row["count"]) == (0.2, "ratio", 5)
    assert path.read_bytes() == original


def test_arbitrary_quality_scores_and_mixed_cases(result):
    task = copy.deepcopy(result["tasks"][0])
    task.update(name="quality", scenario="custom_quality",
                category="model_service_quality")
    task["cases"][0]["metrics"] = {"quality": {"values": {"task_view": {
        "custom": {
            "f1": {"value": 0.3, "unit": "ratio", "count": 2},
            "pass_at_5": {"value": 0.7, "unit": "ratio", "count": 3},
            "new_score": {"value": 0, "unit": "ratio", "count": 4},
        },
    }}}}
    result["tasks"].append(task)
    text = render_text(result, cases=["quality/case_1"], width=120)
    assert "f1" in text and "pass_at_5" in text and "new_score" in text
    assert "TTFT" not in text
    all_text = render_text(result, all_cases=True)
    assert "[task_1/case_1]" in all_text and "[quality/case_1]" in all_text


def test_failure_navigation_preserves_scope_and_original(result, tmp_path):
    case = result["tasks"][0]["cases"][0]
    case.update(status="partial_failed", error={"message": "request timeout"})
    case["coverage"].update(complete=48, failed=2)
    before = copy.deepcopy(result)
    path = tmp_path / "result.json"
    for verbose in (False, True):
        text = render_text(result, result_path=path, verbose=verbose)
        assert "failed=2" in text and "request timeout" in text
        assert "--verbose --case task_1/case_1" in text
        assert "--include-requests" in text
        assert "--errors" not in text
    assert result == before


def test_legacy_versioned_result_rejected_without_rewriting(tmp_path, capsys):
    path = tmp_path / "legacy.json"
    path.write_text('{"schema_version": "1", "metrics": {}}')
    before = path.read_bytes()
    assert cli.main(["report", "--format", "text", "--input", str(path)]) == 2
    assert "unsupported legacy/versioned" in capsys.readouterr().err
    assert path.read_bytes() == before


def test_example_token_weights_match_existing_calculator():
    from luban_meter.benchmarking.generation_performance import online_serving
    from tests.test_serving_guidellm import case_result, record

    result = case_result(online_serving, [
        dict(record(tokens=16, first=20, last=320, end=320),
             input_tokens=64, itl_samples_ms=[20] * 15)
        for _ in range(50)
    ])
    example = load_result(Path(__file__).parents[1]
                          / "examples/terminal-result.json")
    saved = example["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["request_view"]
    for key, weight in (("itl", 750), ("tpot", 800)):
        actual = result["request_view"][key]
        assert actual["weight_sum"] == weight
        for field in ("mean", "p50", "p95", "count", "unit", "weight_sum",
                      "aggregation", "percentile_method"):
            assert actual[field] == saved[key][field]


def test_default_is_shorter_without_hiding_metric_values(result):
    plain = render_text(result, width=100)
    verbose = render_text(result, width=100, verbose=True)
    assert len(plain.splitlines()) < len(verbose.splitlines())
    for text in (plain, verbose):
        for value in ("48.67", "46.356", "59.401", "1.667"):
            assert value in text
    assert "weight_sum" not in plain and "weight_sum=750" in verbose


def test_runtime_error_is_not_a_metric(result):
    result["tasks"][0]["cases"][0]["metrics"] = {
        "failed_metric": {"status": "failed", "error": {
            "code": 503, "message": "service unavailable",
        }},
    }
    text = render_text(result)
    assert "service unavailable" in text
    assert "| code" not in text


@pytest.mark.parametrize("columns", [80, 100, 120, 230])
@pytest.mark.parametrize("verbose", [False, True])
def test_terminal_size_caps_tables_and_aligns_chinese(result, monkeypatch,
                                                    columns, verbose):
    monkeypatch.setenv("COLUMNS", str(columns))
    text = render_text(result, verbose=verbose)
    frames = [line for line in text.splitlines()
              if line.startswith(("+", "|"))]
    assert frames
    assert {display_width(line) for line in frames} == {min(columns, 120)}


def test_unknown_performance_metrics_remain_available_in_verbose(result):
    view = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["request_view"]
    view["new_latency"] = {"mean": 123.45678, "unit": "ms", "count": 50}
    original = copy.deepcopy(result)
    text = render_text(result)
    assert "new_latency" not in text and "--verbose 查看全部" in text
    detail = render_text(result, verbose=True)
    assert "new_latency" in detail and "123.457" in detail
    assert result == original


def test_scalar_quality_layout_and_duration_rounding():
    path = Path(__file__).parents[1] / "examples/terminal-gsm8k-result.json"
    payload = load_result(path)
    before = path.read_bytes()
    text = render_text(payload, width=80)
    assert "duration_s=13.251 s" in text
    assert "P50" not in text and "P95" not in text
    line = next(line for line in text.splitlines() if "| Exact Match" in line)
    assert [cell.strip() for cell in line.split("|")[2:-1]] == [
        "0.2", "ratio", "5",
    ]
    assert "20%" not in text
    assert path.read_bytes() == before


def test_failed_metric_in_success_case_has_navigation(result, tmp_path):
    result["tasks"][0]["cases"][0]["metrics"]["custom"] = {
        "status": "failed", "error": "scoring failed",
    }
    text = render_text(result, result_path=tmp_path / "result.json")
    assert "scoring failed" in text and "定位异常:" in text
    assert "--case task_1/case_1" in text


def test_breakdown_filter_preserves_extensions_diagnostics_and_data(result):
    import copy

    result["tasks"][0]["cases"][0]["metrics"] = {"custom": {
        "status": "success", "values": {
            "accuracy": {"value": 0.75, "unit": "ratio", "count": 4},
            "accuracy_by_category": {"science": {
                "value": 0.5, "unit": "ratio", "count": 2,
                "reason": "category diagnostic",
            }},
            "extension": {"nested_score": {
                "value": 0.123, "unit": "ratio", "count": 4,
            }},
        },
    }}
    original = copy.deepcopy(result)
    plain = render_text(result)
    detail = render_text(result, verbose=True)
    assert "已隐藏 1 项学科/子类别明细" in plain
    assert "category diagnostic" in plain
    assert "nested_score" in plain and "0.123" in plain
    assert "science" in detail and "0.5" in detail
    assert result == original


@pytest.mark.parametrize("width", [40, 76, 80, 100, 120])
@pytest.mark.parametrize("verbose", [False, True])
def test_e2el_percentiles_remain_whole(result, width, verbose):
    view = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["request_view"]
    view["e2el"] = {
        "mean": 1497.01, "p50": 1484.211, "p95": 1591.317,
        "unit": "ms", "count": 20,
    }
    original = copy.deepcopy(result)
    text = render_text(result, width=width, verbose=verbose)
    assert "1484.211" in text and "1591.317" in text
    if width >= 76:
        row = next(line for line in text.splitlines()
                   if line.startswith("| E2EL "))
        assert row.split("|")[3].strip() == "1484.211"
        assert row.split("|")[4].strip() == "1591.317"
    frames = [line for line in text.splitlines()
              if line.startswith(("+", "|"))]
    assert all(display_width(line) == width for line in frames)
    assert result == original


def test_wide_values_fall_back_without_splitting(result):
    view = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["request_view"]
    view["e2el"] = {
        "mean": 123456789.123, "p50": 123456788.123,
        "p95": 123456790.123, "unit": "ms", "count": 123456789,
    }
    text = render_text(result, width=76)
    for value in ("123456789.123", "123456788.123", "123456790.123"):
        assert value in text
    assert all(display_width(line) == 76 for line in text.splitlines()
               if line.startswith(("+", "|")))


def test_summary_reads_saved_throughput_without_recalculating(result):
    case = result["tasks"][0]["cases"][0]
    case["metrics"] = {"token_throughput": {
        "status": "success", "values": {"service_view": {
            "total_token_throughput": {"value": 151, "unit": "token/s"},
        }},
    }}
    case.update(total_tokens=300, window={"duration_s": 2})
    result.update(started_at=100, finished_at=110, duration_s=11)
    result["metrics"] = {"token_throughput": {
        "status": "success", "values": {"service_view": {
            "total_tokens": {"value": 400, "unit": "token"},
            "benchmark_duration": {"value": 4, "unit": "s"},
            "total_token_throughput": {"value": 101, "unit": "token/s"},
        }},
    }}
    original = copy.deepcopy(result)
    text = render_text(result, cases=(case["id"],), width=120)
    assert "整轮 duration=11 s" in text
    assert "正式测试时长=4 s" in text
    assert "性能评测总体 total_token_throughput=101 tokens/s" in text
    assert "total_token_throughput=151 tokens/s" in text
    assert result == original
    del result["metrics"]
    assert "性能评测总体 total_token_throughput=N/A" in render_text(result)


def test_request_token_totals_use_actual_usage_once():
    from luban_meter.core.token_totals import request_total

    requests = [
        {"sample_id": "s", "status": "success",
         "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"sample_id": "s", "status": "success",
         "usage": {"prompt_tokens": 9, "completion_tokens": 0}},
        {"sample_id": "s", "status": "failed"},
    ]
    samples = [{"sample_id": "s", "facts": {"status": "parse_failed"}}]
    assert request_total(requests, samples) == 21
    del requests[1]["usage"]
    assert request_total(requests, samples) is None


def test_run_aggregation_excludes_skips_and_rejects_incomplete_cases():
    from luban_meter.core.token_totals import run_throughput, throughput_result

    cases = [
        {"status": "success", "collection_complete": True,
         "metrics": {"token_throughput": throughput_result(300, 2)}},
        {"status": "partial_failed", "collection_complete": True,
         "metrics": {"token_throughput": throughput_result(100, 1)}},
        {"status": "skipped"},
    ]
    tasks = [{"status": "partial_failed", "cases": cases},
             {"status": "skipped", "cases": []}]
    for task in tasks:
        task["category"] = "generation_performance"
    aggregated = run_throughput(tasks)
    assert aggregated["values"] == throughput_result(400, 3)["values"]
    assert len(aggregated["metadata"]["included_cases"]) == 2
    assert len(aggregated["metadata"]["skipped_cases"]) == 2
    cases[0]["status"] = "discarded"
    assert run_throughput(tasks)["status"] == "unavailable"


@pytest.mark.parametrize("successful,failed", [(20, 0), (0, 20)])
def test_saved_request_metadata_in_verbose_report(result, successful, failed):
    from luban_meter.core.token_totals import throughput_result

    case = result["tasks"][0]["cases"][0]
    case["metrics"] = {"token_throughput": throughput_result(
        100, 2, metadata={
            "successful_requests": successful, "failed_requests": failed,
        },
    )}
    before = copy.deepcopy(result)
    text = render_text(result, verbose=True)
    assert f"successful_requests={successful}" in text
    assert f"failed_requests={failed}" in text
    assert "total_requests=N/A" not in text
    assert result == before


@pytest.mark.parametrize("end_wall", [90, 1000])
def test_run_duration_ignores_wall_clock_changes(tmp_path, monkeypatch,
                                                end_wall):
    from types import SimpleNamespace
    from luban_meter.core import run_coordinator

    wall = iter([100, end_wall])
    monotonic = iter([10, 35])
    monkeypatch.setattr(run_coordinator, "time", SimpleNamespace(
        time=lambda: next(wall),
        perf_counter=lambda: next(monotonic),
    ))
    result, directory = run_coordinator.RunCoordinator().run(
        {"tasks": []}, tmp_path,
    )
    assert result["duration_s"] == 25
    assert result["started_at"] == 100
    assert result["finished_at"] == end_wall
    saved = json.loads((directory / "result.json").read_text())
    assert saved["duration_s"] == 25


def test_quality_report_has_no_throughput(result):
    task = result["tasks"][0]
    task["category"] = "model_service_quality"
    task["cases"][0]["metrics"] = {}
    text = render_text(result, verbose=True)
    assert "total_token_throughput" not in text
    assert "正式测试时长" not in text
    assert "整轮 duration=" in text


def _with_goodput(result, *, rate, req, out, undetermined=0):
    task = result["tasks"][0]
    task["cases"][0]["metrics"]["streaming_performance"]["values"][
        "service_view"
    ]["goodput"] = {
        "status": "applicable",
        "slo_config": {"ttft_ms": 100, "tpot_ms": 5, "e2el_ms": 2000},
        "slo_satisfied_count": {"value": 8, "unit": "request"},
        "slo_violated_count": {"value": 1, "unit": "request"},
        "failed_count": {"value": 1, "unit": "request"},
        "undetermined_count": {"value": undetermined, "unit": "request"},
        "determined_count": {"value": 10, "unit": "request"},
        "slo_satisfied_rate": {"value": rate, "unit": "ratio"},
        "goodput_request_throughput": {"value": req, "unit": "req/s"},
        "goodput_output_token_throughput": {
            "value": out, "unit": "token/s"
        },
    }
    return result


def test_goodput_block_summarizes_slo_attainment_and_throughputs(result):
    payload = _with_goodput(
        copy.deepcopy(result), rate=0.8, req=5, out=1000
    )
    text = render_text(payload, width=120)
    assert "SLO 阈值: TTFT≤100ms TPOT≤5ms E2EL≤2000ms" in text
    assert "SLO 达标率: 80%" in text
    assert "满足=8" in text and "违反=1" in text and "失败=1" in text
    assert "无法判定=0" in text and "分母=10" in text
    assert "Goodput: 请求吞吐=5 req/s" in text
    assert "输出Token吞吐=1000 token/s" in text
    # The compact block replaces the raw field rows in this mode.
    assert "service_view/goodput/" not in text
    assert "slo_satisfied_rate" not in text
    verbose = render_text(payload, width=120, verbose=True)
    assert "SLO 达标率" in verbose
    assert "service_view/goodput" in verbose


def test_goodput_block_marks_undeterminable_values(result):
    payload = _with_goodput(
        copy.deepcopy(result), rate=None, req=None, out=None,
        undetermined=50,
    )
    payload["tasks"][0]["cases"][0]["metrics"]["streaming_performance"][
        "values"
    ]["service_view"]["goodput"]["determined_count"] = {"value": 0}
    text = render_text(payload, width=120)
    assert "SLO 达标率: N/A（无已判定请求）" in text
    assert "无法判定=50" in text
    assert "请求吞吐=N/A" in text
    assert "输出Token吞吐=N/A" in text
    assert "有达标请求缺输出Token计数" in text


def test_no_goodput_block_without_configuration(result):
    text = render_text(result, width=120)
    assert "SLO 阈值" not in text
    assert "Goodput:" not in text
    assert "SLO 达标率" not in text
