"""Terminal reports preserve saved statistics and execution semantics."""

import copy
import json
import threading
from pathlib import Path

import pytest

from luban_meter import cli
from luban_meter.reporting.data import load_result, metric_rows
from luban_meter.reporting.html import report_data
from luban_meter.reporting.terminal import (
    display_width,
    distribution_values,
    render_text,
)


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


def test_text_cli_stdout_and_file_leave_json_unchanged(result, tmp_path,
                                                     capsys):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    original = path.read_bytes()
    assert cli.main(["report", "--input", str(path), "--format", "text"]) == 0
    assert "LuBan-Meter 终端结果摘要" in capsys.readouterr().out
    output = tmp_path / "text" / "report.txt"
    assert cli.main(["report", "--input", str(path), "--format", "text",
                     "--output", str(output)]) == 0
    assert output.is_file()
    assert path.read_bytes() == original
    assert not (tmp_path / "report.html").exists()
    assert load_result(path) == result
    assert cli.main(["report", "--input", str(path), "--format", "text",
                     "--output", str(path)]) == 2
    assert path.read_bytes() == original


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
    assert "error:" in capsys.readouterr().err


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
def test_invalid_result_is_reported_without_traceback(tmp_path, capsys, text):
    path = tmp_path / "result.json"
    path.write_text(text)
    assert cli.main(["report", "--input", str(path), "--format", "text"]) == 2
    assert "error:" in capsys.readouterr().err


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
def test_real_cli_pipeline_preserves_saved_results(
    tmp_path, monkeypatch, capsys, no_summary, mode, no_progress,
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
    assert ("进度 |" in captured.err) is not no_progress
    assert "\x1b" not in captured.err
    assert ("终端结果摘要" in output) is not no_summary
    path, = (tmp_path / "runs").glob("*/result.json")
    before = path.read_bytes()
    result = load_result(path)
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
    chunks = [rows[start + offset].split("|")[2].strip()
              for offset in (0, 1)]
    assert "".join(chunks) == "123456789.123"
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
    assert "TTFT" not in text and "TPOT" not in text
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
    assert "accuracy_by_subject" in text
    assert "math" in text and "physics" in text
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
