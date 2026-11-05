"""Best summaries use saved values and explicit comparison scope."""

import copy

import pytest

from luban_meter.reporting.best_summary import (
    performance_lines, quality_lines,
)
from luban_meter.reporting.data import load_result
from luban_meter.reporting.terminal import render_text
from luban_meter.utils.json_io import write_json_atomic


def record(index=1, rate=10, status="success"):
    return {
        "id": f"case_{index}", "status": status,
        "collection_complete": True, "metrics_status": "complete",
        "parameters": {"max_concurrency": index},
        "metrics": {"custom_performance": {
            "status": "success", "values": {
                "request_view": {
                    "ttft": {"p99": 0.02, "unit": "s", "count": 20},
                    "tpot": {"p95": 1, "unit": "ms/token"},
                },
                "service_view": {
                    "output_token_throughput": {
                        "value": rate, "unit": "token/s"},
                    "total_token_throughput": {
                        "value": rate * 2, "unit": "token/s"},
                    "total_requests": {"value": 20, "unit": "request"},
                    "successful_requests": {"value": 20, "unit": "request"},
                    "failed_requests": {"value": 0, "unit": "request"},
                },
            },
        }},
    }


def task(category="generation_performance"):
    return {"name": "task", "category": category,
            "status": "success", "cases": [], "kind": "task"}


def test_saved_extremes_ties_missing_p99_and_failed_cases():
    t = task()
    rows = [record(), record(2), record(3, 999, "failed")]
    before = copy.deepcopy(rows)
    text = "\n".join(performance_lines([(t, c) for c in rows]))
    assert "最大输出 Token 吞吐量: 10 token/s" in text
    assert "最大总 Token 吞吐量（输入＋输出）: 20 token/s" in text
    assert "最低 TTFT P99: 20 ms" in text
    assert "最低 TPOT P99: N/A" in text
    assert "2 个 Case 并列" in text
    assert "来源 task/case_1" in text
    assert rows == before


def test_filters_apply_before_best_but_display_limit_does_not():
    t = task()
    t["cases"] = [record(i, i * 10) for i in range(1, 7)]
    text = render_text(t)
    assert "最大输出 Token 吞吐量: 60 token/s" in text
    assert "省略 1 个 Case" in text
    text = render_text(t, cases=("case_1",))
    assert "最大输出 Token 吞吐量: 10 token/s" in text
    assert "最大输出 Token 吞吐量: 60" not in text


@pytest.mark.parametrize("wrapped", [False, True])
def test_quality_protocol_groups_directions_and_hidden_subjects(wrapped):
    t = task("model_service_quality")
    records = []
    for i, score in enumerate((0.4, 0.8, 0.9)):
        c = record(i + 1)
        c["parameters"]["dataset_path"] = "a" if i < 2 else "b"
        c["metrics"] = {"accuracy": {"values": {"task_view": {
            "accuracy": {"value": score, "unit": "ratio", "count": 20},
            "perplexity": {"value": 10 - i, "unit": ""},
            "accuracy_by_subject": {
                "subject": {"value": 1, "unit": "ratio"},
            },
        }}}}
        if not wrapped:
            c["metrics"]["accuracy"] = c["metrics"]["accuracy"]["values"]
        records.append((t, c))
    text = "\n".join(quality_lines(records))
    assert "accuracy: 80 %" in text
    assert "accuracy: 90 %" in text
    assert "perplexity: 9 " in text
    assert "条件组 2" in text
    assert "subject" not in text
    records[1][1]["status"] = "failed"
    text = "\n".join(quality_lines(records))
    assert "accuracy: 40 %" in text
    assert "accuracy: 80 %" not in text


@pytest.mark.parametrize("score,count,status,error,expected", [
    (0, 2, "success", None, "accuracy: 0 %"),
    (0.5, 0, "success", None, "最佳得分: N/A"),
    (None, 2, "success", None, "最佳得分: N/A"),
    (1.1, 2, "success", None, "最佳得分: N/A"),
    (0.5, 2, "partial_failed", None, "最佳得分: N/A"),
    (0.5, 2, "success", "scoring failed", "最佳得分: N/A"),
])
def test_top_level_score_keeps_zero_count_and_status_filters(
        score, count, status, error, expected):
    t, c = task("model_service_quality"), record()
    c["coverage"] = {"complete": 100}
    c["metrics"] = {"quality": {
        "status": status, "error": error,
        "accuracy": {"value": score, "unit": "ratio", "sample_count": count},
    }}
    assert expected in "\n".join(quality_lines([(t, c)]))


@pytest.mark.parametrize("request_view", ["missing", None, {}])
@pytest.mark.parametrize("provider", [
    "streaming_performance", "multi_turn_performance", "custom_performance",
])
def test_persisted_service_only_metrics_reach_terminal(
        tmp_path, request_view, provider):
    t, c = task(), record(4)
    metric = c["metrics"].pop("custom_performance")
    metric["values"].pop("request_view")
    if request_view != "missing":
        metric["values"]["request_view"] = request_view
    c["metrics"][provider] = metric
    t["cases"] = [c]
    destination = tmp_path / "result.json"
    write_json_atomic(destination, t)
    saved = load_result(destination)
    before = copy.deepcopy(saved)
    text = render_text(saved, width=120)
    assert "最大输出 Token 吞吐量: 10 token/s" in text
    assert "最大总 Token 吞吐量（输入＋输出）: 20 token/s" in text
    assert "最高请求成功率: 100 %" in text
    assert "最大已测配置并发: 4 并发" in text
    for latency in ("TTFT", "TPOT", "E2E"):
        assert f"最低 {latency} P99: N/A" in text
    assert saved == before


@pytest.mark.parametrize("current,legacy_status,legacy_error,expected", [
    (12, "success", None, "12 token/s"),
    (0, "success", None, "0 token/s"),
    (None, "success", None, "N/A"),
    ("missing", "success", None, "999 token/s"),
    ("missing", "failed", None, "N/A"),
    ("missing", "success", "failed provider", "N/A"),
])
def test_total_throughput_uses_saved_provider_priority(
        tmp_path, current, legacy_status, legacy_error, expected):
    t, c = task(), record()
    service = c["metrics"]["custom_performance"]["values"]["service_view"]
    if current == "missing":
        del service["total_token_throughput"]
    else:
        service["total_token_throughput"]["value"] = current
    c["metrics"]["token_throughput"] = {
        "status": legacy_status, "error": legacy_error,
        "values": {"service_view": {
            "total_token_throughput": {"value": 999, "unit": "token/s"},
        }},
    }
    t["cases"] = [c]
    destination = tmp_path / "result.json"
    write_json_atomic(destination, t)
    text = render_text(load_result(destination), width=120)
    assert f"最大总 Token 吞吐量（输入＋输出）: {expected}" in text


def test_service_only_keeps_selected_provider_status_and_boundary():
    t, c = task(), record()
    metric = c["metrics"]["custom_performance"]
    metric["values"].pop("request_view")
    c["metrics"]["z_valid"] = copy.deepcopy(metric)
    metric["error"] = "selected provider failed"
    text = "\n".join(performance_lines([(t, c)]))
    assert "最大输出 Token 吞吐量: N/A" in text
    assert "最高请求成功率: N/A" in text
    assert "最大已测配置并发: N/A" in text
    c["metrics"] = {"engine_performance": {"values": {
        "engine_view": {"internal_ttft": {"p99": 1, "unit": "ms"}},
        "service_view": {
            "total_token_throughput": {"value": 999, "unit": "token/s"},
        },
    }}}
    text = "\n".join(performance_lines([(t, c)]))
    assert "最大输出 Token 吞吐量: N/A" in text
    assert "最低 TTFT P99: N/A" in text


@pytest.mark.parametrize("value,unit,expected", [
    (0, "token/s", "0 token/s"),
    (None, "token/s", "N/A"),
    (-1, "token/s", "N/A"),
    ("10", "token/s", "N/A"),
    (10, "req/s", "N/A"),
])
def test_service_only_missing_and_invalid_output_preserves_request_counts(
        value, unit, expected):
    t, c = task(), record()
    metric = c["metrics"]["custom_performance"]
    metric["values"].pop("request_view")
    service = metric["values"]["service_view"]
    service["output_token_throughput"] = {"value": value, "unit": unit}
    text = "\n".join(performance_lines([(t, c)]))
    assert f"最大输出 Token 吞吐量: {expected}" in text
    assert "最高请求成功率: 100 %" in text


@pytest.mark.parametrize("service_only", [False, True])
def test_incomplete_and_nonfinite_performance_do_not_win(service_only):
    t = task()
    first, second = record(), record(2, float("inf"))
    first["collection_complete"] = False
    if service_only:
        for c in (first, second):
            c["metrics"]["custom_performance"]["values"].pop("request_view")
    text = "\n".join(performance_lines([(t, first), (t, second)]))
    assert "最大输出 Token 吞吐量: N/A" in text
    assert "最大总 Token 吞吐量（输入＋输出）: N/A" in text


@pytest.mark.parametrize("service_only", [False, True])
def test_success_rate_includes_failed_requests(service_only):
    t, c = task(), record(status="partial_failed")
    if service_only:
        c["metrics"]["custom_performance"]["values"].pop("request_view")
    view = c["metrics"]["custom_performance"]["values"]["service_view"]
    view["successful_requests"]["value"] = 15
    view["failed_requests"]["value"] = 5
    text = "\n".join(performance_lines([(t, c)]))
    assert "最高请求成功率: 75 %" in text
    assert "最大输出 Token 吞吐量: N/A" in text
    view["failed_requests"]["value"] = 4
    text = "\n".join(performance_lines([(t, c)]))
    assert "最高请求成功率: N/A" in text
