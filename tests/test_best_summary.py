"""Best summaries use saved values and explicit comparison scope."""

import copy

from luban_meter.reporting.best_summary import (
    performance_lines, quality_lines,
)
from luban_meter.reporting.terminal import render_text


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


def test_quality_protocol_groups_directions_and_hidden_subjects():
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



def test_incomplete_and_nonfinite_performance_do_not_win():
    t = task()
    first, second = record(), record(2, float("inf"))
    first["collection_complete"] = False
    text = "\n".join(performance_lines([(t, first), (t, second)]))
    assert "最大输出 Token 吞吐量: N/A" in text
    assert "最大总 Token 吞吐量（输入＋输出）: N/A" in text


def test_success_rate_includes_failed_requests():
    t, c = task(), record(status="partial_failed")
    view = c["metrics"]["custom_performance"]["values"]["service_view"]
    view["successful_requests"]["value"] = 15
    view["failed_requests"]["value"] = 5
    text = "\n".join(performance_lines([(t, c)]))
    assert "最高请求成功率: 75 %" in text
    assert "最大输出 Token 吞吐量: N/A" in text
    view["failed_requests"]["value"] = 4
    text = "\n".join(performance_lines([(t, c)]))
    assert "最高请求成功率: N/A" in text
