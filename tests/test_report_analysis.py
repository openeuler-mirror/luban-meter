"""Synthetic display fixtures, separate from real evaluation results."""

import json
import sys
from pathlib import Path

from luban_meter.reporting.html import write_html


def scalar(value, unit="ratio", count=10):
    return {"value": value, "unit": unit, "sample_count": count}


def browser_fixture():
    distribution = {
        "mean": None, "p50": 0, "p90": 2, "p999": 4,
        "min": 0, "max": 5, "stddev": 1, "count": 10, "unit": "ms",
    }
    performance = {
        "kind": "task", "name": "duplicate", "scenario": "online_serving",
        "status": "partial_failed", "category": "generation_performance",
        "cases": [
            {"id": "case_1", "status": "partial_failed",
             "coverage": {"complete": 8, "failed": 1, "incomplete": 1},
             "metrics": {"streaming_performance": {"values": {"request_view": {
                 "ttft": distribution,
                 "unknown_latency": {"p999": 0.001, "unit": "s"},
                 "no_value": scalar(None, "req/s"),
             }}}}},
            {"id": "empty", "status": "failed", "error": "timeout"},
            {"id": "discarded", "status": "discarded",
             "metrics": {"unused": scalar(999)}},
            {"id": "fallback", "status": "success", "metrics": {
                "streaming_performance": {"values": {
                    "request_view": {"tpot": distribution,
                                     "request_throughput": scalar(999)},
                    "service_view": {"request_throughput": scalar(2, "req/s")},
                }},
                "engine": {"values": {"request_view": {
                    "ttft": distribution}}},
            }},
            {"id": "similar_only", "status": "success", "metrics": {
                "other": {"values": {"request_view": {
                    "ttft": distribution}}},
            }},
        ],
    }
    tasks = [performance]
    cases = [
        ("gsm8k", {"exact_match": scalar(0.5),
                   "scored_samples": scalar(10, "sample")}),
        ("humaneval", {"pass_at_1": scalar(0), "pass_at_5": scalar(0.8)}),
        ("wikitext", {"perplexity": scalar(4.256789, "ppl", 1234),
                      "count": scalar(1234, "token", None)}),
        ("ceval", {"accuracy": scalar(0.25), "accuracy_by_subject": {
            "真实零": scalar(0, count=1), "未知学科": scalar(None, count=0)}}),
        ("custom", {"custom_score": scalar(0, "points"),
                    "custom_stats": distribution,
                    "by_group": {"count": 20,
                                 "max": scalar(1, count=None),
                                 "min": scalar(0, count=0)}}),
    ]
    for scenario, metrics in cases:
        tasks.append({
            "kind": "task", "name": "duplicate", "scenario": scenario,
            "category": "model_service_quality", "status": "success",
            "cases": [{"id": "case_1", "status": "success", "metrics": {
                "quality": {"values": {"task_view": {scenario: metrics}}}
            }}],
        })
    tasks.append({"kind": "task", "name": "empty", "cases": [],
                  "status": "failed", "category": "generation_performance"})
    return {"kind": "run", "id": "Synthetic display tests only",
            "status": "partial_failed", "tasks": tasks}


def test_display_embeds_original_values_and_offline_scripts(tmp_path):
    payload = browser_fixture()
    before = json.dumps(payload, ensure_ascii=False)
    html = write_html(payload, tmp_path / "fixture.html").read_text("utf-8")
    embedded = html.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["result"] == payload
    assert json.dumps(payload, ensure_ascii=False) == before
    assert "__CASE_SCRIPT__" not in html
    assert "__REQUEST_SCRIPT__" not in html
    assert 'id="analysis-task"' in html
    assert 'id="analysis-case"' in html
    assert 'id="analysis-metric"' in html
    assert '<script src=' not in html
    assert "function metricsFor" in html


if __name__ == "__main__":
    write_html(browser_fixture(), Path(sys.argv[1]))
