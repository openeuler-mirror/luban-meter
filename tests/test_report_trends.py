"""Stability trend front-end checks over synthetic saved metrics."""

import json
import subprocess
from pathlib import Path

import pytest

from luban_meter.reporting.html import write_html
from tests.node_runtime import node_binary


def scalar(value, unit="request"):
    return {"value": value, "unit": unit}


def window_row(index, start, end, **extra):
    row = {
        "index": index,
        "start_s": start,
        "end_s": end,
        "span_s": end - start,
        "complete": extra.pop("complete", True),
        "started_requests": scalar(extra.pop("started", 0)),
        "completed_requests": scalar(extra.pop("completed", 0)),
        "successful": scalar(extra.pop("successful", 0)),
        "failed": scalar(extra.pop("failed", 0)),
        "timeout_failed": scalar(extra.pop("timeout_failed", 0)),
        "other_failed": scalar(extra.pop("other_failed", 0)),
        "success_rate": {
            "value": extra.pop("success_rate", None), "unit": "ratio",
        },
        "timeout_rate": {
            "value": extra.pop("timeout_rate", None), "unit": "ratio",
        },
        "output_token_throughput": {
            "value": extra.pop("throughput", None),
            "unit": "token/s",
            "condition": "completion-attributed",
        },
        "ttft_ms": {
            "count": extra.pop("ttft_count", 0),
            "mean": extra.pop("ttft_mean", None),
            "p99": extra.pop("ttft_p99", None),
        },
        "e2el_ms": {
            "count": extra.pop("e2el_count", 0),
            "p99": extra.pop("e2el_p99", None),
        },
        "average_concurrency": scalar(None),
        "peak_concurrency": scalar(0),
    }
    row.update(extra)
    return row


def trend_payload():
    windows = [
        window_row(0, 0, 60, started=2, completed=2, successful=2,
                   throughput=1.5, success_rate=1.0, timeout_rate=0.0,
                   ttft_count=12, ttft_mean=80.0, ttft_p99=120.0,
                   e2el_count=12, e2el_p99=900.0),
        window_row(1, 60, 90, complete=False, started=1, completed=1,
                   successful=1, throughput=0.5, success_rate=1.0,
                   timeout_rate=0.0, ttft_count=3, ttft_mean=90.0,
                   ttft_p99=130.0, e2el_count=3, e2el_p99=950.0,
                   **{"ttft_ms": {"count": 3, "mean": 90.0, "p99": 130.0,
                                  "low_sample": True},
                      "e2el_ms": {"count": 3, "p99": 950.0,
                                  "low_sample": True}}),
    ]
    metric = {
        "status": "success",
        "observation_boundary": "client",
        "values": {"window_seconds": 60, "windows": windows},
        "metadata": {
            "window_seconds": 60,
            "warmup_included": False,
            "low_sample_threshold": 10,
            "timeout_error_types": ["ReadTimeout"],
            "throughput_definition": "completion-attributed",
            "unattributed_requests": {"count": 0},
        },
    }
    return {
        "kind": "run",
        "id": "trend-synthetic",
        "status": "success",
        "started_at": 1000.0,
        "tasks": [{
            "name": "perf",
            "category": "generation_performance",
            "status": "success",
            "cases": [{
                "id": "case-test",
                "status": "success",
                "collection_complete": True,
                "coverage": {"complete": 2, "failed": 0, "incomplete": 0,
                             "not_started": 0, "selected": 2},
                "parameters": {"model": "test-model", "profile": "rate",
                               "max_concurrency": 4,
                               "trend_window_seconds": 60},
                "metrics": {"stability_trends": metric},
                "window": {"started_at": 1000.0, "ended_at": 1090.0,
                           "duration_s": 90.0},
            }],
        }],
    }


def test_trend_unit_semantics():
    node = node_binary()
    if node is None:
        pytest.skip("Node.js is required for the report JavaScript checks")
    script = Path(__file__).with_name("report_trends.unit.cjs")
    subprocess.run([node, str(script)], check=True, capture_output=True)


def test_trend_export_embeds_module_and_saved_windows(tmp_path):
    node = node_binary()
    if node is None:
        pytest.skip("Node.js is required for the report JavaScript checks")
    report = write_html(trend_payload(), tmp_path / "report.html")
    text = report.read_text(encoding="utf-8")
    assert "function renderStabilityTrends" not in text
    assert 'id="stability-trends-section"' not in text
    assert 'id="analysis-stability-content"' in text
    assert "function selectAnalysisTab" in text
    assert "__TREND_SCRIPT__" not in text
    embedded = text.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    data = json.loads(embedded)
    case = data["result"]["tasks"][0]["cases"][0]
    assert case["metrics"]["stability_trends"]["values"]["windows"][1][
        "complete"] is False
    script = Path(__file__).with_name("report_trends.unit.cjs")
    subprocess.run([node, str(script), str(report)], check=True,
                   capture_output=True)
