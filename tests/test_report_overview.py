"""Overview arithmetic and export contract, without model requests."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from luban_meter.reporting.html import write_html
from luban_meter.utils.json_io import write_json_atomic


def test_overview_summary_semantics():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the report JavaScript checks")
    script = Path(__file__).with_name("report_overview.unit.cjs")
    subprocess.run([node, str(script)], check=True, capture_output=True)


def test_overview_export_keeps_original_result_and_offline_pages(tmp_path):
    payload = {
        "kind": "run", "id": "empty", "status": "failed", "tasks": [],
    }
    text = write_html(payload, tmp_path / "report.html").read_text("utf-8")
    embedded = text.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["result"] == payload
    assert text.count('class="report-page"') == 6
    overview = text.split('<section id="overview"', 1)[1].split(
        '</section>', 1
    )[0]
    assert 'id="performance-overview"' in overview
    assert 'id="quality-overview"' in overview
    assert 'id="overview-quality-best"' in overview
    assert '<body class="dark">' in text
    assert 'aria-controls="performance-summary-panel"' in text
    assert 'aria-controls="quality-summary-panel"' in text
    assert 'id="performance-overview"' in text
    assert 'id="quality-overview"' in text
    assert "function performanceSummary" in text
    assert "__OVERVIEW_SCRIPT__" not in text
    assert "__SUMMARY_SCRIPT__" not in text
    assert "function summaryCard" in text
    assert '<script src=' not in text


def test_persisted_performance_metric_selection(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the report JavaScript checks")
    def scalar(value, unit):
        return {"value": value, "unit": unit}

    provider = {
        "status": "success", "values": {
            "request_view": {
                "ttft": {"p99": 5, "unit": "ms"},
                "tpot": {"p99": 2, "unit": "ms/token"},
                "e2el": {"p99": 20, "unit": "ms"},
            },
            "service_view": {
                "output_token_throughput": scalar(10, "token/s"),
                "successful_requests": scalar(1, "request"),
                "total_requests": scalar(1, "request"),
                "failed_requests": scalar(0, "request"),
            },
        },
    }
    tasks = []
    for name in ("streaming_performance", "multi_turn_performance",
                 "z_extension_performance", "a_extension_performance"):
        for request_view in ("complete", "missing", "empty", "null"):
            metric = json.loads(json.dumps(provider))
            if name == "multi_turn_performance":
                metric["values"]["turn_view"] = {}
            if request_view == "missing":
                del metric["values"]["request_view"]
            elif request_view == "empty":
                metric["values"]["request_view"] = {}
            elif request_view == "null":
                metric["values"]["request_view"] = None
            tasks.append({
                "name": name, "category": "generation_performance",
                "status": "success",
                "cases": [{
                    "id": "case_1", "status": "success",
                    "collection_complete": True, "metrics_status": "complete",
                    "parameters": {"model": "test", "profile": "concurrent",
                                   "max_concurrency": 4},
                    "metrics": {name: metric, "token_throughput": {
                        "status": "success", "values": {"service_view": {
                            "total_token_throughput": scalar(999, "token/s"),
                        }},
                    }},
                }],
            })
    target = tmp_path / "persisted.json"
    payload = {"kind": "run", "status": "success", "tasks": tasks}
    write_json_atomic(target, payload)
    report = write_html(payload, tmp_path / "report.html")
    subprocess.run([
        node, str(Path(__file__).with_name("report_overview.unit.cjs")),
        str(target), str(report),
    ], check=True, capture_output=True)
