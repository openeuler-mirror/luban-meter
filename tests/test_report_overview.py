"""Overview arithmetic and export contract, without model requests."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from luban_meter.reporting.html import write_html


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
