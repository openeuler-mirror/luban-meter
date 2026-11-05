"""Report facts remain attributable and safe for offline navigation."""

import copy
import json

import pytest

from luban_meter.reporting.html import report_data, write_html


def result_fixture():
    metric = {"value": 0.8, "unit": "ratio", "sample_count": 5}
    task = {
        "kind": "task",
        "name": "same-name",
        "category": "model_service_quality",
        "status": "success",
        "cases": [{
            "id": "case_1",
            "status": "success",
            "metrics": {"accuracy": metric},
        }],
    }
    return {
        "kind": "run",
        "id": "report-test",
        "status": "success",
        "tasks": [task, copy.deepcopy(task)],
    }


def test_duplicate_labels_have_distinct_navigation_identity():
    payload = result_fixture()
    before = copy.deepcopy(payload)
    rows = report_data(payload)["rows"]
    assert [(r["task_index"], r["case_index"]) for r in rows] == [
        (0, 0), (1, 0)
    ]
    assert all(r["count"] == 5 for r in rows)
    assert payload == before


def test_discarded_case_keeps_diagnostic_not_metric_rows():
    payload = result_fixture()
    payload["tasks"][0]["cases"][0].update(
        status="discarded", reason="interrupted"
    )
    data = report_data(payload)
    assert len(data["rows"]) == 1
    assert data["rows"][0]["task_index"] == 1
    assert data["result"]["tasks"][0]["cases"][0]["reason"]


def test_missing_value_and_hostile_text_survive_safely(tmp_path):
    payload = result_fixture()
    payload["tasks"][0]["name"] = "</script><script>alert(1)</script>"
    payload["tasks"][0]["cases"][0]["metrics"]["accuracy"]["value"] = None
    text = write_html(payload, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    embedded = text.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    data = json.loads(embedded)
    assert data["rows"][0]["value"] is None
    assert payload["tasks"][0]["name"] not in text
    assert data["result"] == payload


def test_run_summary_uses_same_saved_values_and_reason_as_terminal(tmp_path):
    from xml.etree import ElementTree

    from luban_meter.core.token_totals import throughput_result
    from luban_meter.reporting.html import run_summary_html
    from luban_meter.reporting.terminal import render_text

    payload = result_fixture()
    payload["tasks"][0]["category"] = "generation_performance"
    metric = throughput_result(400, 3)
    metric["values"]["service_view"]["total_token_throughput"]["value"] = 999
    metric["metadata"]["included_cases"] = [
        {"task_index": 0, "case_index": 0, "case": "<script>unsafe</script>"},
    ]
    payload.update(duration_s=10, metrics={"token_throughput": metric})
    original = copy.deepcopy(payload)
    fragment = run_summary_html(payload)
    node = ElementTree.fromstring(fragment)
    rows = [[cell.text for cell in row] for row in node.findall(".//tbody/tr")]
    assert rows[-1] == ["total_token_throughput", "999", "token/s"]
    assert rows[0] == ["整轮 duration", "10", "s"]
    # The saved total throughput also leads as a large hero number.
    assert 'class="run-hero"' in fragment
    assert 'run-hero-value">999<' in fragment
    text = render_text(payload)
    assert "性能评测总体 total_token_throughput=999 tokens/s" in text
    assert "<script>unsafe" not in fragment
    exported = write_html(payload, tmp_path / "report.html").read_text()
    assert fragment in exported
    assert "__RUN_SUMMARY__" not in exported
    # Model identity and the core metric cards precede the run summary.
    assert exported.index('id="overview-models"') < exported.index(
        'id="run-throughput-summary"'
    )
    assert exported.index('id="cards"') < exported.index(
        'id="run-throughput-summary"'
    )
    assert payload == original
    metric.update(status="unavailable", reason="missing usage in case_2")
    metric["values"]["service_view"]["total_token_throughput"]["value"] = None
    unavailable = run_summary_html(payload)
    assert "missing usage in case_2" in unavailable
    assert 'run-hero-value">N/A<' in unavailable
    assert "missing usage in case_2" in render_text(payload)
    del payload["metrics"]
    assert "N/A" in run_summary_html(payload)
    assert "整轮指标未保存" in render_text(payload)



@pytest.mark.parametrize("kind", ["run", "task"])
@pytest.mark.parametrize("include_requests", [False, True])
def test_template_literals_are_preserved_in_data_and_summary(
    tmp_path, kind, include_requests,
):
    from luban_meter.core.token_totals import throughput_result
    from luban_meter.reporting.html import run_summary_html
    from tests.test_report_requests import fixture

    payload, raw_path, facts = fixture(tmp_path)
    marker = '__RUN_SUMMARY__ __REPORT_DATA__ "中文" </script> &'
    task = payload["tasks"][0]
    task["cases"][0]["parameters"] = {"model": marker, "extra": marker}
    facts["samples"][0]["source"] = {"prompt": marker}
    raw_path.write_text(
        json.dumps(facts, ensure_ascii=False), encoding="utf-8"
    )
    if kind == "task":
        payload = {"kind": "task", **task}
    else:
        metric = throughput_result(30, 2, metadata={"note": marker})
        payload.update(metrics={"token_throughput": metric}, duration_s=3)
    original = copy.deepcopy(payload)
    before = raw_path.read_bytes()
    html = write_html(
        payload, tmp_path / "report.html",
        include_requests=include_requests,
        raw_root=tmp_path if kind == "run" else raw_path.parent.parent,
    ).read_text("utf-8")
    embedded = html.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    data = json.loads(embedded)
    assert data["result"] == original
    if kind == "run":
        assert run_summary_html(payload) in html
    assert (('id="run-throughput-summary"' in html) is (kind == "run"))
    assert marker not in html  # Script-sensitive text is escaped at embedding.
    if include_requests:
        details = data["request_details"][0][0]
        assert details["status"] == "available"
        assert details["samples"][0]["source"]["prompt"] == marker
    assert raw_path.read_bytes() == before
    assert payload == original
