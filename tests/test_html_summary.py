"""Report facts remain attributable and safe for offline navigation."""

import copy
import json

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
