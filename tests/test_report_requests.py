"""Full HTML exports preserve request/sample identity and raw evidence."""

import copy
import json
import sys
from pathlib import Path

import pytest

from luban_meter.cli import main
from luban_meter.reporting.html import report_data, write_html
from luban_meter.reporting.request_details import request_rows


def fixture(tmp_path):
    raw = tmp_path / "tasks" / "task_1" / "raw" / "case_1.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    facts = {
        "samples": [
            {"sample_id": "1", "state": "complete",
             "source": {"prompt": "<script>alert(1)</script>"},
             "facts": {"raw_output": "a" * 100000}},
            {"sample_id": "2", "state": "not_started", "facts": {}},
        ],
        "requests": [
            {"request_id": "b", "sample_id": "1", "status": "failed",
             "error": "timeout", "response": {"duration_ms": 5000}},
            {"request_id": "a", "sample_id": "1", "status": "success",
             "response": {"text": "hello", "input_tokens": 0,
                          "output_tokens": 1, "e2el_ms": 20}},
        ],
    }
    raw.write_text(json.dumps(facts), encoding="utf-8")
    result = {
        "kind": "run", "id": "run", "status": "partial_failed",
        "tasks": [{
            "name": "task_1", "category": "generation_performance",
            "status": "partial_failed", "cases": [{
                "id": "case_1", "status": "partial_failed",
                "raw": "/old/container/tasks/task_1/raw/case_1.json",
                "metrics": {},
            }],
        }],
    }
    return result, raw, facts


def embedded(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text.split('id="report-data">', 1)[1].split(
        "</script>", 1
    )[0])


def test_summary_does_not_read_or_embed_requests(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    raw.write_text("invalid", encoding="utf-8")
    data = report_data(payload)
    assert "request_details" not in data


def test_relocated_export_preserves_all_text_and_identity(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    original = raw.read_bytes()
    before = copy.deepcopy(payload)
    path = write_html(payload, tmp_path / "full.html",
                      include_requests=True, raw_root=tmp_path)
    detail = embedded(path)["request_details"][0][0]
    assert detail["status"] == "available"
    assert detail["requests"] == facts["requests"]
    assert detail["samples"] == facts["samples"]
    assert len(detail["rows"]) == 3
    assert detail["rows"][0]["sample_indices"] == [0]
    assert detail["rows"][1]["sample_indices"] == [0]
    assert detail["rows"][1]["input_tokens"] == 0
    assert detail["rows"][1]["ttft_ms"] is None
    assert detail["rows"][2]["request_index"] is None
    assert "<script>alert(1)</script>" not in path.read_text()
    assert raw.read_bytes() == original
    assert payload == before


@pytest.mark.parametrize("contents", ["bad", "[]", '{"requests": []}'])
def test_bad_raw_report_keeps_explicit_unavailable_reason(tmp_path, contents):
    payload, raw, _ = fixture(tmp_path)
    raw.write_text(contents, encoding="utf-8")
    data = report_data(payload, include_requests=True, raw_root=tmp_path)
    detail = data["request_details"][0][0]
    assert detail["status"] == "unavailable"
    assert detail["reason"]


def test_missing_and_outside_root_are_not_silently_read(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    root = tmp_path / "separate"
    root.mkdir()
    payload["tasks"][0]["cases"][0]["raw"] = str(raw)
    detail = report_data(payload, include_requests=True, raw_root=root)
    assert detail["request_details"][0][0]["status"] == "unavailable"


def test_discarded_case_does_not_expose_residual_raw(tmp_path):
    payload, _, _ = fixture(tmp_path)
    payload["tasks"][0]["cases"][0]["status"] = "discarded"
    data = report_data(payload, include_requests=True, raw_root=tmp_path)
    assert "requests" not in data["request_details"][0][0]


def test_task_export_and_cli_default_filename(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    task = {"kind": "task", **payload["tasks"][0]}
    path = raw.parent.parent / "result.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    assert main(["report", "--input", str(path), "--include-requests"]) == 0
    data = embedded(path.parent / "report-full.html")
    assert data["request_details"][0][0]["status"] == "available"
    assert main(["report", "--input", str(path)]) == 0
    assert "request_details" not in embedded(path.parent / "report.html")
    assert main(["report", "--input", str(path),
                 "--raw-root", str(tmp_path)]) == 2


def test_many_requests_and_duplicate_sample_ids_are_not_dropped():
    samples = [{"sample_id": "x", "state": "complete"}] * 2
    requests = [{"sample_id": "x", "request_id": str(i),
                 "status": "success"} for i in range(16384)]
    rows = request_rows(requests, samples)
    assert len(rows) == 16384
    assert rows[-1]["sample_indices"] == [0, 1]


def test_request_timings_do_not_substitute_scheduler_occupancy():
    rows = request_rows([{"start_offset_s": 1, "end_offset_s": 9}], [])
    assert rows[0]["latency_ms"] is None


def test_missing_ids_are_not_guessed():
    rows = request_rows([{"status": "failed"}], [{"state": "failed"}])
    assert len(rows) == 2
    assert rows[0]["sample_indices"] == []
    assert rows[1]["request_index"] is None


def write_browser_fixture(root):
    """Generate a synthetic offline fixture for the browser regression."""
    root.mkdir(parents=True, exist_ok=True)
    payload, raw, _ = fixture(root)
    requests, samples = [], []
    for index in range(120):
        failed = index % 10 == 0
        samples.append({
            "sample_id": str(index),
            "state": "failed" if failed else "complete",
            "source": {"prompt": "原始输入 " + str(index)},
            "facts": {"prompt": "实际样本提示词 " + str(index),
                      "raw_output": "</script><script>alert(1)</script>"},
        })
        requests.append({
            "sample_id": str(index), "request_id": f"request-{index}",
            "status": "failed" if failed else "success",
            "response": {"ttft_ms": index, "e2el_ms": index + 100,
                         "input_tokens": 64, "output_tokens": 16,
                         "text": "</script><script>alert(1)</script>"},
            "error": "timeout" if failed else None,
            "start_offset_s": index,
            "end_offset_s": index + 0.5,
        })
    samples.append({"sample_id": "orphan", "state": "not_started"})
    raw.write_text(json.dumps({"samples": samples, "requests": requests}),
                   encoding="utf-8")
    case = payload["tasks"][0]["cases"][0]
    case["coverage"] = {"selected": 121, "complete": 108, "failed": 12,
                        "not_started": 1, "incomplete": 0}
    case["parameters"] = {"model": "示例模型（合成数据）",
                          "profile": "concurrent", "max_concurrency": 16}
    case["metrics"] = {"example": {"values": {
        "request_view": {"ttft": {"mean": 59.5, "count": 120, "unit": "ms"}}
    }}}
    payload["id"] = "第二版交互示例（合成数据，非真实评测）"
    payload["tasks"][0]["cases"].append({
        "id": "missing", "status": "failed", "metrics": {},
        "raw": "/missing/case.json",
    })
    write_html(payload, root / "report-full.html",
               include_requests=True, raw_root=root)
    write_html(payload, root / "report.html")


if __name__ == "__main__":
    write_browser_fixture(Path(sys.argv[1]))
