"""Crash boundaries retain committed collections and delete partial facts."""

import json

import pytest

from luban_meter.core.planning import build_plan
from luban_meter.runtime.case_recovery import recover_cases
from luban_meter.utils.json_io import write_json_atomic


@pytest.mark.parametrize("result_saved", [False, True])
def test_collection_commit_survives_missing_task_checkpoint(
    tmp_path, result_saved
):
    task = build_plan(
        {
            "scenario": "online_serving",
            "model": "mock",
        }
    )["tasks"][0]
    raw = tmp_path / "raw"
    raw.mkdir()
    facts = {
        "collection_complete": True,
        "parameters": {"model": "mock"},
        "coverage": {"complete": 1},
        "window": {"duration_s": 1},
        "requests": [{"status": "failed"}],
        "samples": [],
    }
    target = raw / "case_1.json"
    write_json_atomic(target, facts)
    before = target.read_bytes()
    saved = {
        "id": "case_1",
        "collection_complete": True,
        "status": "success",
        "metrics": {"score": 7},
    }
    if result_saved:
        write_json_atomic(raw / "case_1.result.json", saved)
    outcome = {"cases": []}
    recover_cases(task, tmp_path, outcome, "interrupted")
    assert target.read_bytes() == before
    case = outcome["cases"][0]
    assert case["collection_complete"]
    if result_saved:
        assert case["metrics"] == saved["metrics"]
    else:
        assert case["metrics_status"] == "not_computed"
    recover_cases(task, tmp_path, outcome, "interrupted")
    assert target.read_bytes() == before


def test_incomplete_evidence_deleted_without_touching_other_cases(tmp_path):
    task = build_plan(
        {
            "scenario": "online_serving",
            "model": "mock",
            "api_key": "secret",
            "cases": {"items": [{"input_tokens": 8}, {"input_tokens": 16}]},
        }
    )["tasks"][0]
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "case_1.json").write_text('{"samples": ["sensitive text"]}')
    (raw / ".case_1.json.dead.tmp").write_text("partial sample")
    events = raw / "events.jsonl"
    events.write_text(
        json.dumps({"case_id": "case_1", "kind": "request_started"})
        + "\n"
        + json.dumps({"case_id": "unrelated", "kind": "sample_finished"})
        + "\n{truncated"
    )
    outcome = {"cases": []}
    recover_cases(task, tmp_path, outcome, "interrupted")
    case, unstarted = outcome["cases"]
    assert case["status"] == "discarded"
    assert case["parameters"]["api_key"] != "secret"
    assert unstarted["status"] == "skipped"
    assert not (raw / "case_1.json").exists()
    assert not (raw / ".case_1.json.dead.tmp").exists()
    assert "unrelated" in events.read_text()
    assert "case_1" not in events.read_text()
