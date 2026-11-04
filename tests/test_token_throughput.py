"""Persisted throughput scope, evidence and failure handling."""

import copy

import pytest

from luban_meter.core.token_totals import formal_throughput
from luban_meter.core.token_totals import (
    request_totals,
    run_throughput,
    throughput_result,
)


def formal_facts(requests, samples=(), duration=2):
    return {
        "requests": requests, "samples": samples,
        "window": {"duration_s": duration}, "collection_complete": True,
    }


def request(index, usage=None, status="success"):
    row = {
        "request_id": f"r{index}", "sample_id": "s", "status": status,
    }
    if usage is not None:
        row["usage"] = {
            "prompt_tokens": usage[0], "completion_tokens": usage[1],
        }
    return row


def case(case_id, total=300, duration=2):
    return {
        "id": case_id, "status": "success", "collection_complete": True,
        "metrics": {"token_throughput": throughput_result(total, duration)},
    }


def test_multi_request_sample_counts_actual_successes_and_parse_failures():
    requests = [request(0, (10, 2)), request(1, (9, 0)),
                request(2, (100, 100), status="failed")]
    samples = [{"sample_id": "s", "facts": {"status": "parse_failed"}}]
    facts = formal_facts(requests, samples)
    original = copy.deepcopy(facts)
    metric = formal_throughput(facts)
    view, metadata = metric["values"]["service_view"], metric["metadata"]
    assert view["total_tokens"]["value"] == 21
    assert view["total_token_throughput"]["value"] == 10.5
    assert metadata["successful_requests"] == 2
    assert metadata["failed_requests"] == 1
    assert metadata["warmup_tokens_included"] is False
    assert metadata["warmup_duration_included"] is False
    assert metadata["answer_parse_failures_included"] is True
    assert facts == original


def test_missing_usage_keeps_identity_and_explicit_reason():
    requests = [request(0, (10, 2)), request(1)]
    facts = formal_facts(requests)
    metric = formal_throughput(facts)
    assert metric["status"] == "unavailable"
    assert metric["values"]["service_view"]["total_tokens"]["value"] is None
    assert metric["metadata"]["missing_usage_requests"] == 1
    assert metric["metadata"]["unavailable_requests"][0]["request_id"] == "r1"
    assert "usage" in metric["reason"]


def test_duplicate_sample_provenance_never_silently_overwrites():
    samples = [{"sample_id": "s", "facts": {"usage": {
        "prompt_tokens": 10, "completion_tokens": 2,
    }}}] * 2
    assert request_totals([request(0)], samples)["total_tokens"] is None


def test_batch_counts_tokens_and_logical_requests():
    row = request(0)
    row["logical_requests"] = 2
    samples = [{"sample_id": "s", "facts": {"requests": [
        {"actual_prompt_tokens": 10, "actual_output_tokens": 2},
        {"actual_prompt_tokens": 20, "actual_output_tokens": 3},
    ]}}]
    stats = request_totals([row], samples)
    assert stats["total_tokens"] == 35
    assert stats["successful_requests"] == 2


def test_unequal_windows_and_skipped_task_have_explicit_scope():
    tasks = [
        {"category": "generation_performance",
         "name": "same", "status": "success",
         "cases": [case("same", 300, 2), {"id": "skip", "status": "skipped"}]},
        {"category": "generation_performance",
         "name": "same", "status": "success", "cases": [case("same", 100, 1)]},
        {"category": "generation_performance",
         "name": "skip", "status": "skipped", "cases": []},
    ]
    metric = run_throughput(tasks)
    view = metric["values"]["service_view"]
    assert view["total_tokens"] == {"value": 400, "unit": "token"}
    assert view["benchmark_duration"]["value"] == 3
    assert view["total_token_throughput"]["value"] == pytest.approx(400 / 3)
    ledger = metric["metadata"]
    assert [item["task_index"] for item in ledger["included_cases"]] == [0, 1]
    assert len(ledger["skipped_cases"]) == 2
    assert ledger["unavailable_cases"] == []


@pytest.mark.parametrize("status", ["discarded", "interrupted"])
def test_incomplete_case_invalidates_run_with_case_identity(status):
    row = case("bad")
    row["status"] = status
    metric = run_throughput([{
        "category": "generation_performance",
        "name": "task", "status": "partial_failed", "cases": [case("ok"), row],
    }])
    assert metric["status"] == "unavailable"
    assert metric["values"]["service_view"]["total_tokens"]["value"] is None
    assert metric["metadata"]["unavailable_cases"][0]["case"] == "bad"
    assert metric["metadata"]["included_cases"][0]["case"] == "ok"


@pytest.mark.parametrize("duration", [0, -1, None, float("nan")])
def test_bad_duration_is_unavailable(duration):
    metric = formal_throughput(
        formal_facts([request(0, (2, 3))], duration=duration)
    )
    assert metric["status"] == "unavailable"
    assert "duration" in metric["reason"]


def test_legacy_or_failed_task_does_not_become_zero_throughput():
    metric = run_throughput([
        {"category": "generation_performance",
         "name": "failed", "status": "failed", "cases": []},
    ])
    assert metric["status"] == "unavailable"
    assert metric["metadata"]["unavailable_cases"][0]["task"] == "failed"


def test_quality_cases_do_not_contribute_or_invalidate_performance():
    performance = {
        "category": "generation_performance", "name": "performance",
        "status": "success", "cases": [case("one", 300, 2)],
    }
    quality = {
        "category": "model_service_quality", "name": "quality",
        "status": "failed", "cases": [case("one", 9000, 100)],
    }
    expected = run_throughput([performance])
    actual = run_throughput([performance, quality])
    assert actual == expected
    quality["cases"] = []
    assert run_throughput([performance, quality]) == expected


def test_quality_execution_does_not_save_throughput(tmp_path, monkeypatch):
    import json

    from luban_meter.core.benchmark import benchmarks
    from luban_meter.core.planning import build_plan
    from luban_meter.runtime.evaluation_session import run_case

    (tmp_path / "unused.jsonl").write_text("{}\n")
    plan = build_plan({
        "scenario": "gsm8k", "model": "mock",
        "dataset_path": str(tmp_path / "unused.jsonl"),
    })
    row = plan["tasks"][0]["cases"][0]
    row["parameters"]["metrics"] = []
    scenario = benchmarks.get("gsm8k")
    monkeypatch.setattr(scenario, "prepare", lambda self, context: None)
    monkeypatch.setattr(scenario, "samples", lambda self, context: iter([]))
    monkeypatch.setattr(scenario, "metadata", lambda self, context: {})
    (tmp_path / "raw").mkdir()
    result = run_case(scenario, row, tmp_path, tmp_path / ".stop")
    assert "total_tokens" not in result
    assert "token_throughput" not in result["metrics"]
    saved = json.loads(
        (tmp_path / "raw" / (row["id"] + ".result.json")).read_text()
    )
    assert "total_tokens" not in saved
    assert "token_throughput" not in saved["metrics"]



def test_common_throughput_is_not_registered():
    from luban_meter.core.metric import metrics

    with pytest.raises((KeyError, ValueError)):
        metrics.get("token_throughput")
