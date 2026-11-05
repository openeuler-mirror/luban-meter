"""One public streaming metric consumes formal request facts."""

import copy
import subprocess
import sys

import pytest

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.metric import compute_metrics, metrics
from luban_meter.core.planning import bootstrap, build_plan


def response(index=0, status="success", usage=True):
    result = {
        "status": status,
        "duration_ms": 60,
        "dispatch_delay_ms": 2,
        "start_offset_ms": index * 100,
        "scheduled_latency_ms": 62,
    }
    if status == "success":
        result.update(
            ttft_ms=10 + index,
            e2el_ms=60,
            last_output_latency_ms=50,
            itl_samples_ms=[20, 20],
            input_tokens=100 if usage else None,
            output_tokens=3 if usage else None,
            lengths={"condition": "conforming"},
        )
    else:
        result["error"] = {"type": "RuntimeError", "message": "HTTP 500"}
    return result


def formal_facts(responses, *, multi_turn=False, sample_state="complete"):
    requests = [
        {
            "request_id": f"request-{index}",
            "sample_id": "0" if multi_turn else str(index),
            "status": item["status"],
            "response": item,
            "scheduling": {"turn_index": index} if multi_turn else {},
        }
        for index, item in enumerate(responses)
    ]
    sample_facts = {
        "status": "success" if sample_state == "complete" else "failed",
        "planned_turns": len(responses) + 1,
        "completed_turns": sum(r["status"] == "success" for r in responses),
        "abort_reason": "turn_failed" if sample_state == "failed" else None,
    } if multi_turn else response(9)
    return {
        "parameters": {"multi_max_turns": 4, "multi_num_sessions": 1},
        "requests": requests,
        "samples": [{
            "sample_id": "0", "state": sample_state,
            "facts": {} if sample_state == "incomplete" else sample_facts,
        }],
        "metadata": {"slo_config": {"ttft_ms": 100}},
        "coverage": {sample_state: 1, "selected": 1},
        "window": {
            "duration_s": 2, "request_capacity": 2, "peak_requests": 1,
        },
        "collection_complete": True,
    }


def streaming(facts):
    bootstrap()
    capabilities = benchmarks.get("multi_turn_serving").capabilities
    return compute_metrics(
        ["streaming_performance"], capabilities, facts
    )["streaming_performance"]


@pytest.mark.parametrize("scenario", ["online_serving", "multi_turn_serving"])
def test_both_benchmarks_plan_one_streaming_metric(tmp_path, scenario):
    dataset = tmp_path / "conversations.jsonl"
    dataset.write_text('{}\n')
    task = build_plan({
        "scenario": scenario, "model": "mock", "dataset_path": str(dataset),
    })["tasks"][0]
    assert benchmarks.get(scenario).default_metrics == (
        "streaming_performance",
    )
    assert task["cases"][0]["parameters"]["metrics"] == [
        "streaming_performance"
    ]


@pytest.mark.parametrize("scenario", ["online_serving", "multi_turn_serving"])
def test_removed_metric_name_is_rejected_during_planning(tmp_path, scenario):
    dataset = tmp_path / "conversations.jsonl"
    dataset.write_text('{}\n')
    with pytest.raises(ValueError, match="unknown metric.*multi_turn"):
        build_plan({
            "scenario": scenario, "model": "mock",
            "dataset_path": str(dataset),
            "metrics": ["multi_turn_performance"],
        })


def test_cli_lists_one_streaming_metric_and_both_benchmarks(tmp_path):
    bootstrap()
    assert "multi_turn_performance" not in metrics.names()
    for category, expected in (
        ("metrics", ["streaming_performance"]),
        ("benchmarks", ["online_serving", "multi_turn_serving"]),
    ):
        completed = subprocess.run(
            [sys.executable, "-m", "luban_meter.cli", "--log-dir",
             str(tmp_path / "log"), category, "list"],
            text=True, capture_output=True, timeout=15, check=True,
        )
        column = 1 if category == "benchmarks" else 0
        names = [
            line.split("\t")[column]
            for line in completed.stdout.splitlines()
        ]
        assert set(expected) <= set(names)
        if category == "metrics":
            assert "multi_turn_performance" not in names


@pytest.mark.parametrize("sample_state", ["complete", "failed", "incomplete"])
@pytest.mark.parametrize("statuses", [
    ["success", "success"], ["success", "failed"], ["failed", "failed"],
])
def test_shared_views_use_requests_for_every_session_state(
    sample_state, statuses
):
    responses = [response(i, status) for i, status in enumerate(statuses)]
    single = formal_facts(responses, sample_state=sample_state)
    multi = formal_facts(
        responses, multi_turn=True, sample_state=sample_state
    )
    original = copy.deepcopy(multi)
    single_metric, multi_metric = streaming(single), streaming(multi)
    successes = statuses.count("success")
    expected_status = (
        "partial_failed" if 0 < successes < len(statuses)
        else "success" if successes else "failed"
    )
    assert single_metric["status"] == multi_metric["status"] == expected_status
    for view in ("request_view", "service_view"):
        assert single_metric["values"][view] == multi_metric["values"][view]
    service = multi_metric["values"]["service_view"]
    assert service["total_requests"]["value"] == 2
    assert service["successful_requests"]["value"] == successes
    assert service["failed_requests"]["value"] == 2 - successes
    assert service["total_tokens"]["value"] == successes * 103
    assert service["total_token_throughput"]["value"] == successes * 103 / 2
    assert service["goodput"]["slo_satisfied_rate"]["value"] == successes / 2
    assert multi_metric["values"]["turn_view"]["turns_observed"] == successes
    assert multi_metric["values"]["session_view"]["session_coverage"][
        sample_state
    ] == 1
    assert "turn_view" not in single_metric["values"]
    assert multi_metric["values"]["lengths"] == single_metric["lengths"]
    assert multi_metric["values"]["multi_max_turns"] == 4
    assert "prefix_cache" in multi_metric["values"]["caching_view"]
    assert "pressure_view" in multi_metric["values"]
    assert multi == original


def test_missing_usage_preserves_latencies_and_request_counts():
    metric = streaming(formal_facts(
        [response(usage=False)], multi_turn=True, sample_state="incomplete"
    ))
    assert metric["status"] == "success"
    values = metric["values"]
    assert values["request_view"]["ttft"]["mean"] == 10
    assert values["request_view"]["e2el"]["count"] == 1
    assert values["service_view"]["total_requests"]["value"] == 1
    assert values["service_view"]["total_token_throughput"]["value"] is None
    throughput = values["service_view"]["total_token_throughput"]
    assert "usage" in throughput["reason"]


def test_sample_facts_cannot_replace_missing_formal_requests():
    metric = streaming(formal_facts([]))
    assert metric["status"] == "unavailable"
    assert "requests" in metric["reason"]


def test_failed_request_without_response_remains_diagnosable():
    facts = formal_facts([response(status="failed")], multi_turn=True)
    del facts["requests"][0]["response"]
    with pytest.raises(ValueError, match="duration_ms"):
        streaming(facts)
