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
def test_both_benchmarks_plan_default_streaming_metrics(tmp_path, scenario):
    dataset = tmp_path / "conversations.jsonl"
    dataset.write_text('{}\n')
    task = build_plan({
        "scenario": scenario, "model": "mock", "dataset_path": str(dataset),
    })["tasks"][0]
    assert benchmarks.get(scenario).default_metrics == (
        "streaming_performance", "streaming_phase_timing",
    )
    assert task["cases"][0]["parameters"]["metrics"] == [
        "streaming_performance", "streaming_phase_timing",
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


def _with_queue_time(values):
    responses = [response(i) for i in range(len(values))]
    for item, queue_ms in zip(responses, values):
        if queue_ms is not None:
            item["server_queue_time_ms"] = queue_ms
    return streaming(formal_facts(responses, multi_turn=True))


def test_queue_time_waited_mean_averages_only_waited_requests():
    metric = _with_queue_time([12.3, 0.0, 5.0])

    distribution = metric["values"]["request_view"]["queue_time_ms"]
    assert distribution["count"] == 2
    assert distribution["mean"] == pytest.approx(8.65)
    assert distribution["min"] == pytest.approx(5.0)
    assert distribution["max"] == pytest.approx(12.3)
    mean = metric["values"]["service_view"]["queue_time_waited_mean"]
    assert mean == {"value": pytest.approx(8.65), "unit": "ms"}
    assert "reason" not in mean
    assert "condition" not in mean
    coverage = metric["values"]["service_view"]
    assert coverage["queue_time_reported_requests"]["value"] == 3
    assert coverage["queue_time_missing_requests"]["value"] == 0


def test_queue_time_reported_zero_waits_is_a_real_observation():
    metric = _with_queue_time([0.0, 0.0])

    distribution = metric["values"]["request_view"]["queue_time_ms"]
    assert distribution["count"] == 0
    assert distribution["mean"] is None
    assert distribution["reason"] == "no request waited in the server queue"
    mean = metric["values"]["service_view"]["queue_time_waited_mean"]
    assert mean["value"] is None
    assert mean["reason"] == "no request waited in the server queue"


def test_queue_time_zero_wait_with_missing_is_not_no_wait():
    # One request reported a real zero; the other is unknown. That is
    # not "no request waited in the server queue".
    metric = _with_queue_time([0.0, None])

    mean = metric["values"]["service_view"]["queue_time_waited_mean"]
    assert mean["value"] is None
    assert mean["reason"] == (
        "no request with reported timing waited; 1 of 2 requests did "
        "not report timing"
    )
    coverage = metric["values"]["service_view"]
    assert coverage["queue_time_reported_requests"]["value"] == 1
    assert coverage["queue_time_missing_requests"]["value"] == 1


def test_queue_time_partial_coverage_keeps_subset_with_scope():
    # The subset mean stays, with an explicit incomplete-coverage mark;
    # missing requests are not read as zero waits.
    metric = _with_queue_time([12.3, None, None])

    distribution = metric["values"]["request_view"]["queue_time_ms"]
    assert distribution["count"] == 1
    assert distribution["mean"] == pytest.approx(12.3)
    assert "partial timing coverage: 1 of 3" in distribution["condition"]
    mean = metric["values"]["service_view"]["queue_time_waited_mean"]
    assert mean["value"] == pytest.approx(12.3)
    assert "partial timing coverage: 1 of 3" in mean["condition"]
    assert "reason" not in mean
    coverage = metric["values"]["service_view"]
    assert coverage["queue_time_reported_requests"]["value"] == 1
    assert coverage["queue_time_missing_requests"]["value"] == 2


def test_queue_time_nonfinite_record_value_is_ignored_not_fatal():
    # A replayed or legacy record can carry an infinite timing; the
    # aggregation must drop it as missing instead of failing the whole
    # streaming_performance metric (TTFT, throughput stay available).
    metric = _with_queue_time([12.3, float("inf")])

    assert metric["status"] == "success"
    mean = metric["values"]["service_view"]["queue_time_waited_mean"]
    assert mean["value"] == pytest.approx(12.3)
    assert "partial timing coverage: 1 of 2" in mean["condition"]
    coverage = metric["values"]["service_view"]
    assert coverage["queue_time_reported_requests"]["value"] == 1
    assert coverage["queue_time_missing_requests"]["value"] == 1
    ttft = metric["values"]["request_view"]["ttft"]
    assert ttft["count"] == 2


def test_queue_time_absent_metrics_marked_unavailable():
    metric = _with_queue_time([None, None])

    distribution = metric["values"]["request_view"]["queue_time_ms"]
    assert distribution["count"] == 0
    reason = "service did not report per-request queue time metrics"
    assert distribution["reason"] == reason
    mean = metric["values"]["service_view"]["queue_time_waited_mean"]
    assert mean["value"] is None
    assert mean["reason"] == reason
    coverage = metric["values"]["service_view"]
    assert coverage["queue_time_reported_requests"]["value"] == 0
    assert coverage["queue_time_missing_requests"]["value"] == 2


def test_queue_time_raw_precision_is_not_rounded_away():
    metric = _with_queue_time([0.0004, 0.0])

    # 0.4 microseconds is a real positive wait: the distribution, the
    # standalone mean and the waited count must stay consistent.
    distribution = metric["values"]["request_view"]["queue_time_ms"]
    assert distribution["count"] == 1
    for field in ("mean", "min", "max", "p50", "p95"):
        assert distribution[field] == pytest.approx(0.0004), field
    mean = metric["values"]["service_view"]["queue_time_waited_mean"]
    assert mean["value"] == pytest.approx(0.0004)
    assert mean["value"] == pytest.approx(distribution["mean"])
    assert mean["value"] > 0


def test_sample_facts_cannot_replace_missing_formal_requests():
    metric = streaming(formal_facts([]))
    assert metric["status"] == "unavailable"
    assert "requests" in metric["reason"]


def test_failed_request_without_response_remains_diagnosable():
    facts = formal_facts([response(status="failed")], multi_turn=True)
    del facts["requests"][0]["response"]
    with pytest.raises(ValueError, match="duration_ms"):
        streaming(facts)
