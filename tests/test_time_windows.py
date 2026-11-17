"""Windowed stability statistics over persisted streaming facts."""

from __future__ import annotations

import json

import pytest

from luban_meter.benchmarking.benchmark_metrics import StabilityTrends
from luban_meter.benchmarking.generation_performance import (
    generation_benchmarks,
    multi_turn_serving,
    prefix_cache,
)
from luban_meter.benchmarking.generation_performance.common import (
    time_windows,
)
from luban_meter.core.metric import compute_metrics
from luban_meter.reporting.data import metric_groups, metric_rows


def success(index, start_ms, end_ms, tokens=10, ttft_ms=50.0):
    return {
        "request_index": index,
        "status": "success",
        "start_offset_ms": start_ms,
        "end_offset_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "dispatch_delay_ms": 0.0,
        "ttft_ms": ttft_ms,
        "e2el_ms": end_ms - start_ms,
        "itl_samples_ms": [],
        "output_tokens": tokens,
    }


def failed(index, start_ms, end_ms, error_type="RuntimeError"):
    error = {"type": error_type, "message": "boom"}
    if error_type is None:
        return {
            "request_index": index,
            "status": "failed",
            "start_offset_ms": start_ms,
            "end_offset_ms": end_ms,
        }
    return {
        "request_index": index,
        "status": "failed",
        "start_offset_ms": start_ms,
        "end_offset_ms": end_ms,
        "error": error,
    }


def facts(requests, duration, window=None, metadata=None):
    parameters = {} if window is None else {"trend_window_seconds": window}
    return {
        "parameters": parameters,
        "requests": requests,
        "samples": [],
        "metadata": metadata or {},
        "coverage": {},
        "collection_complete": True,
        "window": {
            "started_at": 1000.0,
            "ended_at": 1000.0 + duration,
            "duration_s": duration,
            "request_capacity": 4,
            "peak_requests": 2,
            "admitted_requests": len(requests),
            "stop_reason": "num_requests",
        },
    }


def rows(result):
    return result["values"]["windows"]


def test_window_boundaries_and_final_partial_window():
    requests = [
        success(0, 0.0, 59990.0),
        success(1, 60000.0, 60010.0),
        success(2, 120000.0, 149900.0),
    ]
    result = time_windows.time_window_result(requests, 150.0)
    windows = rows(result)
    assert [window["index"] for window in windows] == [0, 1, 2]
    assert [window["start_s"] for window in windows] == [0, 60, 120]
    assert [window["end_s"] for window in windows] == [60, 120, 150]
    assert [window["span_s"] for window in windows] == [60, 60, 30]
    assert [window["complete"] for window in windows] == [True, True, False]
    assert [window["completed_requests"]["value"] for window in windows] == [
        1, 1, 1,
    ]


def test_exact_multiple_duration_creates_no_extra_window():
    result = time_windows.time_window_result(
        [success(0, 0.0, 1000.0)], 120.0
    )
    windows = rows(result)
    assert len(windows) == 2
    assert all(window["complete"] for window in windows)


def test_completion_attribution_and_started_requests():
    requests = [success(0, 55000.0, 65000.0, tokens=30)]
    windows = rows(time_windows.time_window_result(requests, 120.0))
    assert windows[0]["started_requests"]["value"] == 1
    assert windows[0]["completed_requests"]["value"] == 0
    assert windows[0]["output_token_throughput"]["value"] == 0
    assert windows[1]["started_requests"]["value"] == 0
    assert windows[1]["completed_requests"]["value"] == 1
    assert windows[1]["output_token_throughput"]["value"] == 0.5


def test_concurrency_overlap_and_peak():
    requests = [
        success(0, 0.0, 90000.0),
        success(1, 30000.0, 150000.0),
    ]
    windows = rows(time_windows.time_window_result(requests, 150.0))
    assert [window["average_concurrency"]["value"] for window in windows] == [
        1.5, 1.5, 1.0,
    ]
    assert [window["peak_concurrency"]["value"] for window in windows] == [
        2, 2, 1,
    ]


def test_empty_window_keeps_zero_throughput_and_null_rates():
    windows = rows(
        time_windows.time_window_result([success(0, 0.0, 1000.0)], 120.0)
    )
    empty = windows[1]
    assert empty["completed_requests"]["value"] == 0
    assert empty["output_token_throughput"]["value"] == 0
    assert empty["success_rate"]["value"] is None
    assert empty["success_rate"]["reason"] == (
        "no completed requests in window"
    )
    assert empty["timeout_rate"]["value"] is None
    assert empty["ttft_ms"]["count"] == 0
    assert empty["ttft_ms"]["mean"] is None
    assert empty["ttft_ms"]["p99"] is None
    assert "low_sample" not in empty["ttft_ms"]


def test_all_failed_window_has_zero_rates_not_null():
    requests = [
        failed(0, 0.0, 1000.0, "ReadTimeout"),
        failed(1, 0.0, 2000.0, "HTTPStatusError"),
    ]
    window = rows(time_windows.time_window_result(requests, 60.0))[0]
    assert window["completed_requests"]["value"] == 2
    assert window["successful"]["value"] == 0
    assert window["timeout_failed"]["value"] == 1
    assert window["other_failed"]["value"] == 1
    assert window["success_rate"]["value"] == 0.0
    assert window["timeout_rate"]["value"] == 0.5
    assert window["output_token_throughput"]["value"] == 0


def test_missing_output_tokens_null_throughput_with_reason():
    requests = [
        success(0, 0.0, 1000.0),
        {**success(1, 0.0, 2000.0), "output_tokens": None},
    ]
    window = rows(time_windows.time_window_result(requests, 60.0))[0]
    throughput = window["output_token_throughput"]
    assert throughput["value"] is None
    assert "1 of 2" in throughput["reason"]
    assert window["ttft_ms"]["count"] == 2


def test_timeout_classification_uses_explicit_error_types():
    timeouts = [
        "ConnectTimeout", "ReadTimeout", "WriteTimeout",
        "PoolTimeout", "TimeoutError", "TimeoutException",
    ]
    others = ["RuntimeError", "HTTPStatusError", "ConnectError", None]
    requests = [
        failed(index, 0.0, 1000.0, kind)
        for index, kind in enumerate(timeouts + others)
    ]
    window = rows(time_windows.time_window_result(requests, 60.0))[0]
    assert window["timeout_failed"]["value"] == len(timeouts)
    assert window["other_failed"]["value"] == len(others)
    assert window["completed_requests"]["value"] == len(requests)


def test_low_sample_flag_below_ten_latency_samples():
    nine = [success(index, 0.0, 1000.0, ttft_ms=float(index))
            for index in range(9)]
    window = rows(time_windows.time_window_result(nine, 60.0))[0]
    assert window["ttft_ms"]["count"] == 9
    assert window["ttft_ms"]["low_sample"] is True
    assert window["ttft_ms"]["p99"] is not None
    ten = nine + [success(9, 0.0, 1000.0, ttft_ms=9.0)]
    window = rows(time_windows.time_window_result(ten, 60.0))[0]
    assert window["ttft_ms"]["count"] == 10
    assert "low_sample" not in window["ttft_ms"]


def test_unusable_offsets_are_reported_not_dropped_silently():
    requests = [
        success(0, 0.0, 1000.0),
        {**success(1, 0.0, 2000.0), "end_offset_ms": None},
        {**success(2, 5000.0, 4000.0)},
    ]
    result = time_windows.time_window_result(requests, 60.0)
    unattributed = result["metadata"]["unattributed_requests"]
    assert unattributed["count"] == 2
    assert unattributed["request_indexes"] == [1, 2]
    window = rows(result)[0]
    assert window["started_requests"]["value"] == 3
    assert window["completed_requests"]["value"] == 1


def test_invalid_window_and_duration_handling():
    with pytest.raises(ValueError):
        time_windows.time_window_rows([success(0, 0.0, 1.0)], 60.0, 0)
    with pytest.raises(ValueError):
        time_windows.time_window_rows([success(0, 0.0, 1.0)], 60.0, 1.5)
    result = time_windows.time_window_result([success(0, 0.0, 1.0)], 0.0)
    assert result["status"] == "unavailable"
    assert result["values"]["windows"] == []


def test_metric_compute_uses_parameter_then_metadata_then_default():
    requests = [success(0, 0.0, 1000.0)]
    explicit = StabilityTrends().compute(facts(requests, 120.0, window=30))
    assert explicit["values"]["window_seconds"] == 30
    assert len(rows(explicit)) == 4
    legacy = StabilityTrends().compute(
        facts(requests, 120.0, metadata={"trend_window_seconds": 20})
    )
    assert legacy["values"]["window_seconds"] == 20
    default = StabilityTrends().compute(facts(requests, 120.0))
    assert default["values"]["window_seconds"] == 60
    assert default["status"] == "success"
    assert default["observation_boundary"] == "client"
    metadata = default["metadata"]
    assert metadata["warmup_included"] is False
    assert metadata["attribution"]["completed_requests"] == "end_offset_ms"
    assert metadata["attribution"]["started_requests"] == "start_offset_ms"
    assert "completion-attributed" in metadata["throughput_definition"]
    assert metadata["timeout_error_types"] == sorted(
        time_windows.TIMEOUT_ERROR_TYPES
    )


def test_registry_requires_client_stream_timing():
    computed = compute_metrics(
        ["stability_trends"],
        {"client_stream_timing"},
        facts([success(0, 0.0, 1000.0)], 60.0),
    )
    assert computed["stability_trends"]["status"] == "success"
    with pytest.raises(ValueError):
        compute_metrics(
            ["stability_trends"], set(), facts([success(0, 0.0, 1.0)], 60.0)
        )


def test_result_is_json_serializable_without_nan():
    requests = [success(0, 0.0, 1000.0), failed(1, 0.0, 2000.0)]
    json.dumps(
        time_windows.time_window_result(requests, 60.0), allow_nan=False
    )


def test_default_metrics_enable_only_online_scenarios():
    online = generation_benchmarks.OnlineServingBenchmark
    multi = multi_turn_serving.MultiTurnServingBenchmark
    engine = generation_benchmarks.OfflineEngineBenchmark
    async_engine = generation_benchmarks.AsyncEngineBenchmark
    cache = prefix_cache.PrefixCacheBenchmark
    assert "stability_trends" in online.default_metrics
    assert "stability_trends" in multi.default_metrics
    assert "stability_trends" not in engine.default_metrics
    assert "stability_trends" not in async_engine.default_metrics
    assert "stability_trends" not in cache.default_metrics
    assert cache.default_metrics == (
        "streaming_performance", "prefix_cache_hit_rate",
    )


def test_window_parameter_declared_without_matrix_or_summary():
    for scenario in (
        generation_benchmarks.OnlineServingBenchmark,
        multi_turn_serving.MultiTurnServingBenchmark,
        prefix_cache.PrefixCacheBenchmark,
    ):
        schema = scenario.parameter_schema()
        parameter = schema["trend_window_seconds"]
        assert parameter.default == 60
        assert parameter.minimum == 1
        assert parameter.matrix is False
    engine = generation_benchmarks.OfflineEngineBenchmark
    assert "trend_window_seconds" not in engine.parameter_schema()
    from luban_meter.core.benchmark import Benchmark
    assert "trend_window_seconds" not in Benchmark.summary_fields


def test_windows_array_is_not_flattened_into_metric_tables():
    metric = StabilityTrends().compute(
        facts([success(0, 0.0, 1000.0)], 120.0)
    )
    values = metric["values"]
    groups = [path for path, _ in metric_groups(values)]
    assert groups == ["window_seconds"]
    table = list(metric_rows(values))
    assert [row["metric"] for row in table] == ["/window_seconds"]
    rows = list(metric_rows({"stability_trends": metric}))
    assert [row["metric"] for row in rows] == [
        "/stability_trends/values/window_seconds"
    ]


def test_saved_smoke_response_envelope():
    request = {
        "request_id": "request-0", "status": "success",
        "start_offset_s": 0.0006648099988524336,
        "end_offset_s": 0.9376656970016484,
        "response": {
            "start_offset_ms": 0.87, "end_offset_ms": 937.506,
            "status": "success", "input_tokens": 128, "output_tokens": 32,
            "ttft_ms": 109.951, "e2el_ms": 936.636,
        },
    }
    metric = StabilityTrends().compute(facts([request], 14.570279785999446))
    assert metric["metadata"]["coverage"]["completion_eligible"] == 1
    row = metric["values"]["windows"][0]
    assert row["started_requests"]["value"] == 1
    assert row["ttft_ms"]["p99"] == 109.951
    assert row["output_token_throughput"]["value"] == 2.196


def test_all_excluded_throughput_is_unknown():
    result = StabilityTrends().compute(facts([{"status": "failed"}], 1))
    node = result["values"]["windows"][0]["output_token_throughput"]
    assert node["value"] is None
    assert node["reason"]


def test_existing_streaming_metric_values_unchanged():
    from luban_meter.benchmarking.benchmark_metrics import (
        StreamingPerformance,
    )

    requests = [
        {
            "sample_id": "0",
            "status": "success",
            "response": success(0, 0.0, 10000.0, tokens=20, ttft_ms=100.0),
        }
    ]
    computed = StreamingPerformance().compute(
        facts(requests, 60.0)
    )
    service = computed["values"]["service_view"]
    assert service["output_token_throughput"]["value"] == pytest.approx(
        20 / 60.0, abs=1e-3
    )
    assert computed["values"]["request_view"]["ttft"]["mean"] == 100.0


@pytest.mark.parametrize(
    ("start", "end", "completed", "peak", "excluded", "missing"),
    [
        (0, 60000, 1, 0, 0, 0),
        (60000, 120000, 1, 1, 0, 0),
        (60000, 60000, 1, 0, 0, 0),
        (None, 121000, 0, 0, 1, 0),
        (None, 120000, 1, 0, 0, 1),
        (-1, 120000, 1, 0, 0, 1),
        (70000, 60000, 0, 0, 1, 0),
        (0, -1, 0, 0, 1, 0),
        (0, float("nan"), 0, 0, 1, 0),
        (0, float("inf"), 0, 0, 1, 0),
    ],
)
def test_formal_offset_coverage(
    start, end, completed, peak, excluded, missing
):
    record = {"start_offset_ms": start, "end_offset_ms": end}
    result = time_windows.time_window_result([record], 120)
    window = result["values"]["windows"][1]
    coverage = result["metadata"]["coverage"]
    assert window["completed_requests"]["value"] == completed
    assert window["peak_concurrency"]["value"] == peak
    assert coverage["completion_excluded"] == excluded
    assert coverage["completion_without_valid_start"] == missing
    assert coverage["concurrency_eligible"] == completed - missing


def test_boundary_reproduction_and_adjacent_request():
    records = [{"start_offset_ms": 0, "end_offset_ms": 60000},
               {"end_offset_ms": 121000}]
    for adjacent in (False, True):
        if adjacent:
            records.append({"start_offset_ms": 60000, "end_offset_ms": 120000})
        result = time_windows.time_window_result(records, 120)
        window = result["values"]["windows"][1]
        assert window["completed_requests"]["value"] == 1 + adjacent
        assert window["peak_concurrency"]["value"] == int(adjacent)
        assert window["average_concurrency"]["value"] == int(adjacent)
        meta = result["metadata"]
        assert meta["unattributed_requests"]["request_indexes"] == [1]
        assert meta["coverage"]["exclusion_reasons"] == {
            "end_after_formal_duration": 1
        }


def test_invalid_duration_explains_all_exclusions():
    result = time_windows.time_window_result([{"end_offset_ms": 1}], 0)
    assert result["values"]["windows"] == []
    assert result["metadata"]["coverage"]["exclusion_reasons"] == {
        "invalid_formal_duration": 1
    }


@pytest.mark.parametrize("duration,count,expected", [
    (35.5, 32, 15), (75.178, 38, 30), (8, 5, 10), (75, 150, 10),
])
def test_auto_sample_supported_windows(duration, count, expected):
    records = [
        success(index, 0, (index + 0.5) * duration * 1000 / count)
        for index in range(count)
    ]
    result = time_windows.time_window_result(records, duration, "auto")
    assert result["values"]["window_seconds"] == expected
    assert result["metadata"]["window_mode"] == "auto"
    fixed = time_windows.time_window_result(records, duration, expected)
    assert result["values"] == fixed["values"]
    assert result == time_windows.time_window_result(records, duration, "auto")


def test_auto_uneven_failed_empty_and_invalid():
    concentrated = [success(index, 0, index + 1) for index in range(32)]
    result = time_windows.time_window_result(concentrated, 75, "auto")
    assert result["metadata"]["window_selection"]["fallback"]
    assert len(rows(result)) == 1
    failures = [failed(index, 0, index * 2000) for index in range(38)]
    result = time_windows.time_window_result(failures, 75.178, "auto")
    assert result["values"]["window_seconds"] == 30
    assert all(row["ttft_ms"]["mean"] is None for row in rows(result))
    empty = time_windows.time_window_result([], 75, "auto")
    assert empty["metadata"]["window_selection"]["fallback"]
    invalid = time_windows.time_window_result(
        [{"end_offset_ms": -1}], 75, "auto"
    )
    assert rows(invalid)[0]["output_token_throughput"]["value"] is None
    unavailable = time_windows.time_window_result([], 0, "auto")
    assert unavailable["status"] == "unavailable"


def test_auto_nested_records_and_exact_final_boundary():
    records = [
        {"response": success(index, 0, (index + 1) * 1000)}
        for index in range(32)
    ]
    metric = StabilityTrends().compute(facts(records, 35.5, "auto"))
    width = metric["values"]["window_seconds"]
    fixed = time_windows.time_window_result(
        [record["response"] for record in records], 35.5, width
    )
    assert metric["values"] == fixed["values"]
    exact = [success(index, 0, index * 3000) for index in range(41)]
    chosen = time_windows.time_window_result(exact, 120, "auto")
    completed = sum(
        row["completed_requests"]["value"] for row in rows(chosen)
    )
    assert completed == 41
    assert chosen["metadata"]["coverage"]["completion_excluded"] == 0


@pytest.mark.parametrize(
    "value,expected", [(60, 60), ("60", 60), ("auto", "auto")]
)
def test_window_parameter_accepts_auto_and_fixed(value, expected):
    parameter = generation_benchmarks.TREND_PARAMETERS["trend_window_seconds"]
    assert parameter.validate("trend_window_seconds", value) == expected
    assert parameter.validate("trend_window_seconds", parameter.default) == 60


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1.5", "AUTO", None])
def test_window_parameter_rejects_invalid(value):
    parameter = generation_benchmarks.TREND_PARAMETERS["trend_window_seconds"]
    with pytest.raises(ValueError):
        parameter.validate("trend_window_seconds", value)
