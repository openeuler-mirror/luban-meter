"""Tests for service observation aggregation (queue time window mean)."""

import pytest

from luban_meter.benchmarking.generation_performance import (
    service_observation,
)
from luban_meter.benchmarking.generation_performance.common.prometheus import (
    parse_prometheus_text,
)
from luban_meter.reporting.data import metric_rows

QUEUE_TYPE = "# TYPE vllm:request_queue_time_seconds histogram"


def _snapshot(*lines, error=None):
    text = "\n".join([QUEUE_TYPE, *lines])
    snap = {"finished_at": 0.0, "metrics": parse_prometheus_text(text)}
    if error is not None:
        snap["error"] = error
    return snap


def _process(*snaps):
    return service_observation.process(
        {
            "metrics": {"snapshots": list(snaps)},
            "metadata": {"collect_duration": 10.0},
        }
    )


def _window(result):
    return result["metrics"]["latency_decomposition"]["queue_time_window"]


def _series_lines(sum_value, count_value, model=None):
    labels = f'{{model="{model}"}}' if model else ""
    return [
        f"vllm:request_queue_time_seconds_sum{labels} {sum_value}",
        f"vllm:request_queue_time_seconds_count{labels} {count_value}",
    ]


def test_window_delta_mean_with_history_baseline():
    first = _snapshot(
        "vllm:request_queue_time_seconds_bucket{le=\"0.1\"} 10",
        "vllm:request_queue_time_seconds_bucket{le=\"+Inf\"} 50",
        *_series_lines(100.0, 50),
    )
    last = _snapshot(*_series_lines(112.5, 80))

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(12.5)
    assert window["delta_count"] == {"value": 30, "unit": "request"}
    assert window["series_count"] == {"value": 1, "unit": "series"}
    assert window["mean"] == pytest.approx(12.5 / 30)
    # The legacy cumulative-quantile view stays available alongside.
    latency = _process(first, last)["metrics"]["latency_decomposition"]
    assert "queue_time" in latency


def test_issue2_acceptance_example():
    # Non-zero history baseline must not leak into the window mean:
    # start 1000/100, end 1030/120 -> mean 1.5 s.
    first = _snapshot(*_series_lines(1000.0, 100))
    last = _snapshot(*_series_lines(1030.0, 120))

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(30.0)
    assert window["delta_count"] == {"value": 20, "unit": "request"}
    assert window["mean"] == pytest.approx(1.5)


def test_zero_wait_observations_report_real_zero():
    # All requests in the window waited about zero seconds; the mean is
    # a real zero, not unavailable.
    first = _snapshot(*_series_lines(5.0, 100))
    last = _snapshot(*_series_lines(5.0, 130))

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(0.0)
    assert window["delta_count"] == {"value": 30, "unit": "request"}
    assert window["mean"] == 0.0
    assert "reason" not in window


def test_multi_label_series_pool_before_averaging():
    first = _snapshot(
        *_series_lines(100.0, 50, model="m1"),
        *_series_lines(200.0, 100, model="m2"),
    )
    last = _snapshot(
        *_series_lines(112.5, 80, model="m1"),
        *_series_lines(203.0, 104, model="m2"),
    )

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(12.5 + 3.0)
    assert window["delta_count"] == {"value": 34, "unit": "request"}
    assert window["series_count"] == {"value": 2, "unit": "series"}
    assert window["mean"] == pytest.approx(15.5 / 34)
    # Pooling is not the average of the per-series means.
    assert window["mean"] != pytest.approx((12.5 / 30 + 3.0 / 4) / 2)


def test_new_series_starts_from_zero_baseline():
    first = _snapshot(*_series_lines(100.0, 50, model="m1"))
    last = _snapshot(
        *_series_lines(112.5, 80, model="m1"),
        *_series_lines(5.0, 10, model="m2"),
    )

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(12.5 + 5.0)
    assert window["delta_count"] == {"value": 40, "unit": "request"}


def test_series_missing_from_last_snapshot_dropped():
    first = _snapshot(
        *_series_lines(100.0, 50, model="m1"),
        *_series_lines(200.0, 100, model="m2"),
    )
    last = _snapshot(*_series_lines(112.5, 80, model="m1"))

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(12.5)
    assert window["delta_count"] == {"value": 30, "unit": "request"}
    assert window["series_count"] == {"value": 1, "unit": "series"}
    # The disappeared series is recorded with its reason, not silently
    # skipped.
    assert window["dropped_series"] == [{
        "labels": {"model": "m2"},
        "reason": "series missing from the last snapshot",
    }]


def test_counter_reset_series_dropped():
    first = _snapshot(
        *_series_lines(100.0, 50, model="m1"),
        *_series_lines(200.0, 100, model="m2"),
    )
    last = _snapshot(
        *_series_lines(112.5, 80, model="m1"),
        *_series_lines(3.0, 2, model="m2"),
    )

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(12.5)
    assert window["delta_count"] == {"value": 30, "unit": "request"}
    assert window["series_count"] == {"value": 1, "unit": "series"}


def test_mid_window_counter_reset_detected_after_recovery():
    # 10/100 -> 1/10 -> 30/200: the series regresses mid-window and later
    # exceeds its start again; the reset must still be detected instead
    # of producing the misleading (30-10)/(200-100)=0.2 s.
    snaps = [
        _snapshot(*_series_lines(10.0, 100, model="m1")),
        _snapshot(*_series_lines(1.0, 10, model="m1")),
        _snapshot(*_series_lines(30.0, 200, model="m1")),
    ]

    window = _window(_process(*snaps))

    assert window["mean"] is None
    assert window["delta_sum"] is None
    assert window["series_count"] == {"value": 0, "unit": "series"}
    assert window["dropped_series"] == [{
        "labels": {"model": "m1"},
        "reason": "counter reset between snapshots",
    }]


def test_mid_window_reset_only_drops_affected_series():
    first = _snapshot(
        *_series_lines(100.0, 50, model="m1"),
        *_series_lines(10.0, 100, model="m2"),
    )
    reset = _snapshot(
        *_series_lines(102.0, 52, model="m1"),
        *_series_lines(1.0, 10, model="m2"),
    )
    last = _snapshot(
        *_series_lines(112.5, 80, model="m1"),
        *_series_lines(30.0, 200, model="m2"),
    )

    window = _window(_process(first, reset, last))

    # m2 regressed mid-window and later exceeded its start again; it must
    # be dropped while the untouched m1 series still contributes.
    assert window["delta_sum"] == pytest.approx(12.5)
    assert window["delta_count"] == {"value": 30, "unit": "request"}
    assert window["series_count"] == {"value": 1, "unit": "series"}
    assert window["dropped_series"] == [{
        "labels": {"model": "m2"},
        "reason": "counter reset between snapshots",
    }]


def test_zero_count_delta_keeps_reason_without_zero():
    only = _snapshot(*_series_lines(100.0, 50))

    window = _window(_process(only))

    assert window["mean"] is None
    assert window["delta_sum"] is None
    assert window["delta_count"] is None
    assert window["series_count"] == {"value": 1, "unit": "series"}
    assert "reason" in window


def test_absent_metric_omits_key():
    snap = {
        "finished_at": 0.0,
        "metrics": parse_prometheus_text(
            "# TYPE vllm:num_requests_waiting gauge\n"
            "vllm:num_requests_waiting 3\n"
        ),
    }

    result = _process(snap)

    latency = result["metrics"]["latency_decomposition"]
    assert "queue_time_window" not in latency
    assert "queue_time" not in latency


def test_error_snapshots_ignored():
    poisoned = _snapshot(
        *_series_lines(500.0, 300),
        error="connection refused",
    )
    first = _snapshot(*_series_lines(100.0, 50))
    last = _snapshot(*_series_lines(112.5, 80))

    window = _window(_process(poisoned, first, last))

    assert window["delta_sum"] == pytest.approx(12.5)
    assert window["delta_count"] == {"value": 30, "unit": "request"}


def test_mean_recomputable_from_persisted_deltas():
    first = _snapshot(*_series_lines(100.0, 50))
    last = _snapshot(*_series_lines(113.7, 83))

    window = _window(_process(first, last))

    # Values persist at original precision, so the saved mean equals the
    # quotient of the saved numerator and denominator exactly.
    assert window["mean"] == (
        window["delta_sum"] / window["delta_count"]["value"]
    )


def test_raw_precision_survives_persistence():
    # 0.4 microseconds of total wait over 40 requests: rounding before
    # persistence would save 0 and hide the difference from a real zero.
    first = _snapshot(*_series_lines(5.0, 100))
    last = _snapshot(*_series_lines(5.0000004, 140))

    window = _window(_process(first, last))

    assert window["delta_sum"] == pytest.approx(4e-07)
    assert window["delta_count"] == {"value": 40, "unit": "request"}
    assert window["mean"] == pytest.approx(1e-08)
    assert window["mean"] > 0
    assert window["series"][0]["delta_sum"] == pytest.approx(4e-07)


def test_html_rows_carry_correct_units():
    first = _snapshot(*_series_lines(100.0, 50))
    last = _snapshot(*_series_lines(112.5, 80))

    window = _window(_process(first, last))
    rows = {
        row["metric"]: row
        for row in metric_rows(window, "queue_time_window")
    }

    prefix = "queue_time_window/"
    assert rows[prefix + "mean"]["value"] == pytest.approx(12.5 / 30)
    assert rows[prefix + "mean"]["unit"] == "s"
    assert rows[prefix + "delta_sum"]["unit"] == "s"
    assert rows[prefix + "delta_count"]["unit"] == "request"
    assert rows[prefix + "series_count"]["unit"] == "series"
    # Per-series evidence rows keep their own units too.
    assert rows[prefix + "series/0/delta_sum"]["unit"] == "s"
    assert (rows[prefix + "series/0/delta_count"]["unit"]
            == "request")
    assert rows[prefix + "window/started_at"]["unit"] == "unix_s"
    assert rows[prefix + "window/ended_at"]["unit"] == "unix_s"
    assert set(rows) == {
        prefix + "mean",
        prefix + "delta_sum",
        prefix + "delta_count",
        prefix + "series_count",
        prefix + "series/0/delta_sum",
        prefix + "series/0/delta_count",
        prefix + "window/started_at",
        prefix + "window/ended_at",
    }


def test_window_persists_boundary_and_series_evidence():
    first = _snapshot(*_series_lines(100.0, 50, model="m1"))
    last = _snapshot(*_series_lines(112.5, 80, model="m1"))

    first["finished_at"] = 1000.5
    last["finished_at"] = 1010.5
    window = _window(_process(first, last))

    assert window["boundary"] == "server"
    assert window["window"] == {
        "started_at": {"value": 1000.5, "unit": "unix_s"},
        "ended_at": {"value": 1010.5, "unit": "unix_s"},
    }
    assert window["series"] == [{
        "labels": {"model": "m1"},
        "delta_sum": pytest.approx(12.5),
        "delta_count": {"value": 30, "unit": "request"},
    }]
    # The mean is recomputable from the persisted per-series deltas.
    assert window["mean"] == pytest.approx(
        window["series"][0]["delta_sum"]
        / window["series"][0]["delta_count"]["value"]
    )


def test_dropped_series_recorded_with_reason():
    first = _snapshot(
        *_series_lines(100.0, 50, model="m1"),
        *_series_lines(200.0, 100, model="m2"),
    )
    last = _snapshot(
        *_series_lines(112.5, 80, model="m1"),
        *_series_lines(3.0, 2, model="m2"),
    )

    window = _window(_process(first, last))

    assert window["dropped_series"] == [{
        "labels": {"model": "m2"},
        "reason": "counter reset between snapshots",
    }]
    assert window["series_count"]["value"] == 1
    assert [entry["labels"] for entry in window["series"]] == [
        {"model": "m1"}
    ]


def test_html_rows_without_data_keep_only_series_count():
    only = _snapshot(*_series_lines(100.0, 50))

    window = _window(_process(only))
    rows = list(metric_rows(window, "queue_time_window"))

    # mean/delta rows disappear instead of showing zero or None; the
    # series evidence and boundary stay visible for recomputation.
    metrics = [row["metric"] for row in rows]
    assert metrics == [
        "queue_time_window/window/started_at",
        "queue_time_window/window/ended_at",
        "queue_time_window/series/0/delta_sum",
        "queue_time_window/series/0/delta_count",
        "queue_time_window/series_count",
    ]
    assert rows[-1]["value"] == 1
    assert rows[-1]["unit"] == "series"
