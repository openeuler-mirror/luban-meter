"""Regression coverage for the unbounded classic histogram bucket."""

import json

import pytest

from luban_meter.benchmarking.generation_performance.common.prometheus import (
    histogram_quantiles,
    parse_prometheus_text,
)
from luban_meter.reporting.html import write_html
from luban_meter.runtime.prometheus_observer import PrometheusObserver

METRIC = "vllm:time_to_first_token_seconds"


def histogram(buckets, *, include_count=True, sum_value=12):
    lines = [f"# TYPE {METRIC} histogram"]
    lines.extend(
        f'{METRIC}_bucket{{le="{bound}"}} {count}'
        for bound, count in buckets
    )
    lines.append(f"{METRIC}_sum {sum_value}")
    if include_count:
        lines.append(f"{METRIC}_count {buckets[-1][1]}")
    return parse_prometheus_text("\n".join(lines))


@pytest.mark.parametrize("include_count", [True, False])
def test_quantiles_in_unbounded_bucket_use_last_finite_bound(include_count):
    metrics = histogram(
        [("1", 5), ("2", 8), ("+Inf", 10)],
        include_count=include_count,
    )
    result = histogram_quantiles(metrics[METRIC]["samples"], (0.5, 0.9, 1))
    assert result == {
        "p50": 1.0, "p90": 2.0, "p100": 2.0,
        "_sum": 12.0, "_count": 10.0,
    }


def test_finite_bucket_interpolation_is_preserved():
    metrics = histogram([("1", 5), ("2", 10), ("+Inf", 10)])
    result = histogram_quantiles(metrics[METRIC]["samples"], (0.75, 1))
    assert result["p75"] == 1.5
    assert result["p100"] == 2.0


@pytest.mark.parametrize("trailing", ["missing", "failed", "no_buckets"])
def test_observation_quantiles_use_last_valid_histogram_snapshot(trailing):
    observer = PrometheusObserver("unused", 1, "", lambda *args: None)
    observer.snapshots = [
        {"finished_at": 1.0, "metrics": histogram([
            ("1", 90), ("10", 100), ("+Inf", 100),
        ], sum_value=100)},
        {"finished_at": 2.0, "metrics": histogram([
            ("1", 90), ("10", 200), ("+Inf", 200),
        ], sum_value=600)},
        {"finished_at": 3.0, "metrics": {}},
    ]
    if trailing == "failed":
        observer.snapshots[-1]["error"] = "synthetic scrape failure"
    if trailing == "no_buckets":
        observer.snapshots[-1]["metrics"] = parse_prometheus_text(
            f"# TYPE {METRIC} histogram\n{METRIC}_count 210"
        )
    latency = observer.result()["metrics"]["latency_decomposition"]["ttft"]
    # The last cumulative distribution has rank 100 inside (1, 10].
    assert latency["p50"] == pytest.approx(1.818)
    assert latency["p90"] == pytest.approx(8.364)
    assert latency["count"] == 200
    assert latency["sum"] == 600


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("include_count", [False, True])
def test_histogram_pools_label_series_independently_of_order(
    reverse, include_count,
):
    groups = [
        ("A", [("1", 100), ("10", 100), ("+Inf", 100)], 50),
        ("B", [("1", 0), ("10", 100), ("+Inf", 100)], 500),
    ]
    lines = [f"# TYPE {METRIC} histogram"]
    for model, buckets, total_sum in reversed(groups) if reverse else groups:
        lines.extend(
            f'{METRIC}_bucket{{model="{model}",le="{bound}"}} {count}'
            for bound, count in buckets
        )
        lines.append(f'{METRIC}_sum{{model="{model}"}} {total_sum}')
        if include_count:
            lines.append(f'{METRIC}_count{{model="{model}"}} 100')
    metrics = parse_prometheus_text("\n".join(lines))
    result = histogram_quantiles(metrics[METRIC]["samples"], (0.5, 0.9))
    assert result["p50"] == 1
    assert result["p90"] == pytest.approx(8.2)
    assert result["_count"] == 200
    assert result["_sum"] == 550


def test_histogram_rejects_incompatible_label_bucket_boundaries():
    metrics = parse_prometheus_text("\n".join([
        f"# TYPE {METRIC} histogram",
        f'{METRIC}_bucket{{worker="A",le="1"}} 5',
        f'{METRIC}_bucket{{worker="A",le="+Inf"}} 10',
        f'{METRIC}_bucket{{worker="B",le="2"}} 5',
        f'{METRIC}_bucket{{worker="B",le="+Inf"}} 10',
    ]))
    with pytest.raises(ValueError, match="different bucket boundaries"):
        histogram_quantiles(metrics[METRIC]["samples"])


@pytest.mark.parametrize("count", [0, 10])
def test_no_finite_bucket_has_no_estimate(count):
    metrics = histogram([("+Inf", count)])
    result = histogram_quantiles(metrics[METRIC]["samples"])
    assert [result[key] for key in ("p50", "p90", "p99")] == [None] * 3


def test_zero_observations_keep_measured_count_and_sum():
    metrics = histogram([("1", 0), ("+Inf", 0)], sum_value=0)
    result = histogram_quantiles(metrics[METRIC]["samples"])
    assert result == {
        "p50": None, "p90": None, "p99": None, "_sum": 0, "_count": 0,
    }


@pytest.mark.parametrize("earlier_snapshot", [False, True])
def test_unbounded_histogram_observation_can_be_exported_to_html(
    tmp_path, earlier_snapshot,
):
    observer = PrometheusObserver("unused", 1, "", lambda *args: None)
    observer.snapshots = [{
        "started_at": 1.0,
        "finished_at": 2.0,
        "metrics": histogram([("1", 5), ("2", 8), ("+Inf", 10)]),
    }]
    if earlier_snapshot:
        observer.snapshots.insert(0, {
            "started_at": 0.0,
            "finished_at": 1.0,
            "metrics": histogram([("1", 2), ("2", 4), ("+Inf", 5)]),
        })
    observation = observer.result()
    payload = {
        "kind": "task", "status": "success", "name": "demo",
        "category": "generation_performance",
        "cases": [{
            "id": "case_1", "status": "success",
            "observation": observation,
        }],
    }
    destination = tmp_path / "report.html"
    write_html(payload, destination)
    assert destination.is_file()
    json.dumps(observation, allow_nan=False)
    latency = observation["metrics"]["latency_decomposition"]["ttft"]
    assert latency["p90"] == latency["p99"] == 2.0
    assert latency["count"] == 10
    assert latency["sum"] == 12.0
