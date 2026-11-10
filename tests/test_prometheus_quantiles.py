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


def histogram(buckets, *, include_count=True):
    lines = [f"# TYPE {METRIC} histogram"]
    lines.extend(
        f'{METRIC}_bucket{{le="{bound}"}} {count}'
        for bound, count in buckets
    )
    lines.append(f"{METRIC}_sum 12")
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


@pytest.mark.parametrize("count", [0, 10])
def test_no_finite_bucket_has_no_estimate(count):
    metrics = histogram([("+Inf", count)])
    result = histogram_quantiles(metrics[METRIC]["samples"])
    assert [result[key] for key in ("p50", "p90", "p99")] == [None] * 3


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
