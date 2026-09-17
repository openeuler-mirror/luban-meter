"""Metric aggregation shared by generative Benchmarks."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any


def percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize(samples: Sequence[float], unit: str) -> dict[str, Any]:
    """Return the common descriptive-statistics contract for one
    metric.
    """
    values = [float(value) for value in samples]
    summary: dict[str, Any] = {
        "unit": unit,
        "count": len(values),
        "mean": None,
        "median": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "p999": None,
        "min": None,
        "max": None,
        "stddev": None,
    }
    if not values:
        return summary

    median = percentile(values, 0.50)
    summary.update(
        {
            "mean": round(statistics.mean(values), 3),
            "median": round(median, 3),
            "p50": round(median, 3),
            "p90": round(percentile(values, 0.90), 3),
            "p95": round(percentile(values, 0.95), 3),
            "p99": round(percentile(values, 0.99), 3),
            "p999": round(percentile(values, 0.999), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "stddev": round(statistics.pstdev(values), 3),
        }
    )
    return summary


def scalar(value: float, unit: str, precision: int = 3) -> dict[str, Any]:
    if isinstance(value, float):
        value = round(value, precision)
    return {"value": value, "unit": unit}


def summarize_weighted(
    samples: Sequence[tuple[float, int]], unit: str
) -> dict[str, Any]:
    """Summarize request values with token weights and CDF quantiles.

    ``count`` retains the number of requests; ``weight_sum`` records
    the number of tokens or token intervals represented by those
    requests.
    """
    values: list[tuple[float, int]] = []
    for value, weight in samples:
        if (
            not math.isfinite(value)
            or value < 0
            or not isinstance(weight, int)
            or isinstance(weight, bool)
            or weight <= 0
        ):
            raise ValueError("expected finite values and positive weights")
        values.append((float(value), weight))
    values.sort()
    total_weight = sum(weight for _, weight in values)
    summary = summarize([], unit)
    summary.update(
        count=len(values),
        weight_sum=total_weight,
        aggregation="token_weighted",
        percentile_method="weighted_cdf",
    )
    if not values:
        return summary

    mean = math.fsum(
        value * (weight / total_weight) for value, weight in values
    )
    variance = math.fsum(
        (value - mean) ** 2 * (weight / total_weight)
        for value, weight in values
    )
    for name, fraction in (
        ("p50", 0.50),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
        ("p999", 0.999),
    ):
        cumulative = 0
        for value, weight in values:
            cumulative += weight
            if cumulative >= fraction * total_weight:
                summary[name] = round(value, 3)
                break
    summary.update(
        mean=round(mean, 3),
        median=summary["p50"],
        min=round(values[0][0], 3),
        max=round(values[-1][0], 3),
        stddev=round(math.sqrt(variance), 3),
    )
    return summary
