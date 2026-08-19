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
    """Return the common descriptive-statistics contract for one metric."""
    values = [float(value) for value in samples]
    summary: dict[str, Any] = {
        "unit": unit,
        "count": len(values),
        "mean": None,
        "median": None,
        "p50": None,
        "p90": None,
        "p99": None,
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
            "p99": round(percentile(values, 0.99), 3),
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
