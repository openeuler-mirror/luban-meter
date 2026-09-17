"""SLO, circuit breaker and stop-constraint helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from luban_meter.benchmarking.generation_performance.common.parameters import (
    optional_positive_number,
)
from luban_meter.benchmarking.generation_performance.common.statistics import (
    percentile,
)


def slo_config(parameters: Mapping[str, Any]) -> dict[str, float] | None:
    """Extract Goodput SLO thresholds (per-request, not case-level).

    Only ttft_ms, tpot_ms, and e2el_ms are SLO dimensions.
    The case-level circuit breaker is configured separately via
    ``circuit_breaker`` in parameters.
    """
    raw = parameters.get("slo")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("slo must be an object")
    config: dict[str, float] = {}
    for name in ("ttft_ms", "tpot_ms", "e2el_ms"):
        threshold = optional_positive_number(raw, name)
        if threshold is not None:
            config[name] = threshold
    if not config:
        raise ValueError("slo must contain at least one threshold")
    return config


def circuit_breaker_config(
    parameters: Mapping[str, Any],
) -> float | None:
    """Extract circuit breaker p99 E2EL threshold (case-level).

    Independent of SLO: the circuit breaker stops subsequent cases
    when a case's P99 E2EL exceeds this threshold.
    """
    raw = parameters.get("circuit_breaker")
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return optional_positive_number(raw, "p99_e2el_ms")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if value > 0 and math.isfinite(value):
            return value
    raise ValueError(
        "circuit_breaker must be a positive number or "
        "an object with p99_e2el_ms"
    )


def check_stop_constraints(
    case: Mapping[str, Any],
    *,
    max_duration: float | None,
    max_error_rate: float | None,
) -> str | None:
    """Check if a completed case violates stop constraints.

    Returns a skip reason string if violated, None otherwise.
    """
    if max_error_rate is not None:
        requests = case.get("requests", [])
        if isinstance(requests, list) and requests:
            failed = sum(
                1
                for request_record in requests
                if isinstance(request_record, Mapping)
                and request_record.get("status") != "success"
            )
            rate = failed / len(requests)
            if rate > max_error_rate:
                return "max_error_rate_exceeded"

    if max_duration is not None:
        duration = case.get("benchmark_duration_seconds")
        if (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration > max_duration
        ):
            return "max_duration_exceeded"

    return None


def case_p99_e2el_ms(case: Mapping[str, Any]) -> float | None:
    """Compute P99 E2EL from successful requests in a case.

    Returns None when fewer than 10 successful samples are available,
    because P99 is statistically unstable for small sample sizes.
    """
    successful = [
        record
        for record in case.get("requests", [])
        if isinstance(record, Mapping) and record.get("status") == "success"
    ]
    if len(successful) < 10:
        return None
    e2el_values = [
        float(record["e2el_ms"])
        for record in successful
        if isinstance(record.get("e2el_ms"), (int, float))
        and not isinstance(record.get("e2el_ms"), bool)
    ]
    if len(e2el_values) < 10:
        return None
    return round(percentile(e2el_values, 0.99), 3)
