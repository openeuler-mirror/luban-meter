"""Windowed stability statistics over persisted streaming facts.

Every value is derived from saved formal request records; nothing is
re-collected and missing observations stay missing instead of reading
as zero.  Latency, outcome and token facts are attributed to the window
holding each request's completion offset, while ``started_requests``
uses the start offset so the applied load stays visible next to the
completed work.  Concurrency integrates each request's overlap with the
window.  Output token throughput is therefore completion-attributed:
it is not a per-token realtime generation rate.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.core.parameters import Parameter
from luban_meter.log import get_logger

from .statistics import scalar, summarize

logger = get_logger(__name__)
DEFAULT_WINDOW_SECONDS = 60
LOW_SAMPLE_COUNT = 10


def parse_window_seconds(value):
    """Accept explicit auto or positive integer seconds."""
    if value == "auto":
        return value
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ValueError("trend_window_seconds must be a positive int or auto")


class TrendWindowParameter(Parameter):
    """Normalize CLI text and JSON integers without changing defaults."""

    def validate(self, name, value):
        return parse_window_seconds(value)

# Transport timeouts the streaming client can raise.  The names come
# from httpx plus the builtin TimeoutError used elsewhere in the
# project; unknown error types stay ordinary failures instead of being
# guessed as timeouts.
TIMEOUT_ERROR_TYPES = frozenset({
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "TimeoutError",
    "TimeoutException",
})

THROUGHPUT_CONDITION = (
    "completion-attributed: successful output tokens of requests "
    "completing in the window over the window span; not a per-token "
    "realtime generation rate"
)


def _finite(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _non_negative(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number >= 0 else None


def _token_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def is_timeout_failure(record: Mapping[str, Any]) -> bool:
    """Classify a failed request by its saved error type only."""
    error = record.get("error")
    if not isinstance(error, Mapping):
        return False
    return error.get("type") in TIMEOUT_ERROR_TYPES


def _completion_times(record, limit_ms):
    start = _non_negative(record.get("start_offset_ms"))
    end = _non_negative(record.get("end_offset_ms"))
    reason = None
    if end is None:
        reason = "missing_or_invalid_end"
    elif end > limit_ms:
        reason = "end_after_formal_duration"
    elif start is not None and end < start:
        reason = "end_before_start"
    return start, end, reason


def select_auto_window(requests, duration):
    """Choose a reproducible sample-supported window from formal facts."""
    if duration is None or duration <= 0:
        return DEFAULT_WINDOW_SECONDS, {
            "reason": "invalid_formal_duration", "fallback": True,
        }
    candidates = [1, 2, 5, 10, 15, 30, 60, 120]
    while candidates[-1] < duration:
        candidates.append(candidates[-1] * 2)
    samples = {"completed": [], "ttft": [], "e2el": []}
    for record in requests:
        _, end, reason = _completion_times(record, duration * 1000)
        if reason:
            continue
        seconds = end / 1000
        samples["completed"].append(seconds)
        if record.get("status") == "success":
            for name in ("ttft", "e2el"):
                if _non_negative(record.get(name + "_ms")) is not None:
                    samples[name].append(seconds)
    active = {
        name: sorted(values) for name, values in samples.items()
        if values or name == "completed"
    }
    evaluated = []
    eligible = []
    for window in candidates:
        complete = int(duration // window)
        if not 2 <= complete <= 10:
            continue
        counts = {}
        for name, values in active.items():
            counts[name] = []
            for index in range(complete):
                end = (index + 1) * window
                upper = (
                    bisect_right(values, end) if end == duration
                    else bisect_left(values, end)
                )
                counts[name].append(
                    upper - bisect_left(values, index * window)
                )
        minimum = min(min(values) for values in counts.values())
        accepted = minimum >= LOW_SAMPLE_COUNT
        tail = duration - complete * window
        evaluated.append({
            "seconds": window, "counts": counts, "accepted": accepted,
        })
        if accepted:
            rank = (
                0 if complete >= 5 else 1, abs(complete - 7),
                int(0 < tail < window * 0.1), -minimum, window,
            )
            eligible.append((rank, window))
    chosen = (
        min(eligible)[1] if eligible else
        next(window for window in candidates if window >= duration)
    )
    return chosen, {
        "algorithm": "completion-samples-v1",
        "reason": (
            "sample_supported_windows" if eligible
            else "insufficient_samples_or_distribution"
        ),
        "fallback": not bool(eligible),
        "minimum_samples": LOW_SAMPLE_COUNT,
        "valid_samples": {
            name: len(values) for name, values in samples.items()
        },
        "candidates": evaluated,
    }


def _latency_summary(
    samples: Sequence[float], keys: Sequence[str]
) -> dict[str, Any]:
    summary = summarize(samples, "ms")
    result: dict[str, Any] = {"count": summary["count"]}
    for key in keys:
        result[key] = summary[key]
    if 0 < len(samples) < LOW_SAMPLE_COUNT:
        result["low_sample"] = True
    return result


def _rate(value: float | None, reason: str | None) -> dict[str, Any]:
    node = {"value": value, "unit": "ratio"}
    if reason is not None:
        node["reason"] = reason
    return node


def time_window_rows(
    requests: Sequence[Mapping[str, Any]],
    duration_seconds: float,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Bucket formal request records into fixed analysis windows.

    Windows are left-closed right-open ``[k*W, (k+1)*W)`` seconds of the
    formal collection window; the final window ends at the case end and
    is marked incomplete when shorter than ``W``.  Returns the window
    rows and the indexes of records whose offsets were unusable.
    """
    rows, unattributed, _ = _window_statistics(
        requests, duration_seconds, window_seconds
    )
    return rows, unattributed


def _window_statistics(requests, duration_seconds, window_seconds):
    if (
        not isinstance(window_seconds, int)
        or isinstance(window_seconds, bool)
        or window_seconds < 1
    ):
        raise ValueError("window_seconds must be a positive integer")
    duration = _finite(duration_seconds)
    if duration is None or duration <= 0:
        raise ValueError("duration_seconds must be a positive number")

    count = max(1, math.ceil(duration / window_seconds))
    starts = [index * window_seconds for index in range(count)]
    ends = [
        min(start + window_seconds, duration)
        for start in starts
    ]
    started = [0] * count
    completed = [0] * count
    successful = [0] * count
    timeout_failed = [0] * count
    other_failed = [0] * count
    token_sums = [0] * count
    missing_tokens = [0] * count
    ttft_samples: list[list[float]] = [[] for _ in range(count)]
    e2el_samples: list[list[float]] = [[] for _ in range(count)]
    events: list[tuple[float, int]] = []
    unattributed: list[int] = []
    reasons: dict[str, int] = {}
    missing_start = 0
    limit_ms = duration * 1000

    def bucket(offset_ms: float) -> int:
        seconds = offset_ms / 1000.0
        return min(int(seconds // window_seconds), count - 1)

    for position, record in enumerate(requests):
        start, end, reason = _completion_times(record, limit_ms)
        if start is not None and start <= limit_ms:
            started[bucket(start)] += 1
        if reason:
            unattributed.append(position)
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        index = bucket(end)
        completed[index] += 1
        if start is None:
            missing_start += 1
        elif start < end:
            events.append((start / 1000.0, 1))
            events.append((end / 1000.0, -1))
        if record.get("status") == "success":
            successful[index] += 1
            tokens = _token_count(record.get("output_tokens"))
            if tokens is None:
                missing_tokens[index] += 1
            else:
                token_sums[index] += tokens
            ttft = _non_negative(record.get("ttft_ms"))
            if ttft is not None:
                ttft_samples[index].append(ttft)
            e2el = _non_negative(record.get("e2el_ms"))
            if e2el is not None:
                e2el_samples[index].append(e2el)
        elif is_timeout_failure(record):
            timeout_failed[index] += 1
        else:
            other_failed[index] += 1

    events.sort()
    overlap = [0.0] * count
    peaks = [0] * count
    active = 0
    peak = 0
    cursor = 0.0
    event_index = 0
    for index in range(count):
        peak = 0
        cursor = float(starts[index])
        while (
            event_index < len(events)
            and events[event_index][0] < ends[index]
        ):
            moment, delta = events[event_index]
            clamped = max(moment, cursor)
            if clamped > cursor:
                peak = max(peak, active)
                overlap[index] += active * (clamped - cursor)
            cursor = clamped
            active += delta
            event_index += 1
        if ends[index] > cursor:
            peak = max(peak, active)
            overlap[index] += active * (ends[index] - cursor)
        peaks[index] = peak

    rows = []
    for index in range(count):
        span = ends[index] - starts[index]
        complete = span >= window_seconds
        completed_count = completed[index]
        failed_count = timeout_failed[index] + other_failed[index]
        rate_reason = (
            None if completed_count
            else "no completed requests in window"
        )
        throughput: dict[str, Any] = {
            "value": None,
            "unit": "token/s",
            "condition": THROUGHPUT_CONDITION,
        }
        if span <= 0:
            span_reason = "invalid window span"
            throughput["reason"] = span_reason
            rate_reason = rate_reason or span_reason
        elif unattributed and len(unattributed) == len(requests):
            throughput["reason"] = "no requests have attributable completions"
        elif missing_tokens[index]:
            throughput["reason"] = (
                f"{missing_tokens[index]} of {successful[index]} "
                "successful requests lack output token counts"
            )
        else:
            throughput["value"] = round(token_sums[index] / span, 3)
        rows.append({
            "index": index,
            "start_s": round(starts[index], 3),
            "end_s": round(ends[index], 3),
            "span_s": round(span, 3),
            "complete": complete,
            "started_requests": scalar(started[index], "request", 0),
            "completed_requests": scalar(completed_count, "request", 0),
            "successful": scalar(successful[index], "request", 0),
            "failed": scalar(failed_count, "request", 0),
            "timeout_failed": scalar(timeout_failed[index], "request", 0),
            "other_failed": scalar(other_failed[index], "request", 0),
            "success_rate": _rate(
                round(successful[index] / completed_count, 4)
                if completed_count else None,
                rate_reason,
            ),
            "timeout_rate": _rate(
                round(timeout_failed[index] / completed_count, 4)
                if completed_count else None,
                rate_reason,
            ),
            "output_token_throughput": throughput,
            "ttft_ms": _latency_summary(ttft_samples[index], ("mean", "p99")),
            "e2el_ms": _latency_summary(e2el_samples[index], ("p99",)),
            "average_concurrency": scalar(
                round(overlap[index] / span, 3) if span > 0 else None,
                "request",
            ),
            "peak_concurrency": scalar(peaks[index], "request", 0),
        })
    attributed = len(requests) - len(unattributed)
    return rows, unattributed, {
        "total_requests": len(requests),
        "completion_eligible": attributed,
        "completion_excluded": len(unattributed),
        "exclusion_reasons": reasons,
        "concurrency_eligible": attributed - missing_start,
        "completion_without_valid_start": missing_start,
    }


def time_window_result(
    requests: Sequence[Mapping[str, Any]],
    duration_seconds: float,
    window_seconds: int | str = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Build the persisted stability_trends metric payload."""
    duration = _finite(duration_seconds)
    requested = parse_window_seconds(window_seconds)
    mode = "auto" if requested == "auto" else "fixed"
    selection = {"reason": "configured_or_default_fixed", "fallback": False}
    if mode == "auto":
        window_seconds, selection = select_auto_window(requests, duration)
    else:
        window_seconds = requested
    window_metadata = {
        "window_mode": mode,
        "requested_window_seconds": requested,
        "window_selection": selection,
    }
    logger.info(
        "Stability window selected: mode=%s seconds=%s reason=%s",
        mode, window_seconds, selection["reason"],
    )
    if duration is None or duration <= 0:
        return {
            "status": "unavailable",
            "reason": "missing or invalid formal collection duration",
            "values": {"window_seconds": window_seconds, "windows": []},
            "metadata": {
                **window_metadata,
                "window_seconds": window_seconds,
                "coverage": {
                    "total_requests": len(requests),
                    "completion_eligible": 0,
                    "completion_excluded": len(requests),
                    "exclusion_reasons": {
                        "invalid_formal_duration": len(requests)
                    },
                    "concurrency_eligible": 0,
                    "completion_without_valid_start": 0,
                },
                "unattributed_requests": {
                    "count": len(requests),
                    "reason": "invalid formal duration",
                    "request_indexes": list(range(min(len(requests), 10))),
                },
            },
        }
    rows, unattributed, coverage = _window_statistics(
        requests, duration, window_seconds
    )
    metadata = {
        **window_metadata,
        "window_seconds": window_seconds,
        "window_rule": (
            "left-closed right-open [k*W, (k+1)*W) seconds of the formal "
            "collection window; the final window ends at the case end "
            "and includes completions exactly at the formal end; "
            "complete=false when shorter than W"
        ),
        "attribution": {
            "completed_requests": "end_offset_ms",
            "started_requests": "start_offset_ms",
            "latency_and_outcomes": "end_offset_ms",
            "output_tokens": "end_offset_ms of successful requests",
            "concurrency": (
                "overlap of [start_offset_ms, end_offset_ms) with window"
            ),
        },
        "throughput_definition": (
            "successful output tokens completing in the window divided "
            "by the window span; completion-attributed, not a per-token "
            "realtime generation rate"
        ),
        "rate_denominator": "requests completing in the window",
        "timeout_error_types": sorted(TIMEOUT_ERROR_TYPES),
        "warmup_included": False,
        "low_sample_threshold": LOW_SAMPLE_COUNT,
        "formal_duration_s": round(duration, 3),
        "coverage": coverage,
        "unattributed_requests": {
            "count": len(unattributed),
            "reason": "invalid, reversed or out-of-window completion",
            "request_indexes": unattributed[:10],
        },
    }
    if not rows:
        return {
            "status": "unavailable",
            "reason": "no analysis windows",
            "values": {"window_seconds": window_seconds, "windows": []},
            "metadata": metadata,
        }
    return {
        "status": "success",
        "observation_boundary": "client",
        "values": {"window_seconds": window_seconds, "windows": rows},
        "metadata": metadata,
    }
