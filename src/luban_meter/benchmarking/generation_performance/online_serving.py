"""Online collection and scoring."""

from __future__ import annotations

import itertools
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from luban_meter.benchmarking.generation_performance.common import (
    http_client,
)
from luban_meter.benchmarking.generation_performance.common.statistics import (
    scalar,
    summarize,
    summarize_weighted,
)


class ActiveRequestTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._peak = 0

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)

    def exit(self) -> None:
        with self._lock:
            self._active -= 1


def _build_request_record(
    *,
    request_index: int,
    started: float,
    ended: float,
    benchmark_start: float,
    scheduled_time: float,
    observation: Any | None,
    error: Exception | None,
) -> dict[str, Any]:
    """Build a success or failure request record."""
    if error is not None:
        return {
            "request_index": request_index,
            "submitted_at_monotonic_s": started,
            "status": "failed",
            "scheduled_offset_ms": round(
                (scheduled_time - benchmark_start) * 1000, 3
            ),
            "start_offset_ms": round((started - benchmark_start) * 1000, 3),
            "dispatch_delay_ms": round((started - scheduled_time) * 1000, 3),
            "end_offset_ms": round((ended - benchmark_start) * 1000, 3),
            "scheduled_latency_ms": round((ended - scheduled_time) * 1000, 3),
            "duration_ms": round((ended - started) * 1000, 3),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    event_times = observation.event_times
    ttft_ms = (event_times[0] - started) * 1000
    itl_samples_ms = [
        (current - previous) * 1000
        for previous, current in itertools.pairwise(event_times)
    ]
    return {
        "request_index": request_index,
        "submitted_at_monotonic_s": started,
        "status": "success",
        "scheduled_offset_ms": round(
            (scheduled_time - benchmark_start) * 1000, 3
        ),
        "start_offset_ms": round((started - benchmark_start) * 1000, 3),
        "dispatch_delay_ms": round((started - scheduled_time) * 1000, 3),
        "end_offset_ms": round((ended - benchmark_start) * 1000, 3),
        "scheduled_latency_ms": round((ended - scheduled_time) * 1000, 3),
        "duration_ms": round((ended - started) * 1000, 3),
        "ttft_ms": round(ttft_ms, 3),
        "last_output_latency_ms": round((event_times[-1] - started) * 1000, 3),
        "e2el_ms": round((ended - started) * 1000, 3),
        "itl_samples_ms": [round(value, 3) for value in itl_samples_ms],
        "input_tokens": observation.input_tokens,
        "output_tokens": observation.output_tokens,
        "stream_event_count": len(event_times),
    }


def execute_request(
    *,
    request_index: int,
    payload: dict[str, Any],
    endpoint: str,
    service_url: str,
    api_key: str,
    timeout: float,
    benchmark_start: float,
    scheduled_time: float,
    tracker: ActiveRequestTracker,
    stream_collector: Callable[[Any], Any],
) -> dict[str, Any]:
    """Execute a streaming HTTP request and return a result record.

    ``endpoint`` is ``"/v1/completions"`` or ``"/v1/chat/completions"``.
    ``stream_collector`` is ``collect_completion_stream`` or
    ``collect_chat_stream``.
    """
    url = f"{service_url.rstrip('/')}/{endpoint.lstrip('/')}"
    http_request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=http_client.request_headers(api_key),
        method="POST",
    )
    started = time.perf_counter()
    tracker.enter()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            observation = stream_collector(response)
            ended = time.perf_counter()

        return _build_request_record(
            request_index=request_index,
            started=started,
            ended=ended,
            benchmark_start=benchmark_start,
            scheduled_time=scheduled_time,
            observation=observation,
            error=None,
        )
    except urllib.error.HTTPError as exc:
        ended = time.perf_counter()
        detail = exc.read().decode("utf-8", errors="replace")
        error: Exception = RuntimeError(f"HTTP {exc.code}: {detail}")
        return _build_request_record(
            request_index=request_index,
            started=started,
            ended=ended,
            benchmark_start=benchmark_start,
            scheduled_time=scheduled_time,
            observation=None,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        ended = time.perf_counter()
        return _build_request_record(
            request_index=request_index,
            started=started,
            ended=ended,
            benchmark_start=benchmark_start,
            scheduled_time=scheduled_time,
            observation=None,
            error=exc,
        )
    finally:
        tracker.exit()


def numeric(record: Mapping[str, Any], name: str) -> float:
    value = record.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def positive_numeric(record: Mapping[str, Any], name: str) -> float:
    value = numeric(record, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def token_count(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def positive_token_count(record: Mapping[str, Any], name: str) -> int:
    value = token_count(record, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def object_list(value: Any, name: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise TypeError(f"{name} must be a list of objects")
    return list(value)


def guidellm_token_latencies(
    record: Mapping[str, Any],
) -> tuple[float, float | None]:
    """Return TPOT including first-token wait and ITL excluding it.

    Follow GuideLLM 6eab76f request_stats.py. The final content event
    bounds token time; E2EL includes stream teardown and usage events.
    """
    output_tokens = positive_token_count(record, "output_tokens")
    ttft_ms = numeric(record, "ttft_ms")
    e2el_ms = numeric(record, "e2el_ms")
    if e2el_ms < ttft_ms:
        raise ValueError("request e2el_ms must not be smaller than ttft_ms")
    if record.get("last_output_latency_ms") is None:
        return e2el_ms / output_tokens, None
    last_ms = numeric(record, "last_output_latency_ms")
    if not ttft_ms <= last_ms <= e2el_ms:
        raise ValueError(
            "last_output_latency_ms must be between ttft_ms and e2el_ms"
        )
    itl_ms = (
        (last_ms - ttft_ms) / (output_tokens - 1)
        if output_tokens > 1
        else None
    )
    return last_ms / output_tokens, itl_ms


def compute_goodput(
    *,
    successful: list[Mapping[str, Any]],
    failed_count: int,
    duration_seconds: float,
    slo_config: Mapping[str, float],
) -> dict[str, Any]:
    """Apply GuideLLM objectives; failures lower SLO attainment.

    ``tpot_ms`` is the decode-only ITL objective, not the report TPOT.
    Any missing configured measurement makes a success undetermined,
    even when another configured objective is already violated.
    The rate denominator remains this benchmark's formal case window.
    """
    thresholds = {
        name: positive_numeric(slo_config, name)
        for name in ("ttft_ms", "tpot_ms", "e2el_ms")
        if name in slo_config
    }
    if not thresholds:
        raise ValueError("SLO requires at least one objective")
    satisfied = violated = undetermined = output_tokens = 0
    missing_tokens = False
    measurable: set[str] = set()
    for record in successful:
        measured = {
            name: numeric(record, name)
            if record.get(name) is not None
            else None
            for name in ("ttft_ms", "e2el_ms")
        }
        measured["tpot_ms"] = None
        if "tpot_ms" in thresholds and all(
            record.get(name) is not None
            for name in (
                "ttft_ms",
                "e2el_ms",
                "output_tokens",
                "last_output_latency_ms",
            )
        ):
            _, measured["tpot_ms"] = guidellm_token_latencies(record)
        measurable.update(
            name for name in thresholds if measured[name] is not None
        )
        if any(measured[name] is None for name in thresholds):
            undetermined += 1
        elif any(measured[name] > limit for name, limit in thresholds.items()):
            violated += 1
        else:
            satisfied += 1
            if record.get("output_tokens") is None:
                missing_tokens = True
            else:
                output_tokens += token_count(record, "output_tokens")

    determined = satisfied + violated + failed_count
    result = {
        "status": "applicable" if determined else "not_applicable",
        "slo_config": {
            name: thresholds.get(name)
            for name in ("ttft_ms", "tpot_ms", "e2el_ms")
        },
        "tpot_metric": "inter_token_latency",
        "applicable_dimensions": [
            name for name in thresholds if name in measurable
        ],
        "not_applicable_dimensions": [
            name for name in thresholds if name not in measurable
        ],
        "slo_satisfied_count": scalar(satisfied, "request"),
        "slo_violated_count": scalar(violated, "request"),
        "failed_count": scalar(failed_count, "request"),
        "undetermined_count": scalar(undetermined, "request"),
        "determined_count": scalar(determined, "request"),
        "slo_satisfied_rate": {
            "value": round(satisfied / determined, 4) if determined else None,
            "unit": "ratio",
        },
        "goodput_request_throughput": {
            "value": round(satisfied / duration_seconds, 3)
            if determined
            else None,
            "unit": "req/s",
        },
        "goodput_output_token_throughput": {
            "value": round(output_tokens / duration_seconds, 3)
            if determined and not missing_tokens
            else None,
            "unit": "token/s",
        },
    }
    if not determined:
        result["reason"] = "no requests with determined SLO outcomes"
    return result


def process_case(
    raw_case: Mapping[str, Any],
    slo_config: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], int, int]:
    request_rate = (
        positive_numeric(raw_case, "request_rate")
        if raw_case.get("request_rate") is not None
        else None
    )
    duration_seconds = positive_numeric(raw_case, "benchmark_duration_seconds")
    maximum_concurrency = token_count(raw_case, "maximum_request_concurrency")
    peak_concurrency = token_count(raw_case, "peak_concurrent_requests")
    records = object_list(raw_case.get("requests"), "case requests")
    if not records:
        raise ValueError("case requests must not be empty")

    successful = [
        record for record in records if record.get("status") == "success"
    ]
    failed = [
        record for record in records if record.get("status") != "success"
    ]
    ttft_samples: list[float] = []
    itl_samples: list[float] = []
    weighted_tpots: list[tuple[float, int]] = []
    weighted_itls: list[tuple[float, int]] = []
    e2el_samples: list[float] = []
    input_token_samples: list[float] = []
    output_token_samples: list[float] = []
    output_throughput_samples: list[float] = []
    decode_throughput_samples: list[float] = []

    total_input_tokens = 0
    total_output_tokens = 0
    for record in successful:
        ttft_ms = numeric(record, "ttft_ms")
        e2el_ms = numeric(record, "e2el_ms")
        input_tokens = record.get("input_tokens")
        output_tokens = record.get("output_tokens")
        if e2el_ms < ttft_ms:
            raise ValueError(
                "request e2el_ms must not be smaller than ttft_ms"
            )

        raw_itls = record.get("itl_samples_ms")
        if not isinstance(raw_itls, Sequence) or isinstance(
            raw_itls, (str, bytes)
        ):
            raise TypeError("request itl_samples_ms must be a list")
        request_itls = [
            float(value)
            for value in raw_itls
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        ]
        if len(request_itls) != len(raw_itls):
            raise ValueError(
                "request ITL samples must be non-negative numbers"
            )

        ttft_samples.append(ttft_ms)
        itl_samples.extend(request_itls)
        e2el_samples.append(e2el_ms)
        if input_tokens is not None:
            input_tokens = token_count(record, "input_tokens")
            input_token_samples.append(float(input_tokens))
            total_input_tokens += input_tokens
        if output_tokens is None:
            continue
        output_tokens = positive_token_count(record, "output_tokens")
        output_token_samples.append(float(output_tokens))
        total_output_tokens += output_tokens
        if e2el_ms > 0:
            output_throughput_samples.append(1000 * output_tokens / e2el_ms)
        tpot_ms, itl_ms = guidellm_token_latencies(record)
        weighted_tpots.append((tpot_ms, output_tokens))
        if itl_ms is not None:
            weighted_itls.append((itl_ms, output_tokens - 1))
            if itl_ms > 0:
                decode_throughput_samples.append(1000 / itl_ms)

    request_durations = [numeric(record, "duration_ms") for record in records]
    dispatch_delays = [
        numeric(record, "dispatch_delay_ms") for record in records
    ]
    start_offsets = [numeric(record, "start_offset_ms") for record in records]
    scheduled_latencies = [
        numeric(record, "scheduled_latency_ms")
        for record in records
        if isinstance(record.get("scheduled_latency_ms"), (int, float))
        and not isinstance(record.get("scheduled_latency_ms"), bool)
    ]
    average_concurrency = sum(request_durations) / (1000 * duration_seconds)
    successful_count = len(successful)
    failed_count = len(failed)
    total_requests = len(records)
    if total_requests > 1:
        start_window_seconds = (max(start_offsets) - min(start_offsets)) / 1000
        achieved_start_rate = (
            (total_requests - 1) / start_window_seconds
            if start_window_seconds > 0
            else 0.0
        )
    else:
        achieved_start_rate = 0.0

    request_view = {
        "ttft": summarize(ttft_samples, "ms"),
        "itl": summarize_weighted(weighted_itls, "ms/token"),
        "tpot": summarize_weighted(weighted_tpots, "ms/token"),
        "stream_event_itl": summarize(itl_samples, "ms"),
        "e2el": summarize(e2el_samples, "ms"),
        "scheduled_latency": summarize(scheduled_latencies, "ms")
        if scheduled_latencies
        else summarize([], "ms"),
        "input_tokens": summarize(input_token_samples, "token"),
        "output_tokens": summarize(output_token_samples, "token"),
        "output_token_throughput": summarize(
            output_throughput_samples, "token/s"
        ),
        "decode_token_throughput": summarize(
            decode_throughput_samples, "token/s"
        ),
        "dispatch_delay": summarize(dispatch_delays, "ms"),
    }
    service_view = {
        "benchmark_duration": scalar(duration_seconds, "s", precision=6),
        "offered_request_rate": scalar(request_rate, "req/s"),
        "achieved_request_start_rate": scalar(achieved_start_rate, "req/s"),
        "total_requests": scalar(total_requests, "request"),
        "successful_requests": scalar(successful_count, "request"),
        "failed_requests": scalar(failed_count, "request"),
        "average_concurrency": scalar(average_concurrency, "request"),
        "maximum_request_concurrency": scalar(maximum_concurrency, "request"),
        "peak_concurrent_requests": scalar(peak_concurrency, "request"),
        "total_input_tokens": scalar(total_input_tokens, "token"),
        "total_output_tokens": scalar(total_output_tokens, "token"),
        "request_throughput": scalar(
            successful_count / duration_seconds, "req/s"
        ),
        "input_token_throughput": scalar(
            total_input_tokens / duration_seconds, "token/s"
        ),
        "output_token_throughput": scalar(
            total_output_tokens / duration_seconds,
            "token/s",
        ),
        "total_token_throughput": scalar(
            (total_input_tokens + total_output_tokens) / duration_seconds,
            "token/s",
        ),
    }
    # Missing usage preserves request success but cannot imply zero tokens.
    missing_input = len(input_token_samples) < successful_count
    missing_output = len(output_token_samples) < successful_count
    for field, missing in (
        ("total_input_tokens", missing_input),
        ("input_token_throughput", missing_input),
        ("total_output_tokens", missing_output),
        ("output_token_throughput", missing_output),
        ("total_token_throughput", missing_input or missing_output),
    ):
        if missing:
            service_view[field].update(value=None, reason="missing usage")
    if slo_config is not None:
        goodput_slo = {
            field_name: field_value
            for field_name, field_value in slo_config.items()
            if field_name in ("ttft_ms", "tpot_ms", "e2el_ms")
        }
        if goodput_slo:
            service_view["goodput"] = compute_goodput(
                successful=successful,
                failed_count=failed_count,
                duration_seconds=duration_seconds,
                slo_config=goodput_slo,
            )
    if failed_count == 0:
        request_outcome = "all_success"
    elif successful_count == 0:
        request_outcome = "all_failed"
    else:
        request_outcome = "partial_failed"

    case_info: dict[str, Any] = {"request_rate": request_rate}
    for name in ("arrival_process", "burstiness", "num_prompts", "max_tokens"):
        if raw_case.get(name) is not None:
            case_info[name] = raw_case[name]

    return (
        {
            **case_info,
            "request_outcome": request_outcome,
            "request_view": request_view,
            "service_view": service_view,
        },
        successful_count,
        failed_count,
    )


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")
    raw_cases = object_list(raw_metrics.get("cases"), "raw cases")
    if not raw_cases:
        raise ValueError("raw cases must not be empty")

    raw_metadata = raw_result.get("metadata")
    slo_config: Mapping[str, float] | None = None
    if isinstance(raw_metadata, Mapping):
        raw_slo = raw_metadata.get("slo_config")
        if isinstance(raw_slo, Mapping):
            slo_config = raw_slo

    cases: list[dict[str, Any]] = []
    total_successful = 0
    total_failed = 0
    for raw_case in raw_cases:
        case, successful, failed = process_case(raw_case, slo_config)
        cases.append(case)
        total_successful += successful
        total_failed += failed

    if total_failed == 0:
        result_status = "success"
        request_outcome = "all_success"
    elif total_successful == 0:
        result_status = "failed"
        request_outcome = "all_failed"
    else:
        result_status = "partial_failed"
        request_outcome = "partial_failed"

    metadata = raw_result.get("metadata")
    result_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    result_metadata.update(
        {
            "request_outcome": request_outcome,
            "total_successful_requests": total_successful,
            "total_failed_requests": total_failed,
            "request_metrics_source": "same_case_client_streaming_timeline",
        }
    )
    return {
        "status": result_status,
        "metrics": {"cases": cases},
        "metadata": result_metadata,
        "error": (
            {
                "type": "RequestFailures",
                "message": f"{total_failed} formal requests failed",
            }
            if total_failed
            else None
        ),
    }
