"""Online collection and scoring."""

from __future__ import annotations

import itertools
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from luban_meter.benchmarking.generation_performance.common import (
    http_client,
    phase_timing,
    reasoning_tokens,
)
from luban_meter.benchmarking.generation_performance.common.statistics import (
    scalar,
    summarize,
    summarize_weighted,
)
from luban_meter.core.token_totals import token_throughput_value
from luban_meter.log import get_logger
from luban_meter.runtime.http_client import iter_response_lines

logger = get_logger(__name__)


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
    content_times = observation.content_event_times
    tool_call_times = observation.tool_call_event_times
    # Completion streams have no separate reasoning or tool channel.
    if (
        not content_times and not observation.reasoning_event_times
        and not tool_call_times
    ):
        content_times = event_times
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
        "output_observation": (
            "reasoning_content_and_tool_calls"
            if tool_call_times else "reasoning_and_content"
        ),
        "generated_text": observation.generated_text,
        "reasoning_text": observation.reasoning_text,
        "has_content": bool(observation.generated_text),
        "first_content_latency_ms": (
            round((content_times[0] - started) * 1000, 3)
            if content_times else None
        ),
        "output_event_offsets_ms": [
            round((value - started) * 1000, 3) for value in event_times
        ],
        "reasoning_event_offsets_ms": [
            round((value - started) * 1000, 3)
            for value in observation.reasoning_event_times
        ],
        "content_event_offsets_ms": [
            round((value - started) * 1000, 3) for value in content_times
        ],
        "tool_call_event_offsets_ms": [
            round((value - started) * 1000, 3) for value in tool_call_times
        ],
        "reasoning_tokens": observation.reasoning_tokens,
        "finish_reason": observation.finish_reason,
        "usage": observation.usage,
        # Raw per-request server queue wait; kept unrounded so the
        # aggregated mean stays recomputable from persisted facts.
        "server_queue_time_ms": observation.server_queue_time_ms,
    }


def execute_request(
    *,
    client: httpx.Client,
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
    thinking_metrics: bool = False,
    reasoning_token_counter: Any = None,
    exception_log_info: Callable[[Exception], tuple] | None = None,
) -> dict[str, Any]:
    """Execute a streaming HTTP request and return a result record.

    ``endpoint`` is ``"/v1/completions"`` or ``"/v1/chat/completions"``.
    ``stream_collector`` is ``collect_completion_stream`` or
    ``collect_chat_stream``.
    """
    def failure_details(error):
        if exception_log_info is not None:
            return exception_log_info(error)
        return type(error), error, error.__traceback__

    url = f"{service_url.rstrip('/')}/{endpoint.lstrip('/')}"
    http_request = client.build_request(
        "POST",
        url,
        content=json.dumps(payload).encode("utf-8"),
        headers=http_client.request_headers(api_key),
        timeout=timeout,
    )
    response = None
    observation = None
    error = None
    failure_info = None
    drain_error = close_error = None
    started = time.perf_counter()
    tracker.enter()
    try:
        response = client.send(http_request, stream=True)
        response.raise_for_status()
        lines = iter_response_lines(response)
        observation = stream_collector(lines)
        ended = time.perf_counter()
        try:
            # Preserve the collector's end time while finishing HTTP framing.
            for _ in lines:
                pass
        except Exception as exc:  # noqa: BLE001
            drain_error = exc
    except httpx.HTTPStatusError as exc:
        ended = time.perf_counter()
        failure_info = failure_details(exc)
        response = exc.response
        try:
            detail = response.read().decode("utf-8", errors="replace")
        except httpx.HTTPError as read_error:
            logger.warning("HTTP error body unavailable",
                           exc_info=failure_details(read_error))
            detail = str(read_error)
        error = RuntimeError(f"HTTP {response.status_code}: {detail}")
    except Exception as exc:  # noqa: BLE001
        ended = time.perf_counter()
        error = exc
        failure_info = failure_details(exc)
    finally:
        try:
            if response is not None:
                response.close()
        except Exception as exc:  # noqa: BLE001
            close_error = exc
        finally:
            tracker.exit()
    record = _build_request_record(
        request_index=request_index,
        started=started,
        ended=ended,
        benchmark_start=benchmark_start,
        scheduled_time=scheduled_time,
        observation=observation,
        error=error,
    )
    if endpoint == "/v1/chat/completions" and thinking_metrics:
        reasoning_tokens.supplement_reasoning_tokens(
            record, observation, reasoning_token_counter,
        )
        record["phase_timing"] = phase_timing.request_phase_timing(
            record, thinking_metrics=thinking_metrics,
        )
    if response is not None:
        record["http_version"] = response.http_version
    if failure_info is not None:
        logger.error("Streaming request failed: index=%s", request_index,
                     exc_info=failure_info)
    for cleanup_error in (drain_error, close_error):
        if cleanup_error is None:
            continue
        logger.warning(
            "HTTP connection cleanup failed: index=%s", request_index,
            exc_info=failure_details(cleanup_error),
        )
        record.setdefault("connection_cleanup_error", {
            "type": type(cleanup_error).__name__,
            "message": str(cleanup_error),
        })
    return record


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
    server_queue_times: list[float] = []
    for record in successful:
        ttft_ms = numeric(record, "ttft_ms")
        e2el_ms = numeric(record, "e2el_ms")
        input_tokens = record.get("input_tokens")
        output_tokens = record.get("output_tokens")
        if e2el_ms < ttft_ms:
            raise ValueError(
                "request e2el_ms must not be smaller than ttft_ms"
            )

        reported_queue_ms = record.get("server_queue_time_ms")
        if (
            isinstance(reported_queue_ms, (int, float))
            and not isinstance(reported_queue_ms, bool)
            and math.isfinite(reported_queue_ms)
            and reported_queue_ms >= 0
        ):
            server_queue_times.append(float(reported_queue_ms))

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
    # Only requests that actually waited count towards the customer's
    # "average wait among queued requests"; reported zero waits are real
    # observations, not missing data.
    waited_queue_times = [
        value for value in server_queue_times if value > 0
    ]
    queue_time_reported = len(server_queue_times)
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
        # Microsecond-scale waits need more stored decimals than the
        # default three; display layers still format for reading.
        "queue_time_ms": summarize(
            waited_queue_times, "ms", precision=6
        ),
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
            token_throughput_value(
                total_input_tokens + total_output_tokens, duration_seconds
            ),
            "token/s",
        ),
        "queue_time_waited_mean": (
            scalar(
                sum(waited_queue_times) / len(waited_queue_times),
                "ms",
                precision=6,
            )
            if waited_queue_times
            else scalar(None, "ms")
        ),
        "queue_time_reported_requests": scalar(
            queue_time_reported, "request"
        ),
        "queue_time_missing_requests": scalar(
            successful_count - queue_time_reported, "request"
        ),
    }
    # Missing per-request metrics are not zero waits. Distinguish:
    # nothing reported at all, a complete all-zero observation, zero
    # waits among a partial sample, and subset results whose coverage
    # must stay visible instead of implying the whole Case.
    queue_time_reason = None
    if successful_count and not queue_time_reported:
        queue_time_reason = (
            "service did not report per-request queue time metrics"
        )
    elif queue_time_reported and not waited_queue_times:
        queue_time_missing = successful_count - queue_time_reported
        if queue_time_missing:
            queue_time_reason = (
                "no request with reported timing waited; "
                f"{queue_time_missing} of {successful_count} requests "
                "did not report timing"
            )
        else:
            queue_time_reason = "no request waited in the server queue"
    if queue_time_reason is not None:
        request_view["queue_time_ms"]["reason"] = queue_time_reason
        service_view["queue_time_waited_mean"].update(
            value=None, reason=queue_time_reason
        )
    elif 0 < queue_time_reported < successful_count:
        scope = (
            f"partial timing coverage: {queue_time_reported} of "
            f"{successful_count} successful requests reported queue time"
        )
        request_view["queue_time_ms"]["condition"] = scope
        service_view["queue_time_waited_mean"]["condition"] = scope
    # Missing usage preserves success but cannot imply zero tokens.
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

    tool_requests = sum(
        bool(record.get("tool_call_event_offsets_ms")) for record in successful
    )
    if tool_requests:
        tool_events = sum(
            len(record.get("tool_call_event_offsets_ms", ()))
            for record in successful
        )
        logger.info(
            "Tool-call output observed: requests=%d events=%d",
            tool_requests, tool_events,
        )

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
