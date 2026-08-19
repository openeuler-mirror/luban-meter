"""Calculate online metrics for exact-length, fixed-rate serving cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.benchmark.generate.common.statistics import (
    scalar,
    summarize,
)


def numeric(record: Mapping[str, Any], name: str) -> float:
    value = record.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
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


def process_case(raw_case: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
    input_length = positive_token_count(raw_case, "input_length")
    output_length = positive_token_count(raw_case, "output_length")
    request_rate = positive_numeric(raw_case, "request_rate")
    duration_seconds = positive_numeric(raw_case, "benchmark_duration_seconds")
    maximum_concurrency = positive_token_count(
        raw_case, "maximum_request_concurrency"
    )
    peak_concurrency = token_count(raw_case, "peak_concurrent_requests")
    records = object_list(raw_case.get("requests"), "case requests")
    if not records:
        raise ValueError("case requests must not be empty")

    successful = [record for record in records if record.get("status") == "success"]
    failed = [record for record in records if record.get("status") != "success"]
    ttft_samples: list[float] = []
    itl_samples: list[float] = []
    tpot_samples: list[float] = []
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
        input_tokens = token_count(record, "input_tokens")
        output_tokens = token_count(record, "output_tokens")
        if input_tokens != input_length or output_tokens != output_length:
            raise ValueError("successful request token counts must match the case")
        if e2el_ms < ttft_ms:
            raise ValueError("request e2el_ms must not be smaller than ttft_ms")

        raw_itls = record.get("itl_samples_ms")
        if (
            not isinstance(raw_itls, Sequence)
            or isinstance(raw_itls, (str, bytes))
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
            raise ValueError("request ITL samples must be non-negative numbers")

        ttft_samples.append(ttft_ms)
        itl_samples.extend(request_itls)
        e2el_samples.append(e2el_ms)
        input_token_samples.append(float(input_tokens))
        output_token_samples.append(float(output_tokens))
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        if e2el_ms > 0:
            output_throughput_samples.append(1000 * output_tokens / e2el_ms)
        decode_duration_ms = e2el_ms - ttft_ms
        if output_tokens > 1 and decode_duration_ms > 0:
            tpot_ms = decode_duration_ms / (output_tokens - 1)
            tpot_samples.append(tpot_ms)
            decode_throughput_samples.append(1000 / tpot_ms)

    request_durations = [numeric(record, "duration_ms") for record in records]
    dispatch_delays = [numeric(record, "dispatch_delay_ms") for record in records]
    start_offsets = [numeric(record, "start_offset_ms") for record in records]
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
        "itl": summarize(itl_samples, "ms"),
        "tpot": summarize(tpot_samples, "ms/token"),
        "e2el": summarize(e2el_samples, "ms"),
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
        "request_throughput": scalar(successful_count / duration_seconds, "req/s"),
        "input_token_throughput": scalar(
            total_input_tokens / duration_seconds, "token/s"
        ),
        "output_token_throughput": scalar(
            total_output_tokens / duration_seconds, "token/s"
        ),
        "total_token_throughput": scalar(
            (total_input_tokens + total_output_tokens) / duration_seconds,
            "token/s",
        ),
    }
    if failed_count == 0:
        request_outcome = "all_success"
    elif successful_count == 0:
        request_outcome = "all_failed"
    else:
        request_outcome = "partial_failed"
    return (
        {
            "input_length": input_length,
            "output_length": output_length,
            "request_rate": request_rate,
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

    cases: list[dict[str, Any]] = []
    total_successful = 0
    total_failed = 0
    for raw_case in raw_cases:
        case, successful, failed = process_case(raw_case)
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
