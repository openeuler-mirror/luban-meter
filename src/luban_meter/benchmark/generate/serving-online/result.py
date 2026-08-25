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


def compute_goodput(
    *,
    successful: list[Mapping[str, Any]],
    duration_seconds: float,
    slo_config: Mapping[str, float],
) -> dict[str, Any]:
    """Compute Goodput: throughput of successful requests that satisfy SLO.

    SLO is an AND-relation across all applicable thresholds. A request is
    counted as SLO-satisfied only when every applicable dimension is within
    its threshold.

    Applicability rules:
    - ttft_ms and e2el_ms apply to all requests.
    - tpot_ms only applies when output_tokens > 1 (TPOT is undefined when
      there is no decode phase). If all requests in the case have
      output_tokens == 1, tpot_ms is marked as not_applicable in the result.
    - When no dimension is applicable (e.g. only tpot_ms configured and all
      requests are single-token, or no successful requests), the function
      returns a not_applicable status.
    """
    ttft_threshold = slo_config.get("ttft_ms")
    tpot_threshold = slo_config.get("tpot_ms")
    e2el_threshold = slo_config.get("e2el_ms")

    if not successful:
        return {
            "status": "not_applicable",
            "reason": "no successful requests to evaluate",
            "slo_config": {
                "ttft_ms": ttft_threshold,
                "tpot_ms": tpot_threshold,
                "e2el_ms": e2el_threshold,
            },
            "applicable_dimensions": [],
            "not_applicable_dimensions": [],
        }

    all_single_token = all(
        token_count(record, "output_tokens") == 1 for record in successful
    )
    tpot_applicable = tpot_threshold is not None and not all_single_token
    any_dimension_applicable = (
        ttft_threshold is not None
        or e2el_threshold is not None
        or tpot_applicable
    )
    if not any_dimension_applicable:
        return {
            "status": "not_applicable",
            "reason": "tpot_ms is the only configured dimension and all "
            "requests have output_tokens == 1 (TPOT undefined)",
            "slo_config": {
                "ttft_ms": ttft_threshold,
                "tpot_ms": tpot_threshold,
                "e2el_ms": e2el_threshold,
            },
            "applicable_dimensions": [],
            "not_applicable_dimensions": (
                ["tpot_ms"] if tpot_threshold is not None else []
            ),
        }

    slo_satisfied: list[Mapping[str, Any]] = []
    slo_violated: list[Mapping[str, Any]] = []
    for record in successful:
        ttft_ms = numeric(record, "ttft_ms")
        e2el_ms = numeric(record, "e2el_ms")
        output_tokens = token_count(record, "output_tokens")

        violated = False
        if ttft_threshold is not None and ttft_ms > ttft_threshold:
            violated = True
        if not violated and tpot_applicable and output_tokens > 1:
            decode_duration_ms = e2el_ms - ttft_ms
            if decode_duration_ms > 0:
                tpot_ms = decode_duration_ms / (output_tokens - 1)
                if tpot_ms > tpot_threshold:
                    violated = True
        if e2el_threshold is not None and e2el_ms > e2el_threshold:
            violated = True

        if violated:
            slo_violated.append(record)
        else:
            slo_satisfied.append(record)

    satisfied_count = len(slo_satisfied)
    violated_count = len(slo_violated)
    total_satisfied_output_tokens = sum(
        token_count(record, "output_tokens") for record in slo_satisfied
    )
    satisfied_rate = (
        satisfied_count / (satisfied_count + violated_count)
        if (satisfied_count + violated_count) > 0
        else 0.0
    )

    applicable_dimensions = []
    if ttft_threshold is not None:
        applicable_dimensions.append("ttft_ms")
    if tpot_applicable:
        applicable_dimensions.append("tpot_ms")
    if e2el_threshold is not None:
        applicable_dimensions.append("e2el_ms")

    not_applicable_dimensions = []
    if tpot_threshold is not None and not tpot_applicable:
        not_applicable_dimensions.append("tpot_ms")

    goodput: dict[str, Any] = {
        "status": "applicable",
        "slo_config": {
            "ttft_ms": ttft_threshold,
            "tpot_ms": tpot_threshold,
            "e2el_ms": e2el_threshold,
        },
        "applicable_dimensions": applicable_dimensions,
        "not_applicable_dimensions": not_applicable_dimensions,
        "slo_satisfied_count": scalar(satisfied_count, "request"),
        "slo_violated_count": scalar(violated_count, "request"),
        "slo_satisfied_rate": scalar(satisfied_rate, "ratio", precision=4),
        "goodput_request_throughput": scalar(
            satisfied_count / duration_seconds, "req/s"
        ),
        "goodput_output_token_throughput": scalar(
            total_satisfied_output_tokens / duration_seconds, "token/s"
        ),
    }
    return goodput


def process_case(
    raw_case: Mapping[str, Any],
    slo_config: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], int, int]:
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
    if slo_config is not None:
        goodput_slo = {
            k: v
            for k, v in slo_config.items()
            if k in ("ttft_ms", "tpot_ms", "e2el_ms")
        }
        if goodput_slo:
            service_view["goodput"] = compute_goodput(
                successful=successful,
                duration_seconds=duration_seconds,
                slo_config=goodput_slo,
            )
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
