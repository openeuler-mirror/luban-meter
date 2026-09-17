"""Calculate online serving metrics for random and dataset workload
modes.

In **random** mode, every request in a case has identical
input/output lengths, and token counts are validated against
``input_length`` / ``output_length`` case fields.

In **dataset** mode, input/output lengths vary per request.  Token
counts are summarised as distributions rather than validated against
fixed values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.benchmarking.generation_performance.common.statistics import (
    scalar,
    summarize,
    summarize_weighted,
)
from luban_meter.result.report_spec import line, table

REPORT = {
    "tables": [
        table(
            "在线性能",
            "/cases",
            {
                "/input_length": "输入长度",
                "/output_length": "输出长度",
                "/request_rate": "请求速率 (req/s)",
                "/request_outcome": "状态",
                "/service_view/successful_requests": "成功请求",
                "/service_view/failed_requests": "失败请求",
                "/service_view/request_throughput": "请求吞吐",
                "/service_view/output_token_throughput": "输出吞吐",
                "/request_view/ttft/p50": "TTFT P50",
                "/request_view/ttft/p99": "TTFT P99",
                "/request_view/tpot/p50": "TPOT P50",
                "/request_view/tpot/p99": "TPOT P99",
                "/request_view/itl/p99": "ITL P99",
            },
            charts=[
                line(
                    "/request_rate",
                    metric,
                    ["/input_length", "/output_length"],
                )
                for metric in (
                    "/service_view/request_throughput",
                    "/service_view/output_token_throughput",
                    "/request_view/ttft/p50",
                    "/request_view/ttft/p99",
                )
            ],
        )
    ]
}


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
            if determined
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
    request_rate = positive_numeric(raw_case, "request_rate")
    duration_seconds = positive_numeric(raw_case, "benchmark_duration_seconds")
    maximum_concurrency = token_count(raw_case, "maximum_request_concurrency")
    peak_concurrency = token_count(raw_case, "peak_concurrent_requests")
    records = object_list(raw_case.get("requests"), "case requests")
    if not records:
        raise ValueError("case requests must not be empty")

    # Detect workload mode: random cases have
    # input_length/output_length,
    # dataset cases do not.
    is_random_mode = "input_length" in raw_case and "output_length" in raw_case
    input_length: int | None = None
    output_length: int | None = None
    if is_random_mode:
        input_length = positive_token_count(raw_case, "input_length")
        output_length = positive_token_count(raw_case, "output_length")

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
        input_tokens = token_count(record, "input_tokens")
        output_tokens = token_count(record, "output_tokens")
        if is_random_mode and (
            input_tokens != input_length or output_tokens != output_length
        ):
            raise ValueError(
                "successful request token counts must match the case"
            )
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
        input_token_samples.append(float(input_tokens))
        output_token_samples.append(float(output_tokens))
        total_input_tokens += input_tokens
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
            total_output_tokens / duration_seconds, "token/s"
        ),
        "total_token_throughput": scalar(
            (total_input_tokens + total_output_tokens) / duration_seconds,
            "token/s",
        ),
    }
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

    case_info: dict[str, Any] = {
        "request_rate": request_rate,
    }
    if is_random_mode:
        case_info["input_length"] = input_length
        case_info["output_length"] = output_length
    else:
        case_info["arrival_process"] = raw_case.get("arrival_process")
        case_info["burstiness"] = raw_case.get("burstiness")
        case_info["num_prompts"] = raw_case.get("num_prompts")
        case_info["max_tokens"] = raw_case.get("max_tokens")
        # Drop None values
        case_info = {
            field_name: field_value
            for field_name, field_value in case_info.items()
            if field_value is not None
        }

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
            "report": REPORT,
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
