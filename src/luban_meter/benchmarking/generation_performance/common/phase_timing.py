"""Client-observed Chat output spans and their per-request population."""

import math
from collections import Counter
from itertools import pairwise

from luban_meter.log import get_logger

from .statistics import scalar, summarize

logger = get_logger(__name__)

PHASE_METRICS = (
    "reasoning_duration", "content_duration", "first_content_latency",
    "reasoning_ttft",
)
CONTENT_METRICS = ("content_duration", "first_content_latency")
TOKEN_METRICS = ("reasoning_tpot", "reasoning_itl")
EXCLUSIONS = (
    "failed", "truncated", "missing_finish", "other_finish",
)
CONDITION = (
    "successful Chat requests with finish_reason=stop and saved channel "
    "timestamps; each metric excludes absent channels; one event has a "
    "zero span; channel spans may overlap and do not sum to E2EL"
)


def request_phase_timing(record, *, thinking_metrics=False):
    """Derive milliseconds from saved offsets without changing facts.

    An exclusion applies to the entire request, including an observed
    thinking phase preceding truncated content. Failed partial streams
    have no usable observation. Successful records contain both channel
    timestamp lists; an empty list denotes an unobserved channel.
    """
    metric_names = (
        (*PHASE_METRICS, *TOKEN_METRICS)
        if thinking_metrics else CONTENT_METRICS
    )
    result = {name + "_ms": None for name in metric_names}
    result.update(status="failed", content_event_count=None)
    if thinking_metrics:
        result.update(
            reasoning_event_count=None, interleaved=None,
            reasoning_stream_event_itl_samples_ms=[],
        )
    if record.get("status") != "success":
        return result
    content = record["content_event_offsets_ms"]
    reasoning = (
        record["reasoning_event_offsets_ms"] if thinking_metrics else []
    )
    channels = [("content", content)]
    if thinking_metrics:
        channels.append(("reasoning", reasoning))
    for name, times in channels:
        result[name + "_event_count"] = len(times)
        result[name + "_duration_ms"] = (
            round(times[-1] - times[0], 3) if times else None
        )
    result["first_content_latency_ms"] = content[0] if content else None
    if thinking_metrics:
        result["interleaved"] = bool(
            reasoning and content and content[0] < reasoning[-1]
        )
    if reasoning:
        result["reasoning_ttft_ms"] = reasoning[0]
        result["reasoning_stream_event_itl_samples_ms"] = [
            round(current - previous, 3)
            for previous, current in pairwise(reasoning)
        ]
        tokens = record.get("reasoning_tokens")
        if (isinstance(tokens, int) and not isinstance(tokens, bool)
                and tokens > 0):
            # Keep full precision until the final weighted summary.
            result["reasoning_tpot_ms"] = reasoning[-1] / tokens
            if tokens > 1:
                result["reasoning_itl_ms"] = (
                    result["reasoning_duration_ms"] / (tokens - 1)
                )
    finish = record.get("finish_reason")
    if finish == "length":
        result["status"] = "truncated"
    elif finish is None:
        result["status"] = "missing_finish"
    elif finish != "stop":
        result["status"] = "other_finish"
    else:
        result["status"] = "eligible"
    return result


def phase_timing_metric(facts):
    """Aggregate each admitted Chat request once, including session turns."""
    if facts["parameters"].get("endpoint") != "/v1/chat/completions":
        logger.info("Phase timing unavailable: endpoint is not Chat")
        return {"status": "unavailable", "reason": "requires Chat endpoint"}
    thinking = facts["parameters"].get("thinking_metrics", False)
    names = PHASE_METRICS if thinking else CONTENT_METRICS
    samples = {name: [] for name in names}
    weighted = {name: [] for name in TOKEN_METRICS} if thinking else {}
    channels = ("reasoning", "content") if thinking else ("content",)
    event_intervals = []
    counts = Counter()
    token_sources = Counter()
    token_missing_reasons = Counter()
    for request in facts["requests"]:
        response = request.get("response") or {}
        timing = request_phase_timing(response, thinking_metrics=thinking)
        counts["total_requests"] += 1
        counts[timing["status"] + "_requests"] += 1
        if thinking:
            counts["interleaved_requests"] += bool(timing["interleaved"])
        if timing["status"] != "eligible":
            continue
        if thinking:
            source = response.get("reasoning_tokens_source")
            token_sources[source or "unavailable"] += 1
            if source is None:
                token_missing_reasons[response.get(
                    "reasoning_tokens_reason"
                ) or "missing_reasoning_tokens"] += 1
        for name in names:
            value = timing[name + "_ms"]
            if value is not None:
                samples[name].append(value)
        for name in weighted:
            offset = int(name == "reasoning_itl")
            value = timing[name + "_ms"]
            if value is not None:
                weighted[name].append(
                    (value, response["reasoning_tokens"] - offset)
                )
        if thinking:
            event_intervals.extend(
                timing["reasoning_stream_event_itl_samples_ms"]
            )
        for channel in channels:
            size = timing[channel + "_event_count"]
            counts[channel + "_single_event_requests"] += size == 1
            counts["missing_" + channel + "_requests"] += size == 0
    counts["excluded_requests"] = sum(
        counts[reason + "_requests"] for reason in EXCLUSIONS
    )
    count_keys = (
        "total_requests", "eligible_requests", "excluded_requests",
        *(reason + "_requests" for reason in EXCLUSIONS),
        "content_single_event_requests", "missing_content_requests",
    )
    if thinking:
        count_keys += (
            "interleaved_requests", "reasoning_single_event_requests",
            "missing_reasoning_requests",
        )
    available = any(samples.values())
    result = {
        "status": "success" if available else "unavailable",
        "observation_boundary": "client",
        "condition": CONDITION,
        "values": {
            "request_view": {
                name: summarize(values, "ms")
                for name, values in samples.items()
            },
            "phase_counts": {
                name: scalar(counts[name], "request") for name in count_keys
            },
        },
    }
    if not available:
        result["reason"] = "no eligible channel observations"
    if not thinking:
        logger.info(
            "Phase timing computed: eligible=%s excluded=%s "
            "thinking_metrics=False content_samples=%s "
            "first_content_samples=%s exclusions=%s",
            counts["eligible_requests"], counts["excluded_requests"],
            len(samples["content_duration"]),
            len(samples["first_content_latency"]),
            {reason: counts[reason + "_requests"] for reason in EXCLUSIONS},
        )
        return result
    view = result["values"]["request_view"]
    result["metadata"] = {
        "reasoning_token_count_scope": "eligible_requests",
        "reasoning_token_sources": dict(token_sources),
        "reasoning_token_missing_reasons": dict(token_missing_reasons),
    }
    view["reasoning_ttft"].update(
        sample_unit="request", aggregation="request_equal",
        percentile_method="linear_interpolation",
    )
    for name in TOKEN_METRICS:
        pairs = weighted[name]
        if any(not math.isfinite(value) or value < 0 for value, _ in pairs):
            raise ValueError("expected finite values and positive weights")
        total_weight = sum(weight for _, weight in pairs)
        mean = math.fsum(
            value * (weight / total_weight) for value, weight in pairs
        ) if pairs else None
        view[name] = {
            "unit": "ms/token", "count": len(pairs),
            "mean": round(mean, 3) if mean is not None else None,
            "weight_sum": total_weight, "aggregation": "token_weighted",
        }
        view[name].update(
            sample_unit="request",
            weight_unit="token" if name == "reasoning_tpot"
            else "token_interval",
            condition=(
                "requires observed thinking and reasoning_tokens>0; "
                "includes request-to-first-thinking wait"
                if name == "reasoning_tpot" else
                "requires observed thinking and reasoning_tokens>1; "
                "weighted mean only, not individual token measurements"
            ),
        )
    view["reasoning_stream_event_itl"] = summarize(event_intervals, "ms")
    view["reasoning_stream_event_itl"].update(
        sample_unit="event_interval", aggregation="event_interval_equal",
        percentile_method="linear_interpolation",
        condition="within-request thinking event gaps; not token intervals",
    )
    logger.info(
        "Phase timing computed: eligible=%s excluded=%s "
        "reasoning_samples=%s content_samples=%s first_content_samples=%s "
        "exclusions=%s thinking_samples=%s thinking_weights=%s "
        "thinking_token_summary=mean_only token_sources=%s "
        "token_missing_reasons=%s",
        counts["eligible_requests"], counts["excluded_requests"],
        len(samples["reasoning_duration"]), len(samples["content_duration"]),
        len(samples["first_content_latency"]),
        {reason: counts[reason + "_requests"] for reason in EXCLUSIONS},
        {name: view[name]["count"] for name in (
            "reasoning_ttft", *TOKEN_METRICS, "reasoning_stream_event_itl",
        )},
        {name: view[name]["weight_sum"] for name in TOKEN_METRICS},
        dict(token_sources), dict(token_missing_reasons),
    )
    return result
