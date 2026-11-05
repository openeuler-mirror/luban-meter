"""Service observation collection and scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.benchmarking.generation_performance.common.prometheus import (
    counter_value,
    gauge_value,
    histogram_quantiles,
    histogram_series_totals,
    select_metric,
)
from luban_meter.benchmarking.generation_performance.common.statistics import (
    scalar,
    summarize,
)


def _object_list(value: Any, name: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise TypeError(f"{name} must be a list of objects")
    return list(value)


def _aggregate_gauge(
    snapshots: Sequence[Mapping[str, Any]],
    metric_name: str,
    unit: str,
) -> dict[str, Any] | None:
    values: list[float] = []
    for snap in snapshots:
        entry = select_metric(snap.get("metrics", {}), metric_name)
        metric_value = gauge_value(entry)
        if metric_value is not None:
            values.append(metric_value)
    if not values:
        return None
    return summarize(values, unit)


def _aggregate_counter_rate(
    snapshots: Sequence[Mapping[str, Any]],
    metric_name: str,
    duration_seconds: float,
    unit: str,
) -> dict[str, Any] | None:
    first_val: float | None = None
    last_val: float | None = None
    for snap in snapshots:
        entry = select_metric(snap.get("metrics", {}), metric_name)
        metric_value = counter_value(entry)
        if metric_value is None:
            continue
        if first_val is None:
            first_val = metric_value
        last_val = metric_value
    if first_val is None or last_val is None:
        return None
    if last_val < first_val:
        return scalar(0, unit, precision=3)
    if duration_seconds <= 0:
        return None
    rate = (last_val - first_val) / duration_seconds
    return scalar(rate, unit, precision=3)


def _aggregate_histogram_quantiles(
    snapshots: Sequence[Mapping[str, Any]],
    metric_name: str,
    unit: str,
) -> dict[str, Any] | None:
    all_samples: list[dict[str, Any]] = []
    for snap in snapshots:
        entry = select_metric(snap.get("metrics", {}), metric_name)
        if entry is not None:
            all_samples.extend(entry.get("samples", []))

    if not all_samples:
        return None

    quantiles = histogram_quantiles(all_samples, (0.50, 0.90, 0.99))
    result: dict[str, Any] = {"unit": unit}
    for quantile_name in ("p50", "p90", "p99"):
        metric_value = quantiles.get(quantile_name)
        result[quantile_name] = (
            round(metric_value, 3) if metric_value is not None else None
        )
    total_count = quantiles.get("_count")
    total_sum = quantiles.get("_sum")
    if total_count is not None:
        result["count"] = int(total_count)
    if total_sum is not None:
        result["sum"] = round(float(total_sum), 3)
    return result


def _aggregate_window_delta_mean(
    snapshots: Sequence[Mapping[str, Any]],
    metric_name: str,
    unit: str,
) -> dict[str, Any] | None:
    """Average per-request value of a histogram over the observation window.

    Diffs the cumulative ``_sum``/``_count`` of each label series between
    the first and the last snapshot that exposes the metric, then sums
    numerator and denominator across series:
    ``mean = total_delta_sum / total_delta_count``.

    Series rules:
      - non-zero baselines in the first snapshot are subtracted;
      - a series that only appears later starts from a zero baseline;
      - a counter reset is detected between adjacent snapshots that
        expose the metric (cumulative totals may only grow); the whole
        series is dropped with a diagnostic, even when the final totals
        exceed the window start again;
      - series missing from the last snapshot are dropped with a
        diagnostic.

    Values are persisted at their original precision; the presentation
    layers format them. The result carries the first/last snapshot
    timestamps, the label sets of the included series and the per-series
    deltas so the mean is recomputable from the persisted facts. The
    denominator counts every request reported in the window, including
    ones that waited about zero seconds. Returns None when the metric is
    absent from all snapshots; returns a reason-marked result without
    substituting zero when the count delta is not positive.
    """
    first_totals: (
        dict[tuple[tuple[str, str], ...], tuple[float, float]] | None
    ) = None
    last_totals: (
        dict[tuple[tuple[str, str], ...], tuple[float, float]] | None
    ) = None
    first_timestamp: Any = None
    last_timestamp: Any = None
    last_seen: dict[
        tuple[tuple[str, str], ...], tuple[float, float]
    ] = {}
    reset_keys: set[tuple[tuple[str, str], ...]] = set()
    for snap in snapshots:
        totals = histogram_series_totals(
            snap.get("metrics", {}), metric_name
        )
        if not totals:
            continue
        for key, values in totals.items():
            seen = last_seen.get(key)
            if seen is not None and (
                values[0] < seen[0] or values[1] < seen[1]
            ):
                reset_keys.add(key)
            last_seen[key] = values
        if first_totals is None:
            first_totals = totals
            first_timestamp = snap.get("finished_at")
        last_totals = totals
        last_timestamp = snap.get("finished_at")
    if first_totals is None or last_totals is None:
        return None

    delta_sum = 0.0
    delta_count = 0.0
    series_deltas: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for key in sorted(set(first_totals) | set(last_totals)):
        labels = dict(key)
        if key in reset_keys:
            dropped.append({
                "labels": labels,
                "reason": "counter reset between snapshots",
            })
            continue
        if key not in last_totals:
            dropped.append({
                "labels": labels,
                "reason": "series missing from the last snapshot",
            })
            continue
        last_sum, last_count = last_totals[key]
        first_sum, first_count = first_totals.get(key, (0.0, 0.0))
        delta_sum += last_sum - first_sum
        delta_count += last_count - first_count
        series_deltas.append({
            "labels": labels,
            "delta_sum": last_sum - first_sum,
            "delta_count": {"value": int(last_count - first_count),
                            "unit": "request"},
        })
    series_count = len(series_deltas)
    result: dict[str, Any] = {
        "unit": unit,
        "boundary": "server",
        "window": {
            "started_at": {
                "value": first_timestamp, "unit": "unix_s",
            },
            "ended_at": {
                "value": last_timestamp, "unit": "unix_s",
            },
        },
        "series": series_deltas,
        "series_count": {"value": series_count, "unit": "series"},
    }
    if dropped:
        result["dropped_series"] = dropped
    if delta_count <= 0:
        return {
            **result,
            "mean": None,
            "delta_sum": None,
            "delta_count": None,
            "reason": "no completed observations in the window",
        }
    return {
        **result,
        "mean": delta_sum / delta_count,
        "delta_sum": delta_sum,
        "delta_count": {"value": int(delta_count), "unit": "request"},
    }


def _aggregate_prefix_cache_hit_rate(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    hits: list[float] = []
    queries: list[float] = []
    for snap in snapshots:
        metrics = snap.get("metrics", {})
        hits_entry = select_metric(metrics, "vllm:prefix_cache_hits_total")
        queries_entry = select_metric(
            metrics, "vllm:prefix_cache_queries_total"
        )
        hits_val = counter_value(hits_entry)
        queries_val = counter_value(queries_entry)
        if (
            hits_val is not None
            and queries_val is not None
            and queries_val > 0
        ):
            hits.append(hits_val)
            queries.append(queries_val)
    if not hits:
        return None
    rates: list[float] = []
    for cache_hit_count, cache_query_count in zip(hits, queries):
        if cache_query_count > 0:
            rates.append(cache_hit_count / cache_query_count)
    if not rates:
        return None
    return summarize(rates, "ratio")


def _infer_bottleneck(
    server_status: dict[str, Any],
    latency: dict[str, Any],
) -> dict[str, Any]:
    kv_cache_pressure: dict[str, Any] = {}
    queue_pressure: dict[str, Any] = {}

    kv_cache = server_status.get("kv_cache_usage_perc")
    if kv_cache is not None:
        kv_mean = kv_cache.get("mean")
        kv_max = kv_cache.get("max")
        if kv_mean is not None and kv_max is not None:
            if kv_mean > 0.8:
                kv_cache_pressure = {
                    "level": "high",
                    "mean_usage": kv_mean,
                    "peak_usage": kv_max,
                    "suggestion": "increase GPU count or reduce max_model_len",
                }
            elif kv_max > 0.95:
                kv_cache_pressure = {
                    "level": "spike",
                    "mean_usage": kv_mean,
                    "peak_usage": kv_max,
                    "suggestion": (
                        "consider increasing GPU count to handle peaks"
                    ),
                }
            else:
                kv_cache_pressure = {
                    "level": "normal",
                    "mean_usage": kv_mean,
                    "peak_usage": kv_max,
                }

    num_waiting = server_status.get("num_requests_waiting")
    if num_waiting is not None:
        wait_mean = num_waiting.get("mean")
        wait_max = num_waiting.get("max")
        if wait_mean is not None and wait_max is not None:
            if wait_mean > 0:
                queue_pressure = {
                    "level": "high",
                    "mean_queue_length": wait_mean,
                    "peak_queue_length": wait_max,
                    "suggestion": (
                        "increase max_num_seqs or reduce concurrency"
                    ),
                }
            elif wait_max > 0:
                queue_pressure = {
                    "level": "spike",
                    "mean_queue_length": wait_mean,
                    "peak_queue_length": wait_max,
                    "suggestion": "monitor for occasional queue buildup",
                }
            else:
                queue_pressure = {
                    "level": "normal",
                    "mean_queue_length": wait_mean,
                    "peak_queue_length": wait_max,
                }

    prefill = latency.get("prefill_time")
    decode = latency.get("decode_time")
    prefill_decode_ratio: dict[str, Any] | None = None
    if prefill is not None and decode is not None:
        p50_prefill = prefill.get("p50")
        p50_decode = decode.get("p50")
        if (
            p50_prefill is not None
            and p50_decode is not None
            and p50_decode > 0
        ):
            ratio = p50_prefill / p50_decode
            prefill_decode_ratio = {
                "p50_prefill_decode_ratio": scalar(
                    ratio, "ratio", precision=4
                ),
                "interpretation": (
                    "prefill-heavy workload"
                    if ratio > 1.0
                    else "decode-heavy workload"
                ),
            }

    dominant = "none"
    if not kv_cache_pressure and not queue_pressure:
        dominant = "insufficient_data"
    elif (
        kv_cache_pressure.get("level") == "high"
        and queue_pressure.get("level") == "high"
    ):
        dominant = "kv_cache_and_queue"
    elif kv_cache_pressure.get("level") == "high":
        dominant = "kv_cache"
    elif queue_pressure.get("level") == "high":
        dominant = "queue"
    elif queue_pressure.get("level") == "spike":
        dominant = "queue_spike"
    else:
        dominant = "none"

    analysis: dict[str, Any] = {
        "dominant_bottleneck": dominant,
        "kv_cache_pressure": kv_cache_pressure if kv_cache_pressure else None,
        "queue_pressure": queue_pressure if queue_pressure else None,
    }
    if prefill_decode_ratio is not None:
        analysis["prefill_decode_ratio"] = prefill_decode_ratio
    return analysis


def _aggregate_concurrency(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    total_values: list[float] = []
    for snap in snapshots:
        metrics = snap.get("metrics", {})
        running = gauge_value(
            select_metric(metrics, "vllm:num_requests_running")
        )
        waiting = gauge_value(
            select_metric(metrics, "vllm:num_requests_waiting")
        )
        if running is not None and waiting is not None:
            total_values.append(running + waiting)
    if not total_values:
        return None
    return summarize(total_values, "request")


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")

    raw_snapshots = _object_list(raw_metrics.get("snapshots"), "snapshots")
    if not raw_snapshots:
        raise ValueError("snapshots must not be empty")

    snapshots = [snap for snap in raw_snapshots if snap.get("error") is None]
    if not snapshots:
        raise ValueError("all snapshots contain errors")

    raw_metadata = raw_result.get("metadata")
    duration_seconds = 0.0
    if isinstance(raw_metadata, Mapping):
        duration_seconds = float(raw_metadata.get("collect_duration", 0.0))

    # --- Server Status ---
    server_status: dict[str, Any] = {}
    gauges = {
        "num_requests_running": ("vllm:num_requests_running", "request"),
        "num_requests_waiting": ("vllm:num_requests_waiting", "request"),
        "num_requests_swapped": ("vllm:num_requests_swapped", "request"),
    }
    for key, (name, unit) in gauges.items():
        result = _aggregate_gauge(snapshots, name, unit)
        if result is not None:
            server_status[key] = result

    kv_cache = _aggregate_gauge(
        snapshots, "vllm:kv_cache_usage_perc", "ratio"
    ) or _aggregate_gauge(snapshots, "vllm:gpu_cache_usage_perc", "ratio")
    if kv_cache is not None:
        server_status["kv_cache_usage_perc"] = kv_cache

    prefix_cache = _aggregate_prefix_cache_hit_rate(snapshots)
    if prefix_cache is not None:
        server_status["prefix_cache_hit_rate"] = prefix_cache

    total_concurrency = _aggregate_concurrency(snapshots)
    if total_concurrency is not None:
        server_status["total_concurrency"] = total_concurrency

    # --- Throughput ---
    throughput: dict[str, Any] = {}
    counters = {
        "prompt_token_throughput": ("vllm:prompt_tokens_total", "token/s"),
        "generation_token_throughput": (
            "vllm:generation_tokens_total",
            "token/s",
        ),
    }
    for key, (name, unit) in counters.items():
        result = _aggregate_counter_rate(
            snapshots, name, duration_seconds, unit
        )
        if result is not None:
            throughput[key] = result

    request_rate = _aggregate_counter_rate(
        snapshots, "vllm:request_success_total", duration_seconds, "req/s"
    )
    if request_rate is not None:
        throughput["request_throughput"] = request_rate

    # --- Latency Decomposition ---
    latency: dict[str, Any] = {}
    histograms = {
        "queue_time": ("vllm:request_queue_time_seconds", "s"),
        "prefill_time": ("vllm:request_prefill_time_seconds", "s"),
        "decode_time": ("vllm:request_decode_time_seconds", "s"),
        "model_execution_time": ("vllm:request_inference_time_seconds", "s"),
        "ttft": ("vllm:time_to_first_token_seconds", "s"),
        "tpot": ("vllm:request_time_per_output_token_seconds", "s"),
        "e2e_latency": ("vllm:e2e_request_latency_seconds", "s"),
    }
    for key, (name, unit) in histograms.items():
        result = _aggregate_histogram_quantiles(snapshots, name, unit)
        if result is not None:
            latency[key] = result

    queue_time_window = _aggregate_window_delta_mean(
        snapshots, "vllm:request_queue_time_seconds", "s"
    )
    if queue_time_window is not None:
        latency["queue_time_window"] = queue_time_window

    # --- Request Profile ---
    request_profile: dict[str, Any] = {}
    profile_hists = {
        "prompt_tokens": ("vllm:request_prompt_tokens", "token"),
        "generation_tokens": ("vllm:request_generation_tokens", "token"),
    }
    for key, (name, unit) in profile_hists.items():
        result = _aggregate_histogram_quantiles(snapshots, name, unit)
        if result is not None:
            request_profile[key] = result

    # --- Bottleneck Analysis ---
    bottleneck = _infer_bottleneck(server_status, latency)

    # --- Metadata ---
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    metadata.update(
        {
            "successful_snapshot_count": len(snapshots),
            "total_snapshot_count": len(raw_snapshots),
            "measurement": "vllm_service_metrics_aggregation",
        }
    )

    return {
        "status": "success",
        "metrics": {
            "server_status": server_status,
            "throughput": throughput,
            "latency_decomposition": latency,
            "request_profile": request_profile,
            "bottleneck_analysis": bottleneck,
        },
        "metadata": metadata,
        "error": None,
    }
