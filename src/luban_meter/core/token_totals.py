"""Compute persisted throughput from actual formal request evidence."""

import math
from collections import Counter


def token_pair(record, first, second):
    values = [record.get(first), record.get(second)]
    if all(type(value) is int and value >= 0 for value in values):
        return sum(values)
    return None


def token_throughput_value(total, duration):
    """Shared formula for successful tokens per formal collection second."""
    if (type(total) is int and total >= 0
            and type(duration) in (int, float)
            and math.isfinite(duration) and duration > 0):
        return total / duration
    return None


def request_totals(requests, samples):
    """Count logical requests once, preserving missing usage diagnostics."""
    samples_by_id = {row.get("sample_id"): row.get("facts") or {}
                     for row in samples}
    sample_counts = Counter(row.get("sample_id") for row in samples)
    multiplicity = Counter(row.get("sample_id") for row in requests)
    total, successful, failed, missing = 0, 0, 0, 0
    unavailable = []
    if samples and not requests:
        unavailable.append({"reason": "missing formal request records"})
    for row in requests:
        cost = row.get("logical_requests", 1)
        if type(cost) is not int or cost < 1:
            unavailable.append({
                "request_id": row.get("request_id"),
                "reason": "invalid logical request count",
            })
            continue
        status = row.get("status")
        if status == "failed":
            failed += cost
            continue
        if status != "success":
            unavailable.append({
                "request_id": row.get("request_id"),
                "reason": "unknown request completion status",
            })
            continue
        successful += cost
        sample_id = row.get("sample_id")
        facts = samples_by_id.get(sample_id, {})
        unique_sample = (
            multiplicity[sample_id] == 1 and sample_counts[sample_id] == 1
        )
        usage = row.get("usage")
        if usage is None and unique_sample:
            usage = facts.get("usage")
        if isinstance(usage, dict):
            count = token_pair(usage, "prompt_tokens", "completion_tokens")
        elif isinstance(row.get("response"), dict):
            count = token_pair(
                row["response"], "input_tokens", "output_tokens"
            )
        elif (unique_sample and isinstance(facts.get("requests"), list)
              and len(facts["requests"]) == cost):
            parts = [token_pair(item, "actual_prompt_tokens",
                                "actual_output_tokens")
                     for item in facts["requests"]]
            count = sum(parts) if None not in parts else None
        else:
            count = None
        if count is None:
            missing += cost
            unavailable.append({
                "request_id": row.get("request_id"),
                "sample_id": sample_id,
                "reason": "missing or ambiguous actual token usage",
            })
        else:
            total += count
    return {
        "total_tokens": None if unavailable else total,
        "successful_requests": successful,
        "failed_requests": failed,
        "missing_usage_requests": missing,
        "unavailable_requests": unavailable,
    }


def request_total(requests, samples):
    """Return the validated successful formal token total, if available."""
    return request_totals(requests, samples)["total_tokens"]


def throughput_result(total, duration, *, reason=None, metadata=None):
    """Build the common Case/run metric before persistence."""
    rate = token_throughput_value(total, duration)
    valid_total = type(total) is int and total >= 0
    valid_duration = (
        type(duration) in (int, float)
        and math.isfinite(duration) and duration > 0
    )
    result = {
        "status": "success" if rate is not None else "unavailable",
        "observation_boundary": "client",
        "metadata": {
            "formula": (
                "successful_input_plus_output_tokens / formal_duration_s"
            ),
            "token_scope": "successful formal model requests",
            "warmup_tokens_included": False,
            "warmup_duration_included": False,
            "failed_request_tokens_included": False,
            "answer_parse_failures_included": True,
            **(metadata or {}),
        },
        "values": {"service_view": {
            "total_tokens": {
                "value": total if valid_total else None, "unit": "token",
            },
            "benchmark_duration": {
                "value": duration if valid_duration else None, "unit": "s",
            },
            "total_token_throughput": {"value": rate, "unit": "token/s"},
        }},
    }
    if rate is None:
        result["reason"] = reason or (
            "missing actual token usage" if not valid_total
            else "missing or invalid formal duration"
        )
    return result


def run_throughput(tasks):
    """Aggregate persisted Case metrics with an explicit inclusion ledger."""
    total, duration = 0, 0.0
    included, skipped, unavailable = [], [], []
    for task_index, task in enumerate(tasks):
        if task.get("category") != "generation_performance":
            continue
        identity = {"task_index": task_index, "task": task.get("name")}
        if task["status"] == "skipped":
            skipped.append({**identity, "reason": "task skipped"})
            continue
        if not task.get("cases"):
            unavailable.append({**identity, "reason": "missing Case results"})
        for case_index, case in enumerate(task.get("cases", [])):
            item = {**identity, "case_index": case_index,
                    "case": case.get("id"), "status": case["status"]}
            if case["status"] == "skipped":
                skipped.append({**item, "reason": "Case skipped"})
                continue
            metric = saved_throughput_metric(case)
            if (case["status"] in ("discarded", "interrupted")
                    or case.get("collection_complete") is not True):
                unavailable.append({
                    **item, "reason": "Case collection not complete",
                })
                continue
            if (not isinstance(metric, dict)
                    or metric.get("status", "success") not in (
                        "success", "partial_failed", "failed")):
                unavailable.append({
                    **item, "reason": (
                        metric.get("reason", "Case metric unavailable")
                        if isinstance(metric, dict) else "Case metric missing"
                    ),
                })
                continue
            view = (metric.get("values") or {}).get("service_view") or {}
            count = (view.get("total_tokens") or {}).get("value")
            seconds = (view.get("benchmark_duration") or {}).get("value")
            if token_throughput_value(count, seconds) is None:
                unavailable.append({
                    **item, "reason": "invalid saved Case totals or duration",
                })
                continue
            total += count
            duration += seconds
            included.append({
                **item, "total_tokens": count, "formal_duration_s": seconds,
            })
    reason = None
    if unavailable:
        total, duration = None, None
        reason = "one or more Cases lack complete throughput evidence"
    elif not included:
        total, duration = None, None
        reason = "no measured Cases"
    return throughput_result(total, duration, reason=reason, metadata={
        "scope": "generation_performance Cases, independent of report filters",
        "included_cases": included,
        "skipped_cases": skipped,
        "unavailable_cases": unavailable,
    })


def formal_throughput(facts):
    """Shared calculation used by existing performance metrics."""
    counts = request_totals(facts["requests"], facts["samples"])
    complete = facts.get("collection_complete") is True
    total = counts["total_tokens"] if complete else None
    reason = None
    if not complete:
        reason = "Case collection not complete"
    elif counts["unavailable_requests"]:
        reason = "missing or ambiguous successful request token usage"
    result = throughput_result(
        total, facts["window"]["duration_s"], reason=reason,
        metadata={"scope": "Case formal collection", **counts},
    )
    if result.get("reason"):
        for node in result["values"]["service_view"].values():
            if node["value"] is None:
                node["reason"] = result["reason"]
    return result


def saved_throughput_metric(result):
    """Find a saved provider; legacy common metrics remain readable."""
    metrics = result.get("metrics") or {}
    for name, metric in metrics.items():
        if name == "token_throughput" or not isinstance(metric, dict):
            continue
        service = (metric.get("values") or {}).get("service_view") or {}
        if "total_token_throughput" in service:
            return metric
    return metrics.get("token_throughput")
