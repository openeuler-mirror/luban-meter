"""Token-weighted cache measurement over formal request responses only."""

from collections import defaultdict
from fractions import Fraction

from luban_meter.api import Metric, metrics

from .common.cache_usage import CacheUsage, parse_cache_usage


def count(value):
    return value if type(value) is int and value >= 0 else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def request_set(facts):
    """Reject ambiguous identities; explicit non-formal rows are excluded."""
    rows = facts.get("requests")
    issues = []
    successes = []
    seen = set()
    failed = incomplete = excluded = 0
    if not isinstance(rows, list):
        return [], 0, 0, 0, 0, ["requests_missing"]
    for row in rows:
        if not isinstance(row, dict):
            issues.append("malformed_request")
            continue
        if row.get("phase", "formal") != "formal":
            excluded += 1
            continue
        identity = row.get("request_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            issues.append("ambiguous_request_identity")
            continue
        seen.add(identity)
        response = row.get("response") or {}
        if not isinstance(response, dict):
            issues.append("malformed_response")
            incomplete += 1
            continue
        if row.get("status") == "failed":
            failed += 1
        elif row.get("status") == response.get("status") == "success":
            successes.append(response)
        else:
            incomplete += 1
            issues.append("incomplete_request")
    return successes, failed, incomplete, len(seen), excluded, issues


def summarize_responses(responses, input_length, shared):
    """Use paired counts for rates and all successful prompts for coverage."""
    measured = []
    prompts = []
    missing = invalid = conforming = short = excess = zero = reused = 0
    for response in responses:
        usage = parse_cache_usage(response.get("usage"))
        prompt = usage.prompt_tokens
        if prompt is not None:
            prompts.append(prompt)
            conforming += prompt == input_length
        if ("input_tokens" in response and prompt is not None
                and count(response["input_tokens"]) != prompt):
            usage = CacheUsage("invalid", reason="inconsistent_input_count")
        if usage.state == "missing":
            missing += 1
        elif usage.state == "invalid":
            invalid += 1
        else:
            measured.append((prompt, usage.cached_tokens))
            short += usage.cached_tokens < shared
            excess += usage.cached_tokens > shared
            zero += usage.cached_tokens == 0
            reused += usage.cached_tokens > 0
    measured_prompt = sum(p for p, _ in measured)
    measured_cached = sum(c for _, c in measured)
    total_prompt = (
        sum(prompts) if responses and len(prompts) == len(responses) else None
    )
    return {
        "reporting_requests": len(measured),
        "missing_cache_stats_requests": missing,
        "invalid_cache_stats_requests": invalid,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": measured_cached if measured else None,
        "measured_prompt_tokens": measured_prompt if measured else None,
        "request_coverage": ratio(len(measured), len(responses)),
        "token_coverage": ratio(measured_prompt, total_prompt),
        "input_length_conformance_rate": (
            ratio(conforming, len(responses))
            if len(prompts) == len(responses) else None
        ),
        "input_length_reporting_requests": len(prompts),
        "conforming_input_requests": conforming,
        "observed_subset_hit_rate": ratio(measured_cached, measured_prompt),
        "below_expected_requests": short,
        "above_expected_requests": excess,
        "zero_hit_requests": zero,
        "reuse_request_rate": ratio(reused, len(measured)),
    }


def warmup_condition(warmup, shared, prefix_count):
    if (warmup.get("warmup_completed") is False
            or warmup.get("warmup_fully_matched") is False):
        return False
    if shared == 0:
        return (
            warmup.get("status") == "not_applicable"
            and warmup.get("phase") == "capability"
            and warmup.get("cached_tokens") == 0
            and "reason" not in warmup
        )
    return warmup.get("status") == "success" and all(
        warmup.get(stage + "_completed") == prefix_count
        for stage in ("seed", "probe", "verify")
    )


@metrics.register("prefix_cache_hit_rate")
class PrefixCacheMetric(Metric):
    requires = frozenset({"prefix_cache_usage"})

    def compute(self, facts):
        p = facts["parameters"]
        layout = facts["metadata"]["prefix_cache"]
        shared = layout["shared_prefix_tokens"]
        target = Fraction(str(p["prefix_cache_target"]))
        theoretical = Fraction(shared, p["input_tokens"])
        tolerance = Fraction(str(p.get("prefix_cache_runtime_tolerance", 0)))
        successful, failed, incomplete, observed, excluded, issues = (
            request_set(facts)
        )
        summary = summarize_responses(successful, p["input_tokens"], shared)
        planned = count(p.get("num_requests"))
        admitted = count(facts.get("window", {}).get("admitted_requests"))
        if admitted is not None and admitted < observed:
            issues.append("admission_count_mismatch")
        complete = facts.get("collection_complete") is True
        # A complete, uniquely identified array establishes R if older
        # facts lack the admission counter.
        submitted = admitted
        if submitted is None and complete and not issues:
            submitted = observed
        closed = complete and submitted == observed and incomplete == 0
        measured = summary["reporting_requests"]
        success_coverage = summary["request_coverage"]
        success_token_coverage = summary["token_coverage"]
        all_successful = (
            submitted is not None and submitted > 0
            and submitted == len(successful)
        )
        summary.update(
            successful_request_coverage=success_coverage,
            successful_token_coverage=success_token_coverage,
            request_coverage=ratio(measured, submitted),
            token_coverage=(
                success_token_coverage if closed and all_successful else None
            ),
        )
        actual = None
        if (closed and all_successful and not issues and successful
                and measured == len(successful)
                and summary["measured_prompt_tokens"]):
            actual = Fraction(
                summary["total_cached_tokens"],
                summary["measured_prompt_tokens"],
            )
        if issues or summary["invalid_cache_stats_requests"]:
            measurement = "invalid"
        elif actual is not None:
            measurement = "valid"
        elif measured and summary["measured_prompt_tokens"]:
            measurement = "partial"
        else:
            measurement = "unavailable"

        reasons = list(issues)
        for condition, reason in (
            (not complete, "collection_incomplete"),
            (not closed, "request_set_not_closed"),
            (not successful, "no_successful_requests"),
            (failed, "failed_formal_requests"),
            (not all_successful, "formal_observation_incomplete"),
            (summary["missing_cache_stats_requests"], "missing_cache_stats"),
            (summary["invalid_cache_stats_requests"], "invalid_cache_stats"),
            (measured and not summary["measured_prompt_tokens"],
             "zero_token_denominator"),
        ):
            if condition:
                reasons.append(reason)

        warmup = facts["metadata"].get("cache_warmup", {})
        warmed = warmup_condition(warmup, shared, p["prefix_count"])
        runtime_passed = (
            abs(actual - theoretical) <= tolerance
            if actual is not None else None
        )
        exact_target = actual == target if actual is not None else None
        values = {
            **summary,
            "target_rate": float(target),
            "theoretical_shared_ratio": float(theoretical),
            "actual_hit_rate": float(actual) if actual is not None else None,
            "construction_deviation": float(theoretical - target),
            "runtime_deviation": (
                float(actual - theoretical) if actual is not None else None
            ),
            "target_deviation": (
                float(actual - target) if actual is not None else None
            ),
            "runtime_tolerance": float(tolerance),
            "planned_requests": planned,
            "submitted_requests": submitted,
            "completed_requests": len(successful) + failed,
            "successful_requests": len(successful),
            "failed_requests": failed,
            "incomplete_requests": (
                submitted - len(successful) - failed
                if submitted is not None and submitted >= observed else None
            ),
            "success_rate": ratio(len(successful), submitted),
        }
        groups = defaultdict(list)
        for response in successful:
            identity = response.get("prefix_cache", {}).get("prefix_id")
            key = (
                str(identity) if type(identity) is int
                and 0 <= identity < p["prefix_count"] else "unknown"
            )
            groups[key].append(response)
        return {
            "status": {
                "valid": "success", "partial": "partial_failed",
                "invalid": "failed", "unavailable": "unavailable",
            }[measurement],
            "values": {
                name: {
                    "value": value,
                    "unit": (
                        "token" if name.endswith("_tokens")
                        else "request" if name.endswith("_requests")
                        else "ratio"
                    ),
                }
                for name, value in values.items()
            },
            "measurement": {
                "status": measurement, "scope": "submitted formal requests",
                "formal_observation_complete": actual is not None,
                "reasons": sorted(set(reasons)),
                "collection_complete": complete,
            },
            "warmup": {
                "warmup_status": warmup.get("status", "unknown"),
                "warmup_policy": warmup.get(
                    "warmup_policy",
                    p.get("prefix_cache_warmup_policy", "strict"),
                ),
                "warmup_completed": warmup.get(
                    "warmup_completed", warmed if warmup else None,
                ),
                "warmup_fully_matched": warmup.get(
                    "warmup_fully_matched", warmed if warmup else None,
                ),
                "warmup_under_hit_prefixes": warmup.get(
                    "warmup_under_hit_prefixes",
                ),
                "warmup_summary": warmup.get("warmup_summary", {}),
                "reasons": (
                    ["warmup_cache_mismatch"]
                    if warmup.get("warmup_completed") is True
                    and warmup.get("warmup_fully_matched") is False
                    else ["warmup_not_verified"] if warmup and not warmed
                    else []
                ),
            },
            "target_attainment": {
                "status": (
                    "unknown" if actual is None
                    else "failed" if not runtime_passed
                    else "construction_limited" if theoretical != target
                    else "passed"
                ),
                "runtime_within_tolerance": runtime_passed,
                "exact_target_reached": exact_target,
                "construction_exact": theoretical == target,
                "above_theoretical": (
                    actual > theoretical if actual is not None else None
                ),
            },
            "diagnostics": {
                "eviction": (
                    "suspected" if warmed
                    and summary["below_expected_requests"] else "unknown"
                ),
                "eviction_confirmed": False,
                "post_warmup_shortfall": (
                    warmed and summary["below_expected_requests"] > 0
                ),
                "excluded_nonformal_records": excluded,
                "prefix_groups": {
                    group: {
                        "successful_requests": len(rows),
                        **summarize_responses(rows, p["input_tokens"], shared),
                    }
                    for group, rows in sorted(groups.items())
                },
            },
            "metadata": {
                "source": "facts.requests[*].response.usage",
                "formula": "sum(M.cached_tokens) / sum(M.prompt_tokens)",
                "sets": {
                    "R": "submitted formal requests; admission counter",
                    "S": "successful completed responses in R",
                    "M": "S with valid, consistent paired cache counts",
                },
                "denominators": {
                    "request_coverage": "|R|, known admitted requests",
                    "successful_request_coverage": "|S|",
                    "token_coverage": "sum(R.prompt_tokens), all R observed",
                    "successful_token_coverage": "sum(S.prompt_tokens)",
                    "success_rate": "|R|",
                    "input_length_conformance_rate": "|S|, all S known",
                    "reuse_request_rate": "|M|",
                },
                "token_totals_scope": {
                    "total_prompt_tokens": "S, null if any prompt missing",
                    "total_cached_tokens": "M, null if M empty",
                    "measured_prompt_tokens": "M, null if M empty",
                },
                "window": facts.get("window", {}),
                "limitations": [
                    "Cache shortfall alone cannot confirm eviction.",
                    "Actual rate requires all submitted requests observed.",
                    "Subset counts cover successful responses only.",
                ],
            },
        }
