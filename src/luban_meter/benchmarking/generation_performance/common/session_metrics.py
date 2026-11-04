"""Aggregation for multi-turn session benchmarks.

Turn-level facts are read from the request records rather than the
sample facts.  An interrupted session keeps its completed turns in the
request stream even though ``collect_sample`` records only the
interruption, so this survives Ctrl+C, session timeouts and transport
failures without special cases.

These views supplement the shared streaming request and service views;
they do not calculate those common statistics again.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .lengths import length_summary


def _as_int(value: Any) -> int:
    """Order sessions numerically without failing on odd identifiers."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _turn_index(request: Mapping[str, Any]) -> int | None:
    scheduling = request.get("scheduling")
    if not isinstance(scheduling, Mapping):
        return None
    turn = scheduling.get("turn_index")
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
        return None
    return turn


def turn_rows(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collect one row per recorded turn, ordered by session then turn.

    Read from the request stream rather than from sample facts: a
    session that was interrupted leaves only the interruption in its
    sample record, while the turns it already finished are still here.
    """
    rows = []
    for request in requests:
        turn = _turn_index(request)
        if turn is None:
            continue
        response = request.get("response")
        rows.append(
            {
                "session": _as_int(request.get("sample_id")),
                "turn": turn,
                "status": request.get("status"),
                "response": response if isinstance(response, Mapping) else {},
            }
        )
    rows.sort(key=lambda row: (row["session"], row["turn"]))
    return rows


def _slope(points: Sequence[tuple[float, float]]) -> float | None:
    """Least-squares slope; None when it cannot be defined."""
    if len(points) < 2:
        return None
    count = len(points)
    mean_x = math.fsum(x for x, _ in points) / count
    mean_y = math.fsum(y for _, y in points) / count
    denominator = math.fsum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    numerator = math.fsum(
        (x - mean_x) * (y - mean_y) for x, y in points
    )
    return round(numerator / denominator, 6)


def _mean(values: Sequence[float]) -> float | None:
    return round(math.fsum(values) / len(values), 3) if values else None


def _successful(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row["response"]
        for row in rows
        if row.get("status") == "success" and row.get("response")
    ]


def turn_view(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-turn distributions, so growth across a session is visible."""
    from .statistics import summarize

    by_turn: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        response = row.get("response")
        if not response:
            continue
        by_turn.setdefault(row["turn"], []).append(response)

    # Total successful turn observations: the sum of the per-turn
    # counts below, not the number of distinct turn indexes.
    observed = sum(len(responses) for responses in by_turn.values())
    view: dict[str, Any] = {"turns_observed": observed}
    for name, unit in (
        ("ttft_ms", "ms"),
        ("e2el_ms", "ms"),
        ("input_tokens", "token"),
        ("output_tokens", "token"),
    ):
        per_turn = []
        for turn in sorted(by_turn):
            values = [
                float(response[name])
                for response in by_turn[turn]
                if isinstance(response.get(name), (int, float))
                and not isinstance(response.get(name), bool)
            ]
            per_turn.append(
                {"turn_index": turn, **summarize(values, unit)}
            )
        view[f"{name}_by_turn"] = per_turn
    return view


def caching_view(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Quantify how much a growing context costs, and whether it is cached.

    ``first_to_rest_ttft_ratio`` compares the first turn's mean TTFT with
    the mean of all later turns.  Later turns carry strictly more
    context, so a ratio near 1 means the engine reused the prefix and a
    ratio far below 1 means it reprefilled the whole history.
    """
    from .statistics import scalar

    points: list[tuple[int, float]] = []
    context_points: list[tuple[float, float]] = []
    first: list[float] = []
    rest: list[float] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        response = row.get("response") or {}
        ttft = response.get("ttft_ms")
        if not isinstance(ttft, (int, float)) or isinstance(ttft, bool):
            continue
        turn = row["turn"]
        points.append((turn, float(ttft)))
        if turn == 0:
            first.append(float(ttft))
        else:
            rest.append(float(ttft))
        tokens = response.get("input_tokens")
        if (
            isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and tokens > 0
        ):
            context_points.append((tokens / 1000.0, float(ttft)))

    first_mean = _mean(first)
    rest_mean = _mean(rest)
    ratio = None
    if first_mean is not None and rest_mean:
        ratio = round(first_mean / rest_mean, 4)
    return {
        "first_turn_ttft_mean": (
            scalar(first_mean, "ms") if first_mean is not None
            else scalar(None, "ms")
        ),
        "rest_turn_ttft_mean": (
            scalar(rest_mean, "ms") if rest_mean is not None
            else scalar(None, "ms")
        ),
        "first_to_rest_ttft_ratio": {
            "value": ratio,
            "unit": "ratio",
            "condition": (
                "first-turn mean TTFT / later-turn mean TTFT; describes "
                "how latency scales with context, and is not evidence "
                "about prefix cache state"
            ),
        },
        "ttft_growth_per_turn": {
            "value": _slope(points),
            "unit": "ms/turn",
            "condition": "least squares over per-turn TTFT samples",
        },
        "ttft_growth_per_ktok": {
            "value": _slope(context_points),
            "unit": "ms/1k token",
            "condition": "least squares over measured context tokens",
        },
    }


def _usage_int(usage: Any, *path: str) -> int | None:
    """Read a nested non-negative integer without guessing at absence."""
    current = usage
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, int):
        return None
    return current if current >= 0 else None


def _reported_cached_tokens(response: Mapping[str, Any]) -> int | None:
    """Read the server's own cached-token count, trying known aliases."""
    usage = response.get("usage")
    for path in (
        ("prompt_tokens_details", "cached_tokens"),
        ("cached_tokens",),
    ):
        value = _usage_int(usage, *path)
        if value is not None:
            return value
    return None


def prefix_cache_view(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate what could be cached from what the server says was cached.

    ``eligible_prefix_tokens`` is computed client side.  For turn k > 0
    the byte-identical prefix already sent is the previous turn's prompt
    plus its completion, so that sum is the ceiling for a cache hit;
    turn 0 has no prefix and contributes nothing.  This needs no server
    support and is always available.

    ``server_cache_hit_rate`` needs ``usage.prompt_tokens_details
    .cached_tokens``.  Services that do not report it leave the value
    null with an explicit reason instead of a guessed zero, and
    ``reporting_turns`` states how much of the case the server actually
    covered.  No cause is inferred from these ratios: an aggregate
    cannot distinguish a disabled cache from eviction under capacity
    pressure, so the numbers are reported and interpretation is left to
    whoever knows the deployment.
    """
    from .statistics import scalar

    by_session: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "success":
            by_session.setdefault(row["session"], []).append(row)

    prompt_total = 0
    eligible_total = 0
    cached_total = 0
    measured = 0
    reporting = 0
    for session_rows in by_session.values():
        previous_prompt: int | None = None
        previous_completion: int | None = None
        for row in sorted(session_rows, key=lambda item: item["turn"]):
            response = row.get("response") or {}
            prompt = response.get("input_tokens")
            completion = response.get("output_tokens")
            if not isinstance(prompt, int) or isinstance(prompt, bool):
                continue
            measured += 1
            prompt_total += prompt
            if (
                previous_prompt is not None
                and previous_completion is not None
            ):
                # Cap at what this turn actually sent: a truncated or
                # shorter history must not claim a larger prefix.
                eligible_total += min(
                    prompt, previous_prompt + previous_completion
                )
            cached = _reported_cached_tokens(response)
            if cached is not None:
                reporting += 1
                cached_total += min(cached, prompt)
            if (
                isinstance(completion, int)
                and not isinstance(completion, bool)
            ):
                previous_prompt, previous_completion = prompt, completion
            else:
                # Without a completion count the next prefix is unknown.
                previous_prompt = previous_completion = None

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    eligible_ratio = rate(eligible_total, prompt_total)
    server_hit = (
        rate(cached_total, prompt_total) if reporting else None
    )
    server_eligible = (
        rate(cached_total, eligible_total)
        if reporting and eligible_total
        else None
    )
    if not measured:
        reason = "no successful turns"
    elif not reporting:
        reason = (
            "service did not report usage.prompt_tokens_details"
            ".cached_tokens; only the client-side ceiling is available"
        )
    elif reporting < measured:
        reason = (
            f"only {reporting} of {measured} turns reported cached tokens"
        )
    else:
        reason = None
    return {
        "measured_turns": scalar(measured, "turn", precision=0),
        "prompt_tokens_total": scalar(
            prompt_total, "token", precision=0
        ),
        "eligible_prefix_tokens": scalar(
            eligible_total, "token", precision=0
        ),
        "eligible_prefix_ratio": {
            "value": eligible_ratio,
            "unit": "ratio",
            "condition": (
                "client-side estimate: share of prompt tokens repeating a "
                "prefix already sent in the same session; an upper bound, "
                "not a measured hit rate"
            ),
        },
        "server_cached_tokens": scalar(
            cached_total if reporting else None, "token", precision=0
        ),
        "server_cache_hit_rate": {
            "value": server_hit,
            "unit": "ratio",
            "condition": (
                "server-reported cached_tokens / prompt_tokens over the "
                "reporting turns, turn 0 included in the denominator"
            ),
        },
        "server_cache_eligible_rate": {
            "value": server_eligible,
            "unit": "ratio",
            "condition": (
                "server-reported cached_tokens / client-side eligible "
                "prefix; numerator and denominator come from different "
                "sources, so read it alongside reporting_turns"
            ),
        },
        "reporting_turns": scalar(reporting, "turn", precision=0),
        "reason": reason,
    }


def pressure_view(
    rows: Sequence[Mapping[str, Any]], duration_seconds: float
) -> dict[str, Any]:
    """Prefill work implied by concurrent sessions.

    Read together with a device exporter this shows whether aggregate
    context growth, not single-request latency, is what limits the
    deployment.
    """
    from .statistics import scalar, summarize

    contexts = [
        float(response["input_tokens"])
        for response in _successful(rows)
        if isinstance(response.get("input_tokens"), int)
        and not isinstance(response.get("input_tokens"), bool)
    ]
    total = math.fsum(contexts)
    return {
        "turn_context_tokens": summarize(contexts, "token"),
        "total_prefill_tokens": scalar(total, "token", precision=0),
        "prefill_token_rate": (
            scalar(total / duration_seconds, "token/s")
            if duration_seconds > 0 and contexts
            else scalar(None, "token/s")
        ),
        "peak_turn_context_tokens": (
            scalar(max(contexts), "token", precision=0) if contexts
            else scalar(None, "token")
        ),
    }


def session_view(
    samples: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    duration_seconds: float,
) -> dict[str, Any]:
    """Session-level outcomes, distinct from per-turn request stats."""
    from .statistics import scalar, summarize

    facts = [
        sample.get("facts")
        for sample in samples
        if isinstance(sample.get("facts"), Mapping)
    ]
    latencies = [
        float(item["session_latency_ms"])
        for item in facts
        if isinstance(item.get("session_latency_ms"), (int, float))
        and not isinstance(item.get("session_latency_ms"), bool)
    ]
    think_times = [
        float(item["think_time_total_ms"])
        for item in facts
        if isinstance(item.get("think_time_total_ms"), (int, float))
        and not isinstance(item.get("think_time_total_ms"), bool)
    ]
    completed = [
        item["completed_turns"]
        for item in facts
        if isinstance(item.get("completed_turns"), int)
        and not isinstance(item.get("completed_turns"), bool)
    ]
    planned = [
        item["planned_turns"]
        for item in facts
        if isinstance(item.get("planned_turns"), int)
        and not isinstance(item.get("planned_turns"), bool)
    ]
    not_started = sum(
        item["not_started_turns"]
        for item in facts
        if isinstance(item.get("not_started_turns"), int)
        and not isinstance(item.get("not_started_turns"), bool)
    )
    sent_failed = sum(
        item["sent_failed_turns"]
        for item in facts
        if isinstance(item.get("sent_failed_turns"), int)
        and not isinstance(item.get("sent_failed_turns"), bool)
    )
    reasons: dict[str, int] = {}
    for item in facts:
        reason = item.get("abort_reason")
        if isinstance(reason, str) and reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    finished = sum(1 for item in facts if item.get("status") == "success")
    return {
        "session_latency": summarize(latencies, "ms"),
        "think_time_total": summarize(think_times, "ms"),
        "completed_turns": summarize(
            [float(value) for value in completed], "turn"
        ),
        "planned_turns": summarize(
            [float(value) for value in planned], "turn"
        ),
        # Later turns only contain sessions that survived, so the counts
        # have to be visible or a shrinking sample reads as a speedup.
        # "Not started" excludes aborted turns whose request was sent;
        # those are already counted in the error-rate denominator.
        "not_started_turns": scalar(not_started, "turn", precision=0),
        "sent_failed_turns": scalar(sent_failed, "turn", precision=0),
        "sessions_finished": scalar(finished, "session", precision=0),
        "sessions_per_second": (
            scalar(finished / duration_seconds, "session/s")
            if duration_seconds > 0
            else scalar(None, "session/s")
        ),
        "abort_reasons": reasons,
        "session_coverage": {
            state: coverage.get(state)
            for state in (
                "complete",
                "failed",
                "incomplete",
                "not_started",
                "selected",
            )
        },
    }


def session_views(
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Return additional views only when formal requests contain turns."""
    duration = facts["window"]["duration_s"]
    parameters = facts["parameters"]
    rows = turn_rows(facts["requests"])
    if not rows:
        return {}
    responses = [row["response"] for row in rows]
    values: dict[str, Any] = {}
    values["multi_max_turns"] = parameters.get("multi_max_turns")
    values["multi_num_sessions"] = parameters.get("multi_num_sessions")
    values["turn_view"] = turn_view(rows)
    values["caching_view"] = caching_view(rows)
    values["caching_view"]["prefix_cache"] = prefix_cache_view(rows)
    values["pressure_view"] = pressure_view(rows, duration)
    values["session_view"] = session_view(
        facts["samples"], facts["coverage"], duration
    )
    values["lengths"] = length_summary(responses)
    return values
