"""Thinking response facts shared by HTTP benchmarks."""

from collections.abc import Mapping


def reasoning_text(message):
    """Read either service alias without silently duplicating reasoning."""
    values = []
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string or null")
        values.append(value)
    if len(set(values)) > 1:
        raise ValueError("conflicting reasoning and reasoning_content fields")
    return values[0] if values else ""


def reasoning_token_count(usage):
    """Preserve reported reasoning counts without estimating missing ones."""
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("completion_tokens_details")
    if details is None:
        return None
    if not isinstance(details, Mapping):
        raise ValueError("completion_tokens_details must be an object")
    count = details.get("reasoning_tokens")
    if count is not None and (
        not isinstance(count, int) or isinstance(count, bool) or count < 0
    ):
        raise ValueError("reasoning_tokens must be a non-negative integer")
    return count


def response_facts(response):
    """Copy chat diagnostics into quality facts without changing scoring."""
    return {
        name: response[name]
        for name in (
            "reasoning_text", "reasoning_tokens", "has_content",
            "finish_reason", "usage",
        )
        if name in response
    }
