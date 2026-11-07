"""Validate service-reported prefix-cache token counts without estimating."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class CacheUsage:
    """Validated counts from one usage object, independent of HTTP status.

    Only a valid pair exposes cached_tokens. The original usage remains
    the evidence for missing or invalid fields. Zero prompt and cached
    counts form a valid pair, but do not define a hit-rate denominator.
    """

    state: Literal["valid", "missing", "invalid"]
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    reason: str | None = None


def _is_count(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def parse_cache_usage(usage: Any) -> CacheUsage:
    """Parse one usage object; never fill, clamp or infer cached counts.

    Missing fields include absent keys and null values. Malformed fields
    take precedence over missing fields. Only the nested OpenAI field is
    accepted; top-level cached_tokens is not an alias. Completion counts
    and request success are outside this parser's responsibility.
    """
    if usage is None:
        return CacheUsage("missing", reason="missing_usage")
    if not isinstance(usage, Mapping):
        return CacheUsage("invalid", reason="invalid_usage")

    prompt = usage.get("prompt_tokens")
    if prompt is not None and not _is_count(prompt):
        return CacheUsage("invalid", reason="invalid_prompt_tokens")

    details = usage.get("prompt_tokens_details")
    if details is not None and not isinstance(details, Mapping):
        return CacheUsage(
            "invalid", prompt_tokens=prompt,
            reason="invalid_prompt_tokens_details",
        )
    cached = details.get("cached_tokens") if details is not None else None
    if cached is not None and not _is_count(cached):
        return CacheUsage(
            "invalid", prompt_tokens=prompt, reason="invalid_cached_tokens",
        )

    if prompt is None:
        return CacheUsage("missing", reason="missing_prompt_tokens")
    if details is None:
        return CacheUsage(
            "missing", prompt_tokens=prompt,
            reason="missing_prompt_tokens_details",
        )
    if cached is None:
        return CacheUsage(
            "missing", prompt_tokens=prompt, reason="missing_cached_tokens",
        )
    if cached > prompt:
        return CacheUsage(
            "invalid", prompt_tokens=prompt,
            reason="cached_tokens_exceed_prompt_tokens",
        )
    return CacheUsage("valid", prompt_tokens=prompt, cached_tokens=cached)
