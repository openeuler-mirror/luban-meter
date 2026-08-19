"""Validate token counts reported by the OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def token_usage(value: Any) -> tuple[int, int] | None:
    """Return prompt and completion token counts from a usage object."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("stream usage must be an object")

    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    for name, count in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
    ):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"stream usage {name} must be a non-negative integer")
    return prompt_tokens, completion_tokens
