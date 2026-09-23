"""Explicit field projections, separate from complete-record loading."""

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def select_columns(
    records: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Return selected fields without changing the source records."""
    return [
        {key: value for key, value in row.items() if key in columns}
        for row in records
    ]


def extract_prompt(
    record: Mapping[str, Any], field: str = "prompt", *, strip: bool = True
) -> str:
    """Read a named field or the first nonempty human ShareGPT turn."""
    if isinstance(record.get("conversations"), list):
        for turn in record["conversations"]:
            if not isinstance(turn, Mapping):
                raise ValueError("conversation turn must be an object")
            value = turn.get("value")
            if turn.get("from") == "human" and isinstance(value, str):
                if value.strip():
                    return value.strip() if strip else value
        raise ValueError("sample has no non-empty human prompt")
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sample has no non-empty prompt field: {field}")
    return value.strip() if strip else value


def extract_prompts(
    records: Sequence[Mapping[str, Any]],
    field: str = "prompt",
    *,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[str]:
    """Project valid prompts, then shuffle and limit the whole population.

    Unlike loading, this explicit projection can omit records without a
    usable prompt. Source records remain available for other consumers.
    """
    prompts = []
    for record in records:
        try:
            prompts.append(extract_prompt(record, field))
        except ValueError:
            continue
    if not prompts:
        raise ValueError(f"no valid prompts found (field: {field!r})")
    if shuffle:
        random.Random(seed).shuffle(prompts)
    if max_samples is not None:
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")
        prompts = prompts[:max_samples]
    return prompts


def extract_traces(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project valid timing and token fields without discarding originals."""
    traces = []
    for record in records:
        offset = record.get("offset_s")
        tokens = [record.get("input_tokens"), record.get("output_tokens")]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(offset)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in tokens
            )
        ):
            continue
        traces.append(
            {
                "offset_s": offset,
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
            }
        )
    if not traces:
        raise ValueError("no valid trace records found")
    return traces
