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


# ShareGPT speaker labels mapped to OpenAI chat roles.
CONVERSATION_ROLES = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}


def extract_conversation(
    record: Mapping[str, Any], *, strip: bool = False
) -> list[dict[str, str]]:
    """Project a conversation record into role/content turns.

    Turns with an unknown speaker or a missing text value are
    skipped rather than rejected, so partially malformed datasets
    stay usable.  Structural problems (``conversations`` absent or
    a turn that is not an object) are errors.
    """
    turns = record.get("conversations")
    if not isinstance(turns, list):
        raise ValueError("sample has no conversations list")
    projected: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("conversation turn must be an object")
        role = CONVERSATION_ROLES.get(turn.get("from"))
        value = turn.get("value")
        if role is None or not isinstance(value, str):
            continue
        if not value.strip():
            continue
        projected.append(
            {
                "role": role,
                "content": value.strip() if strip else value,
            }
        )
    return projected


def split_turns(
    projected: Sequence[Mapping[str, Any]]
) -> list[list[dict[str, str]]]:
    """Group projected messages into per-turn increments.

    Reference assistant messages delimit turns and are dropped: a
    multi-turn benchmark regenerates them with the model under test.
    Every other role stays with the turn it precedes, so a dataset
    provided system instruction survives instead of being rewritten.

    ``[system, user1, assistant_ref, user2]`` becomes
    ``[[system, user1], [user2]]``.  An increment holding no user
    message is merged into the following one, since it cannot start a
    request on its own; a trailing one is dropped.
    """
    increments: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for turn in projected:
        if turn["role"] == "assistant":
            if current:
                increments.append(current)
                current = []
            continue
        current.append(dict(turn))
    if current:
        increments.append(current)

    turns: list[list[dict[str, str]]] = []
    pending: list[dict[str, str]] = []
    for increment in increments:
        pending.extend(increment)
        if any(message["role"] == "user" for message in pending):
            turns.append(pending)
            pending = []
    return turns


def summarize_turn_counts(counts: Sequence[int]) -> dict[str, Any]:
    """Describe a turn-count distribution without interpolating.

    Recording the source distribution alongside the selected one is
    what makes sampling bias visible in results, whether sessions were
    truncated to a maximum depth or filtered some other way.
    """
    ordered = sorted(counts)
    total = len(ordered)
    if not total:
        return {
            "conversations": 0,
            "mean": 0.0,
            "min": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }

    def at(ratio: float) -> int:
        return ordered[min(total - 1, int(total * ratio))]

    return {
        "conversations": total,
        "mean": round(sum(ordered) / total, 2),
        "min": ordered[0],
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


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
