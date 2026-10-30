"""Parse one complete OpenAI-compatible streaming completion response."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from luban_meter.runtime.thinking import reasoning_text, reasoning_token_count

from .token_counting import (
    token_usage,
)


@dataclass(frozen=True)
class StreamObservation:
    generated_text: str
    event_times: tuple[float, ...]
    input_tokens: int | None
    output_tokens: int | None
    reasoning_text: str = ""
    reasoning_event_times: tuple[float, ...] = ()
    content_event_times: tuple[float, ...] = ()
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    usage: Mapping | None = None


def collect_completion_stream(
    stream: Iterable[bytes],
    clock: Callable[[], float] = time.perf_counter,
) -> StreamObservation:
    """Consume an SSE stream and retain only valid, non-empty output
    events.
    """
    event_times: list[float] = []
    generated_parts: list[str] = []
    usage: tuple[int, int] | None = None

    for raw_line in stream:
        line = raw_line.decode("utf-8").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break

        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise TypeError("stream event must contain a JSON object")

        parsed_usage = token_usage(payload.get("usage"))
        if parsed_usage is not None:
            usage = parsed_usage

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, Mapping):
            continue
        text = choice.get("text")
        if isinstance(text, str) and text:
            event_times.append(clock())
            generated_parts.append(text)

    if not event_times:
        raise RuntimeError(
            "stream ended before the first non-empty generated output"
        )
    input_tokens, output_tokens = usage if usage is not None else (None, None)
    if output_tokens is not None and output_tokens <= 0:
        raise RuntimeError("stream reported no output tokens")
    return StreamObservation(
        generated_text="".join(generated_parts),
        event_times=tuple(event_times),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def collect_chat_stream(
    stream: Iterable[bytes],
    clock: Callable[[], float] = time.perf_counter,
) -> StreamObservation:
    """Collect reasoning and content, counting each output event once.

    Thinking aliases are kept separate from final content. A response
    with reasoning alone is a valid performance observation.
    """
    event_times: list[float] = []
    content_times: list[float] = []
    reasoning_times: list[float] = []
    generated_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: tuple[int, int] | None = None
    raw_usage = None
    finish_reason = None

    for raw_line in stream:
        line = raw_line.decode("utf-8").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break

        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise TypeError("stream event must contain a JSON object")

        parsed_usage = token_usage(payload.get("usage"))
        if parsed_usage is not None:
            usage = parsed_usage
            raw_usage = payload["usage"]

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, Mapping):
            continue
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue
        thought = reasoning_text(delta)
        content = delta.get("content")
        content = content if isinstance(content, str) else ""
        if not thought and not content:
            continue
        timestamp = clock()
        event_times.append(timestamp)
        if thought:
            reasoning_parts.append(thought)
            reasoning_times.append(timestamp)
        if content:
            generated_parts.append(content)
            content_times.append(timestamp)

    if not event_times:
        raise RuntimeError(
            "stream ended before the first non-empty generated output"
        )
    input_tokens, output_tokens = usage if usage is not None else (None, None)
    if output_tokens is not None and output_tokens <= 0:
        raise RuntimeError("stream reported no output tokens")
    return StreamObservation(
        generated_text="".join(generated_parts),
        event_times=tuple(event_times),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_text="".join(reasoning_parts),
        reasoning_event_times=tuple(reasoning_times),
        content_event_times=tuple(content_times),
        reasoning_tokens=reasoning_token_count(raw_usage),
        finish_reason=finish_reason,
        usage=raw_usage,
    )
