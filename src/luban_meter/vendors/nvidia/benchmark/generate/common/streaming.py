"""Parse one complete OpenAI-compatible streaming completion response."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import BinaryIO

from luban_meter.vendors.nvidia.benchmark.generate.common.token_counting import (
    token_usage,
)


@dataclass(frozen=True)
class StreamObservation:
    generated_text: str
    event_times: tuple[float, ...]
    input_tokens: int
    output_tokens: int


def collect_completion_stream(
    stream: BinaryIO,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamObservation:
    """Consume an SSE stream and retain only valid, non-empty output events."""
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
        raise RuntimeError("stream ended before the first non-empty generated output")
    if usage is None:
        raise RuntimeError(
            "stream did not report token usage; the endpoint must support "
            "stream_options.include_usage"
        )
    input_tokens, output_tokens = usage
    if output_tokens <= 0:
        raise RuntimeError("stream reported no output tokens")
    return StreamObservation(
        generated_text="".join(generated_parts),
        event_times=tuple(event_times),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
