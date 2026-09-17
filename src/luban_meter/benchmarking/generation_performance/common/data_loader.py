"""Load datasets and construct prompts for generation performance.

Supports three dataset formats:
- ``sharegpt``: ShareGPT JSON with ``conversations`` array
- ``jsonl``: Custom JSONL with configurable prompt field
- ``burstgpt``: BurstGPT CSV trace (timestamp, input/output lengths)
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .http_client import post_json


def tokenize_seed_prompt(
    service_url: str,
    served_model_name: str,
    seed_prompt: str,
    api_key: str,
    timeout: float,
) -> tuple[list[int], int]:
    """Tokenize a seed prompt and return (token_ids, max_model_len)."""
    response = post_json(
        f"{service_url.rstrip('/')}/tokenize",
        {
            "model": served_model_name,
            "prompt": seed_prompt,
            "add_special_tokens": False,
        },
        api_key,
        timeout,
    )
    tokens = response.get("tokens")
    count = response.get("count")
    max_model_len = response.get("max_model_len")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(
            not isinstance(token, int) or isinstance(token, bool)
            for token in tokens
        )
        or count != len(tokens)
    ):
        raise RuntimeError("/tokenize returned invalid token IDs or count")
    if (
        not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
        or max_model_len <= 0
    ):
        raise RuntimeError("/tokenize returned an invalid max_model_len")
    return tokens, max_model_len


def exact_prompt_token_ids(
    seed_tokens: Sequence[int], length: int, request_index: int
) -> list[int]:
    """Build an exact-length prompt by rotating seed tokens."""
    offset = request_index % len(seed_tokens)
    rotated = [*seed_tokens[offset:], *seed_tokens[:offset]]
    repeats = math.ceil(length / len(rotated))
    return (rotated * repeats)[:length]


def load_sharegpt_prompts(
    dataset_path: str, num_prompts: int, seed: int = 0
) -> list[str]:
    """Load human prompts from a ShareGPT JSON file.

    Each conversation has
    ``{"conversations": [{"from": "human", "value": ...}, ...]}``.
    We extract the **first** human message from each conversation to
    serve as a single-turn prompt.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(
            "ShareGPT dataset must be a JSON array of conversations"
        )

    rng = random.Random(seed)
    rng.shuffle(data)

    prompts: list[str] = []
    for conversation in data:
        if not isinstance(conversation, Mapping):
            continue
        conversations = conversation.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        for turn in conversations:
            if not isinstance(turn, Mapping):
                continue
            if turn.get("from") == "human":
                value = turn.get("value")
                if isinstance(value, str) and value.strip():
                    prompts.append(value.strip())
                    break
        if len(prompts) >= num_prompts:
            break

    if not prompts:
        raise ValueError("no valid prompts found in the dataset")
    return prompts


def load_jsonl_prompts(
    dataset_path: str,
    num_prompts: int,
    prompt_field: str = "prompt",
    seed: int = 0,
) -> list[str]:
    """Load prompts from a custom JSONL file.

    Each line is a JSON object. The field named by ``prompt_field``
    is extracted as the prompt text. Example::

        {"prompt": "What is the capital of India?"}
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    records: list[str] = []
    with path.open(encoding="utf-8") as dataset_stream:
        for line in dataset_stream:
            line = line.strip()
            if not line:
                continue
            prompt_record = json.loads(line)
            if not isinstance(prompt_record, Mapping):
                continue
            value = prompt_record.get(prompt_field)
            if isinstance(value, str) and value.strip():
                records.append(value.strip())
            if len(records) >= num_prompts:
                break

    if not records:
        raise ValueError(
            f"no valid prompts found in {dataset_path} "
            f"(field: {prompt_field!r})"
        )
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:num_prompts]


def load_burstgpt_trace(
    dataset_path: str,
    num_prompts: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Load BurstGPT trace from a CSV file.

    Each record has: ``timestamp``, ``input_token``,
    ``output_token``. Returns a list of dicts with keys
    ``offset_s``, ``input_tokens``, ``output_tokens``.

    The BurstGPT dataset is a production trace from Azure and can be
    downloaded from::

        wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    traces: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as dataset_stream:
        reader = csv.DictReader(dataset_stream)
        for row in reader:
            try:
                timestamp_seconds = float(row.get("timestamp", 0))
                input_token_count = int(row.get("input_token", 0))
                output_token_count = int(row.get("output_token", 0))
            except (ValueError, TypeError):
                continue
            if input_token_count <= 0 or output_token_count <= 0:
                continue
            traces.append(
                {
                    "offset_s": timestamp_seconds,
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                }
            )
            if len(traces) >= num_prompts:
                break

    if not traces:
        raise ValueError(f"no valid trace records found in {dataset_path}")
    return traces[:num_prompts]


def load_prompts(
    dataset_path: str,
    num_prompts: int,
    dataset_format: str = "sharegpt",
    prompt_field: str = "prompt",
    seed: int = 0,
) -> list[str]:
    """Load prompts from a dataset file.

    Supported formats:
    - ``sharegpt``: ShareGPT JSON with conversations array
    - ``jsonl``: Custom JSONL with configurable prompt field
    """
    if dataset_format == "sharegpt":
        return load_sharegpt_prompts(dataset_path, num_prompts, seed)
    if dataset_format == "jsonl":
        return load_jsonl_prompts(
            dataset_path, num_prompts, prompt_field, seed
        )
    raise ValueError(
        f"unsupported dataset_format: {dataset_format!r}. "
        f"Use 'sharegpt' or 'jsonl'."
    )
