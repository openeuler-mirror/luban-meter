"""Load datasets and construct prompts for generation performance.

Re-exports the unified loaders from
:mod:`luban_meter.benchmarking.common.dataset` and keeps the
generate-specific helpers (``tokenize_seed_prompt``,
``exact_prompt_token_ids``) that talk to a running inference
server.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from luban_meter.benchmarking.common.dataset import (
    extract_prompts,
    load_burstgpt_trace,
    load_csv,
    load_json,
    load_jsonl,
    load_records,
    load_sharegpt,
    resolve_data_path,
    select_records,
)
from luban_meter.benchmarking.generation_performance.common.http_client import (
    post_json,
)


def tokenize_seed_prompt(
    service_url: str,
    served_model_name: str,
    seed_prompt: str,
    api_key: str,
    timeout: float,
) -> tuple[list[int], int]:
    """Tokenize a seed prompt and return (token_ids,
    max_model_len)."""
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
            not isinstance(token, int)
            or isinstance(token, bool)
            for token in tokens
        )
        or count != len(tokens)
    ):
        raise RuntimeError(
            "/tokenize returned invalid token IDs or count"
        )
    if (
        not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
        or max_model_len <= 0
    ):
        raise RuntimeError(
            "/tokenize returned an invalid max_model_len"
        )
    return tokens, max_model_len


def exact_prompt_token_ids(
    seed_tokens: Sequence[int],
    length: int,
    request_index: int,
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

    Extracts the first human message from each conversation.
    """
    records = load_sharegpt(dataset_path)
    return extract_prompts(
        records,
        max_samples=num_prompts,
        shuffle=True,
        seed=seed,
    )


def load_jsonl_prompts(
    dataset_path: str,
    num_prompts: int,
    prompt_field: str = "prompt",
    seed: int = 0,
) -> list[str]:
    """Load prompts from a custom JSONL file."""
    records = load_jsonl(dataset_path)
    return extract_prompts(
        records,
        field=prompt_field,
        max_samples=num_prompts,
        shuffle=True,
        seed=seed,
    )


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
        return load_sharegpt_prompts(
            dataset_path, num_prompts, seed
        )
    if dataset_format == "jsonl":
        return load_jsonl_prompts(
            dataset_path, num_prompts, prompt_field, seed
        )
    raise ValueError(
        f"unsupported dataset_format: "
        f"{dataset_format!r}. Use 'sharegpt' or 'jsonl'."
    )


__all__ = [
    "exact_prompt_token_ids",
    "extract_prompts",
    "load_burstgpt_trace",
    "load_csv",
    "load_json",
    "load_jsonl",
    "load_jsonl_prompts",
    "load_prompts",
    "load_records",
    "load_sharegpt",
    "load_sharegpt_prompts",
    "resolve_data_path",
    "select_records",
    "tokenize_seed_prompt",
]
