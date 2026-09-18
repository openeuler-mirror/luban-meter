"""Unified dataset loading for benchmarking modules.

This module provides a single entry point for loading datasets from
local files.  It supports JSON, JSONL, CSV, ShareGPT and BurstGPT
formats, plus path resolution, record sampling and field extraction.

Format-specific loaders return ``list[dict[str, Any]]`` so that all
structured information is preserved.  Callers then use
:func:`extract_prompts` or :func:`extract_traces` to obtain the
exact fields they need.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Bundled data roots for relative-path fallback.
_MODEL_SERVICE_QUALITY_ROOT = (
    Path(__file__).resolve().parent.parent
    / "model_service_quality"
)
_GENERATION_PERFORMANCE_ROOT = (
    Path(__file__).resolve().parent.parent
    / "generation_performance"
)


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a dataset path.

    Absolute paths and existing CWD-relative paths are used
    verbatim. Other relative paths fall back to the bundled
    package data directories so default configs resolve stably
    regardless of the install layout.
    """
    file = Path(path)
    if file.is_absolute() or file.is_file():
        return file
    for root in (
        _MODEL_SERVICE_QUALITY_ROOT,
        _GENERATION_PERFORMANCE_ROOT,
    ):
        packaged = root / file
        if packaged.is_file():
            return packaged
    return file


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file (one JSON object per
    line)."""
    file = resolve_data_path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"dataset file not found: {file}"
        )
    records: list[dict[str, Any]] = []
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise TypeError(
                    f"jsonl line must be an object in {file}"
                )
            records.append(item)
    return records


def load_json(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSON file.

    Supports:
    - JSON array of objects
    - JSON object with a nested array under ``data``, ``rows``,
      ``examples`` or ``questions``
    - ShareGPT conversations array
    """
    file = resolve_data_path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"dataset file not found: {file}"
        )
    text = file.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return [
            dict(item) for item in data
            if isinstance(item, dict)
        ]
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return [
                    dict(item)
                    for item in value
                    if isinstance(item, dict)
                ]
        raise ValueError(
            f"unsupported json structure in {file}"
        )
    raise ValueError(
        f"unsupported dataset content in {file}"
    )


def load_csv(
    path: str | Path,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load records from a CSV file.

    Each row becomes a dict keyed by the CSV header.  If *columns*
    is given, only those columns are kept.
    """
    file = resolve_data_path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"dataset file not found: {file}"
        )
    records: list[dict[str, Any]] = []
    with file.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if columns is not None:
                row = {
                    k: v
                    for k, v in row.items()
                    if k in columns
                }
            records.append(dict(row))
    return records


def load_sharegpt(path: str | Path) -> list[dict[str, Any]]:
    """Load conversations from a ShareGPT JSON file.

    Each conversation has
    ``{"conversations": [{"from": "human", "value": ...}, ...]}``.
    Returns the full conversation dicts so callers can extract
    prompts or build multi-turn dialogues.
    """
    records = load_json(path)
    conversations = [
        record
        for record in records
        if isinstance(record.get("conversations"), list)
    ]
    if not conversations:
        raise ValueError(
            "no valid ShareGPT conversations found in dataset"
        )
    return conversations


def load_burstgpt_trace(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load BurstGPT trace from a CSV file.

    Each record has ``timestamp``, ``input_token`` and
    ``output_token`` columns.  Returns dicts with keys
    ``offset_s``, ``input_tokens`` and ``output_tokens``.
    """
    records = load_csv(path)
    traces: list[dict[str, Any]] = []
    for row in records:
        try:
            ts = float(row.get("timestamp") or 0)
            in_tok = int(row.get("input_token") or 0)
            out_tok = int(row.get("output_token") or 0)
        except (ValueError, TypeError):
            continue
        if in_tok <= 0 or out_tok <= 0:
            continue
        traces.append(
            {
                "offset_s": ts,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        )
    if not traces:
        raise ValueError(
            "no valid trace records found in dataset"
        )
    return traces


# Registry of format-specific loaders, used by ``load_records``.
_FMT_LOADERS: dict[str, Any] = {
    "jsonl": load_jsonl,
    "json": load_json,
    "csv": load_csv,
    "sharegpt": load_sharegpt,
    "burstgpt": load_burstgpt_trace,
}

# File suffix to format name for format inference.
_SUFFIX_FMT: dict[str, str] = {
    ".jsonl": "jsonl",
    ".json": "json",
    ".csv": "csv",
}


def load_records(
    path: str | Path,
    fmt: str | None = None,
) -> list[dict[str, Any]]:
    """Load dataset records from a local file.

    When *fmt* is ``None`` the format is inferred from the file
    suffix (``.jsonl``, ``.json``, ``.csv``).  Explicit *fmt*
    values override the suffix and also select format-specific
    loaders such as ``"sharegpt"`` and ``"burstgpt"``.
    """
    if fmt is not None:
        loader = _FMT_LOADERS.get(fmt)
        if loader is None:
            raise ValueError(
                f"unsupported dataset format: {fmt!r}"
            )
        return loader(path)

    file = resolve_data_path(path)
    suffix = file.suffix.lower()
    suffix_fmt = _SUFFIX_FMT.get(suffix)
    if suffix_fmt is None:
        raise ValueError(
            f"cannot infer format from suffix {suffix!r}; "
            f"pass fmt= explicitly"
        )
    return _FMT_LOADERS[suffix_fmt](path)


def select_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample records.

    Shuffles then truncates when asked.
    """
    selected = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        selected = selected[:max_samples]
    return selected


def select_few_shot(
    records: list[dict[str, Any]],
    key: str,
    value: Any,
    count: int,
) -> list[dict[str, Any]]:
    """Pick the first *count* records whose *key* field equals
    *value*.
    """
    if count <= 0:
        return []
    matched = [
        record
        for record in records
        if record.get(key) == value
    ]
    return matched[:count]


def extract_prompts(
    records: Sequence[Mapping[str, Any]],
    field: str = "prompt",
    *,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[str]:
    """Extract prompt strings from records.

    For ShareGPT records (with ``conversations``), extracts the
    first human turn.  For other records, reads the named
    *field*.
    """
    prompts: list[str] = []
    for record in records:
        conversations = record.get("conversations")
        if isinstance(conversations, list):
            for turn in conversations:
                if not isinstance(turn, Mapping):
                    continue
                if turn.get("from") == "human":
                    value = turn.get("value")
                    if (
                        isinstance(value, str)
                        and value.strip()
                    ):
                        prompts.append(value.strip())
                    break
        else:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                prompts.append(value.strip())
        if (
            max_samples is not None
            and len(prompts) >= max_samples
        ):
            break

    if not prompts:
        raise ValueError(
            f"no valid prompts found (field: {field!r})"
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        prompts = prompts[:max_samples]

    return prompts


def extract_traces(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract trace records (offset_s, input_tokens,
    output_tokens)."""
    traces = [
        {
            "offset_s": record["offset_s"],
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
        }
        for record in records
        if isinstance(record.get("offset_s"), (int, float))
        and not isinstance(
            record.get("offset_s"), bool
        )
        and isinstance(record.get("input_tokens"), int)
        and not isinstance(
            record.get("input_tokens"), bool
        )
        and isinstance(record.get("output_tokens"), int)
        and not isinstance(
            record.get("output_tokens"), bool
        )
    ]
    if not traces:
        raise ValueError("no valid trace records found")
    return traces
