"""Shared local record loading for performance and quality."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Shared datasets live in the source repo, outside the Python package.
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def resolve_data_path(
    path: str | Path, *, base_dir: str | Path | None = None
) -> Path:
    """Resolve a dataset path.

    Absolute paths and existing CWD-relative paths are used
    verbatim. In a source checkout, dataset/... can also reference the
    shared repository directory. Wheel installs use user-supplied files.
    """
    file = Path(path).expanduser()
    if base_dir is not None:
        return Path(base_dir).expanduser() / file
    if file.is_absolute() or file.is_file():
        return file
    if file.parts and file.parts[0] == "dataset":
        shared = _SOURCE_ROOT / file
        if (_SOURCE_ROOT / "pyproject.toml").is_file() and shared.is_file():
            return shared
    return file


def load_records(
    path: str | Path, *, format: str | None = None
) -> list[dict[str, Any]]:
    """Load dataset records from a local json or jsonl file."""
    file = resolve_data_path(path)
    if not file.is_file():
        raise FileNotFoundError(f"dataset file not found: {file}")
    if (format or file.suffix.lower().lstrip(".")) == "jsonl":
        records: list[dict[str, Any]] = []
        # Iterate physical lines (split on \n only). Do not use
        # splitlines():
        # official GSM8K text contains U+2028/U+2029 which splitlines()
        # would
        # treat as line breaks and corrupt JSON records.
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                item = json.loads(stripped)
                if not isinstance(item, dict):
                    raise TypeError(f"jsonl line must be an object in {file}")
                records.append(item)
        return records
    text = file.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return object_records(data, file)
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return object_records(value, file)
        raise ValueError(f"unsupported json structure in {file}")
    raise ValueError(f"unsupported dataset content in {file}")


def object_records(values, file):
    if any(not isinstance(item, dict) for item in values):
        raise TypeError(f"dataset entries must be objects in {file}")
    return [dict(item) for item in values]


def select_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample records; shuffle then truncate when
    asked.
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
    records: list[dict[str, Any]], key: str, value: Any, count: int
) -> list[dict[str, Any]]:
    """Pick the first `count` records with `key` equal to `value`."""
    if count <= 0:
        return []
    matched = [record for record in records if record.get(key) == value]
    return matched[:count]
