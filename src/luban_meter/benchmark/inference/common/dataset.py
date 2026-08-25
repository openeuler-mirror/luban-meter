"""Local dataset loading for inference benchmarks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load dataset records from a local json or jsonl file."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"dataset file not found: {file}")
    if file.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        # Iterate physical lines (split on \n only). Do not use splitlines():
        # official GSM8K text contains U+2028/U+2029 which splitlines() would
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
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
        raise ValueError(f"unsupported json structure in {file}")
    raise ValueError(f"unsupported dataset content in {file}")


def select_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample records; shuffle then truncate when asked."""
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
    """Pick the first `count` records whose `key` field equals `value`."""
    if count <= 0:
        return []
    matched = [record for record in records if record.get(key) == value]
    return matched[:count]
