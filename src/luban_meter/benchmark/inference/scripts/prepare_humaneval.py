"""Convert the official HumanEval jsonl release into a local dataset.

Usage:
    python src/luban_meter/benchmark/inference/scripts/prepare_humaneval.py \
        --source /path/to/HumanEval.jsonl.gz \
        --out data/humaneval/HumanEval.jsonl

This is an offline preparation tool. The benchmark never downloads datasets.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import TextIO

REQUIRED_FIELDS = ("task_id", "prompt", "canonical_solution", "test", "entry_point")


def resolve_source(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("HumanEval.jsonl.gz", "HumanEval.jsonl"):
            candidate = path / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"HumanEval source not found: {path}")


def open_source(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_humaneval(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    task_ids: set[str] = set()
    with open_source(resolve_source(path)) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"line {line_number}: record must be an object")
            record: dict[str, str] = {}
            for field in REQUIRED_FIELDS:
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"line {line_number}: {field} must be a non-empty string"
                    )
                record[field] = value
            if record["task_id"] in task_ids:
                raise ValueError(f"duplicate task_id: {record['task_id']}")
            task_ids.add(record["task_id"])
            records.append(record)
    if not records:
        raise ValueError("HumanEval source contains no records")
    return records


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/humaneval/HumanEval.jsonl"),
    )
    args = parser.parse_args()
    records = load_humaneval(args.source)
    write_jsonl(records, args.out)
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
