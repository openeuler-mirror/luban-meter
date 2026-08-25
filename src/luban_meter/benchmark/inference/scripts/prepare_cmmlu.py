"""Convert official CMMLU CSV files into luban-meter jsonl datasets.

Usage:
    python src/luban_meter/benchmark/inference/scripts/prepare_cmmlu.py \
        --source /path/to/cmmlu [--out data/cmmlu]

Expected layout of the official CMMLU release (e.g. cmmlu_v1_0_1):
    <source>/test/<subject>.csv    # evaluation split -> val.jsonl
    <source>/dev/<subject>.csv     # 5-shot examples    -> dev.jsonl
    CSV columns: Question,A,B,C,D,Answer (case-insensitive)

This is a one-off offline tool. Benchmarks only read local jsonl files and
never download data at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED_COLUMNS = ("question", "a", "b", "c", "d", "answer")
CHOICE_COLUMNS = ("a", "b", "c", "d")


def normalize_row(row: dict) -> dict:
    normalized = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized[key.strip().lower()] = value
    return normalized


def convert_split(source_dir: Path, out_file: Path) -> int:
    records = []
    for csv_file in sorted(source_dir.glob("*.csv")):
        subject = csv_file.stem
        with csv_file.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [
                (name or "").strip().lower() for name in (reader.fieldnames or [])
            ]
            missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
            if missing:
                raise ValueError(f"{csv_file}: missing columns {missing}")
            for index, raw_row in enumerate(reader):
                row = normalize_row(raw_row)
                answer = (row.get("answer") or "").strip().upper()
                if answer not in ("A", "B", "C", "D"):
                    raise ValueError(
                        f"{csv_file} row {index}: invalid answer {answer!r}"
                    )
                records.append(
                    {
                        "id": f"{subject}-{index}",
                        "subject": subject,
                        "question": (row.get("question") or "").strip(),
                        "choices": [
                            (row.get(column) or "").strip()
                            for column in CHOICE_COLUMNS
                        ],
                        "answer": answer,
                    }
                )
    if not records:
        raise ValueError(f"no csv files found in {source_dir}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory containing test/ and dev/ subject CSV files",
    )
    parser.add_argument("--out", type=Path, default=Path("data/cmmlu"))
    args = parser.parse_args()
    # Output split -> official CMMLU source split. The official evaluation
    # split is named "test"; luban-meter stores it as val.jsonl.
    split_map = {"val": "test", "dev": "dev"}
    for output_split, source_split in split_map.items():
        split_dir = args.source / source_split
        if not split_dir.is_dir():
            raise SystemExit(f"expected directory not found: {split_dir}")
        count = convert_split(split_dir, args.out / f"{output_split}.jsonl")
        print(
            f"wrote {count} records -> {args.out / f'{output_split}.jsonl'}"
            f" (source split '{source_split}')"
        )


if __name__ == "__main__":
    main()
