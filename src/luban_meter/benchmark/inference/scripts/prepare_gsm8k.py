"""Convert official GSM8K jsonl files into luban-meter jsonl datasets.

Usage:
    python src/luban_meter/benchmark/inference/scripts/prepare_gsm8k.py \
        --source /path/to/gsm8k [--out data/gsm8k]
        [--few-shot 8] [--few-shot-file /path/to/curated.jsonl]

Expected layout of the official GSM8K release:
    <source>/test.jsonl    # {"question": ..., "answer": "... #### N"}
    <source>/train.jsonl   # used for few-shot when --few-shot-file is omitted

Note: the canonical GSM8K 8-shot examples are hand-picked from train; pass
--few-shot-file to use a curated file, otherwise the first N train samples
are used. This is a one-off offline tool; benchmarks never download data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    # Split on \n only: GSM8K text contains U+2028/U+2029 characters that
    # str.splitlines() would treat as line breaks and corrupt the records.
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise TypeError(f"line must be an object in {path}")
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError(f"record needs string question/answer in {path}")
        records.append({"question": question, "answer": answer})
    return records


def write_jsonl(records: list[dict], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory containing official test.jsonl and train.jsonl",
    )
    parser.add_argument("--out", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--few-shot", type=int, default=8)
    parser.add_argument("--few-shot-file", type=Path, default=None)
    args = parser.parse_args()

    test_file = args.source / "test.jsonl"
    if not test_file.is_file():
        raise SystemExit(f"expected file not found: {test_file}")
    test_records = load_jsonl(test_file)
    for index, record in enumerate(test_records):
        record["id"] = f"gsm8k-test-{index}"
    write_jsonl(test_records, args.out / "test.jsonl")
    print(f"wrote {len(test_records)} records -> {args.out / 'test.jsonl'}")

    if args.few_shot > 0:
        few_shot_source = args.few_shot_file or args.source / "train.jsonl"
        if not few_shot_source.is_file():
            raise SystemExit(f"expected file not found: {few_shot_source}")
        few_shot_records = load_jsonl(few_shot_source)[: args.few_shot]
        if len(few_shot_records) < args.few_shot:
            raise SystemExit(
                f"only {len(few_shot_records)} few-shot records in {few_shot_source}"
            )
        for index, record in enumerate(few_shot_records):
            record["id"] = f"gsm8k-fs-{index}"
        write_jsonl(few_shot_records, args.out / "few_shot.jsonl")
        print(
            f"wrote {len(few_shot_records)} records -> {args.out / 'few_shot.jsonl'}"
        )


if __name__ == "__main__":
    main()
