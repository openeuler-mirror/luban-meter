"""Convert local official dev-v2.0.json to SQuAD 2.0 JSONL.

Run after installing luban-meter (pip install -e .):
python -m \
luban_meter.benchmarking.model_service_quality.scripts.prepare_squad \
    --source /path/to/dev-v2.0.json --out data/squad/validation.jsonl
No dataset download occurs. Original question IDs are preserved.
"""

import argparse
import json
from pathlib import Path

from luban_meter.benchmarking.model_service_quality.squad.dataset import (
    load_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("data/squad/validation.jsonl")
    )
    args = parser.parse_args()
    if args.source.resolve() == args.out.resolve():
        parser.error("source and output must be different files")
    records = load_samples(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} samples -> {args.out}")


if __name__ == "__main__":
    main()
