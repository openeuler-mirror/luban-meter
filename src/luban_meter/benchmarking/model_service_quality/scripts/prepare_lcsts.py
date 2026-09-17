"""Convert LCSTS dataset into luban-meter jsonl format.

The LCSTS (Large Scale Chinese Short Text Summarization) dataset is
available on Hugging Face as suolyer/lcsts. This script downloads
or reads local copies and converts them to the format expected by
the LCSTS benchmark.

Usage::

    # Download from Hugging Face
    python -m \
        luban_meter.benchmarking.model_service_quality.scripts.prepare_lcsts \
        --out data/lcsts

    # Use local files
    python -m \
        luban_meter.benchmarking.model_service_quality.scripts.prepare_lcsts \
        --source /path/to/lcsts --out data/lcsts

    # Specify number of few-shot examples
    python -m \
        luban_meter.benchmarking.model_service_quality.scripts.prepare_lcsts \
        --out data/lcsts --few-shot 8

Source format (from Hugging Face suolyer/lcsts)::

    {"input": "instruction prefix + article", "output": "summary", "id": 0}

The input field contains a task instruction prefix followed by the
actual article content. This script strips the prefix and maps fields:

- input -> content (article text, prefix removed)
- output -> abst (reference summary)

Output: jsonl records with id, content, and abst.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The instruction prefix present in all LCSTS records
LCSTS_INSTRUCTION_PREFIX = (
    "在本任务中，您将获得一段文本，您的任务是生成该文本的摘要。"
)

# Expected dataset counts
EXPECTED_TEST_COUNT = 725
EXPECTED_TRAIN_COUNT = 10000


def strip_instruction_prefix(text: str) -> str:
    """Remove the LCSTS instruction prefix from text."""
    if text.startswith(LCSTS_INSTRUCTION_PREFIX):
        return text[len(LCSTS_INSTRUCTION_PREFIX) :]
    return text


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file and return list of records."""
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as dataset_stream:
        for line_num, line in enumerate(dataset_stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at line {line_num} in {path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"record at line {line_num} in {path} "
                    f"must be an object, got "
                    f"{type(record).__name__}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"file is empty: {path}")
    return records


def validate_record(record: dict, line_num: int, source: str) -> None:
    """Validate a single LCSTS record."""
    if "input" not in record:
        raise ValueError(
            f"record {line_num} in {source}: missing 'input' field"
        )
    if "output" not in record:
        raise ValueError(
            f"record {line_num} in {source}: missing 'output' field"
        )
    input_text = record["input"]
    output_text = record["output"]
    if not isinstance(input_text, str):
        raise ValueError(
            f"record {line_num} in {source}: 'input' must be string"
        )
    if not isinstance(output_text, str):
        raise ValueError(
            f"record {line_num} in {source}: 'output' must be string"
        )
    content = strip_instruction_prefix(input_text)
    if not content.strip():
        raise ValueError(
            f"record {line_num} in {source}: "
            f"'input' is empty after "
            f"stripping prefix"
        )
    if not output_text.strip():
        raise ValueError(f"record {line_num} in {source}: 'output' is empty")


def convert_record(record: dict, index: int, prefix: str) -> dict[str, str]:
    """Convert an LCSTS record to luban-meter format."""
    content = strip_instruction_prefix(record["input"])
    return {
        "id": f"{prefix}-{index}",
        "content": content,
        "abst": record["output"],
    }


def validate_dataset_integrity(
    records: list[dict],
    expected_count: int,
    source: str,
) -> None:
    """Validate dataset has expected number of records."""
    actual_count = len(records)
    if actual_count != expected_count:
        raise ValueError(
            f"{source}: expected {expected_count} records, got {actual_count}"
        )


def validate_all_records(records: list[dict], source: str) -> None:
    """Validate all records in a dataset."""
    for idx, record in enumerate(records, start=1):
        validate_record(record, idx, source)


def convert_records(records: list[dict], prefix: str) -> list[dict[str, str]]:
    """Convert a list of LCSTS records."""
    return [
        convert_record(record, idx, prefix)
        for idx, record in enumerate(records)
    ]


def write_jsonl(records: list[dict], out_file: Path) -> None:
    """Write records to a JSONL file."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as dataset_stream:
        for record in records:
            dataset_stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_from_huggingface(
    out_dir: Path,
) -> tuple[Path, Path]:
    """Download LCSTS dataset from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download "
            "LCSTS. Install with: "
            "pip install huggingface_hub"
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading LCSTS test.json from Hugging Face...")
    test_src = hf_hub_download(
        repo_id="suolyer/lcsts",
        filename="test.json",
        repo_type="dataset",
    )
    test_dst = out_dir / "test.json"
    test_dst.write_text(
        Path(test_src).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print("Downloading LCSTS train.json from Hugging Face...")
    train_src = hf_hub_download(
        repo_id="suolyer/lcsts",
        filename="train.json",
        repo_type="dataset",
    )
    train_dst = out_dir / "train.json"
    train_dst.write_text(
        Path(train_src).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"Downloaded to {out_dir}")
    return test_dst, train_dst


def process_split(
    source_file: Path,
    out_file: Path,
    expected_count: int,
    id_prefix: str,
) -> int:
    """Process a single split (test or train)."""
    print(f"Processing {source_file.name}...")
    records = load_jsonl(source_file)
    validate_dataset_integrity(records, expected_count, source_file.name)
    validate_all_records(records, source_file.name)
    converted = convert_records(records, id_prefix)
    write_jsonl(converted, out_file)
    print(f"  Wrote {len(converted)} records -> {out_file}")
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(argparse.RawDescriptionHelpFormatter),
    )
    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "Directory containing test.json and "
            "train.json. If not provided, downloads "
            "from Hugging Face."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/lcsts"),
        help="Output directory (default: data/lcsts)",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        help=("Number of few-shot examples to extract (default: 0)"),
    )
    args = parser.parse_args()
    if args.source is None:
        test_file, train_file = download_from_huggingface(args.out / "_raw")
    else:
        if not args.source.is_dir():
            raise FileNotFoundError(
                f"source directory not found: {args.source}"
            )
        test_file = args.source / "test.json"
        train_file = args.source / "train.json"
        if not test_file.is_file():
            raise FileNotFoundError(f"test.json not found in {args.source}")
        if not train_file.is_file():
            raise FileNotFoundError(f"train.json not found in {args.source}")
    test_count = process_split(
        test_file,
        args.out / "test.jsonl",
        EXPECTED_TEST_COUNT,
        "lcsts-test",
    )
    if args.few_shot > 0:
        if not train_file.is_file():
            raise FileNotFoundError(
                "train.json required for few-shot but not found"
            )
        print(f"Extracting {args.few_shot} few-shot examples...")
        train_records = load_jsonl(train_file)
        validate_all_records(train_records, train_file.name)
        if len(train_records) < args.few_shot:
            raise ValueError(
                f"train.json has "
                f"{len(train_records)} records, "
                f"but --few-shot={args.few_shot} "
                f"requested"
            )
        few_shot_records = convert_records(
            train_records[: args.few_shot], "lcsts-fs"
        )
        write_jsonl(
            few_shot_records,
            args.out / "few_shot.jsonl",
        )
        print(
            f"  Wrote {len(few_shot_records)} "
            f"few-shot records -> "
            f"{args.out / 'few_shot.jsonl'}"
        )
    print("\nLCSTS dataset preparation complete!")
    print(f"  Test: {test_count} records")
    if args.few_shot > 0:
        print(f"  Few-shot: {args.few_shot} records")


if __name__ == "__main__":
    main()
