"""Convert WikiText-2 raw files into luban-meter jsonl datasets.

Usage:
    python -m \
luban_meter.benchmarking.model_service_quality.scripts.prepare_wikitext \
        --source /path/to/wikitext-2-raw-v1 [--out data/wikitext]

Expected layout of the official WikiText-2-raw-v1 release:
    <source>/wiki.valid.raw    # validation split
    <source>/wiki.test.raw     # test split

Each raw file is a plain-text corpus where paragraphs are separated
by blank lines. The wikitext_detokenizer restores the @-@/@,@/@.@
placeholders and space-punctuation patterns back to natural text,
matching lm-eval-harness preprocess_wikitext.wikitext_detokenizer.

Output: jsonl records with id, text (detokenized), raw_text (original),
bytes (UTF-8 byte count on raw_text), words (whitespace-split on
raw_text). Benchmarks never download data at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def wikitext_detokenizer(text: str) -> str:
    """Restore WikiText-2 tokenization artifacts to natural text.

    Equivalent to lm-eval-harness
    ``lm_eval.tasks.wikitext.preprocess_wikitext.wikitext_detokenizer``.
    """
    string = text
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # num separators
    string = string.replace("@-@", "-")
    string = string.replace("@,@", ",")
    string = string.replace("@.@", ".")
    # punctuation
    string = string.replace(" :", ":")
    string = string.replace(" ;", ";")
    string = string.replace(" .", ".")
    string = string.replace(" ?", "?")
    string = string.replace(" !", "!")
    string = string.replace(" ,", ",")
    # parens
    string = string.replace("( ", "(")
    string = string.replace(" )", ")")
    # quotes
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = string.replace("' ", "'")
    string = string.replace(" '", "'")
    # dashes
    string = string.replace(" -- ", " -- ")
    string = string.replace("—", "—")
    string = string.replace(" – ", " – ")
    return string


def split_paragraphs(raw_text: str) -> list[str]:
    """Split raw WikiText file into paragraph blocks."""
    blocks: list[str] = []
    current: list[str] = []
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(" ".join(current))
                current = []
        else:
            current.append(stripped)
    if current:
        blocks.append(" ".join(current))
    return blocks


def load_wikitext_raw(path: Path) -> list[dict[str, object]]:
    """Read a raw WikiText file and produce per-paragraph records.

    Includes section headers to match lm-eval-harness coverage.
    """
    raw_text = path.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(raw_text)
    records: list[dict[str, object]] = []
    index = 0
    for para in paragraphs:
        detokenized = wikitext_detokenizer(para)
        if not detokenized.strip():
            continue
        raw_bytes = len(para.encode("utf-8"))
        raw_words = len(para.split())
        records.append(
            {
                "id": f"wikitext-{index}",
                "text": detokenized,
                "raw_text": para,
                "bytes": raw_bytes,
                "words": raw_words,
            }
        )
        index += 1
    return records


def write_jsonl(records: list[dict[str, object]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=("directory containing wiki.valid.raw and/or wiki.test.raw"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/wikitext"))
    parser.add_argument(
        "--prefix",
        type=str,
        default="wikitext",
        help="id prefix for generated records",
    )
    args = parser.parse_args()

    for name, out_name in [
        ("wiki.valid.raw", "val.jsonl"),
        ("wiki.test.raw", "test.jsonl"),
    ]:
        raw_file = args.source / name
        if not raw_file.is_file():
            print(f"skipping {raw_file} (not found)")
            continue
        records = load_wikitext_raw(raw_file)
        # Apply the user-specified prefix
        for paragraph_index, record in enumerate(records):
            record["id"] = (
                f"{args.prefix}-{out_name.split('.')[0]}-{paragraph_index}"
            )
        out_path = args.out / out_name
        write_jsonl(records, out_path)
        print(f"wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
