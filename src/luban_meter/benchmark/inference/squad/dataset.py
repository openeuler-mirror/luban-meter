"""Strict local SQuAD 2.0 loading and official JSON conversion."""

import json
from pathlib import Path

from luban_meter.benchmark.inference.common.dataset import resolve_data_path


def validate_records(records: list[dict]) -> list[dict]:
    """Preserve all references and reject ambiguous or malformed samples."""
    if not records:
        raise ValueError("dataset contains no samples")
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("sample must be an object")
        for field in ("id", "context", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"sample {field} must be a non-empty string")
        if row["id"] in seen:
            raise ValueError(f"duplicate sample id: {row['id']}")
        seen.add(row["id"])
        if not isinstance(row.get("title", ""), str):
            raise ValueError("title must be a string")
        impossible = row.get("is_impossible")
        if not isinstance(impossible, bool):
            raise ValueError("is_impossible must be a boolean")
        answers = row.get("answers")
        if not isinstance(answers, list) or any(
            not isinstance(a, str) or not a.strip() for a in answers
        ):
            raise ValueError("answers must be a list of non-empty strings")
        if impossible != (len(answers) == 0):
            raise ValueError("answers contradict is_impossible")
    return records


def convert_official(data: dict) -> list[dict]:
    """Flatten v2.0 JSON; plausible_answers are never gold references."""
    if not isinstance(data, dict) or data.get("version") != "v2.0":
        raise ValueError("expected official SQuAD version v2.0")
    articles = data.get("data")
    if not isinstance(articles, list):
        raise ValueError("official data must be a list")
    rows = []
    for article in articles:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                rows.append(
                    {
                        "id": qa["id"],
                        "title": article.get("title", ""),
                        "context": paragraph["context"],
                        "question": qa["question"],
                        "answers": [a["text"] for a in qa["answers"]],
                        "is_impossible": qa["is_impossible"],
                    }
                )
    return validate_records(rows)


def load_samples(path: str | Path) -> list[dict]:
    """Read official JSON or prepared JSONL, without network access."""
    source = resolve_data_path(path)
    with source.open(encoding="utf-8") as handle:
        if source.suffix == ".jsonl":
            return validate_records(
                [json.loads(line) for line in handle if line.strip()]
            )
        return convert_official(json.load(handle))
