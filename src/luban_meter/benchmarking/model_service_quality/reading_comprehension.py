"""Reading collection and scoring."""

from __future__ import annotations

import json
import re
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from luban_meter.benchmarking.model_service_quality.common import (
    template_adapters,
)
from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
    ServiceError,
)
from luban_meter.benchmarking.model_service_quality.common.metrics import (
    token_f1,
)
from luban_meter.benchmarking.model_service_quality.common.parameters import (
    boolean_value,
    enum_value,
    fixed_value,
    non_negative_integer,
    positive_integer,
    positive_number,
    string_list,
    string_value,
)
from luban_meter.benchmarking.model_service_quality.common.parsers import (
    extract_final_output,
)
from luban_meter.benchmarking.model_service_quality.common.prompts import (
    render_squad_prompt,
    validate_prompt_version,
)
from luban_meter.benchmarking.model_service_quality.common.records import (
    scalar,
)
from luban_meter.data.engine import (
    resolve_data_path,
)
from luban_meter.runtime.thinking import response_facts


def validate_records(records: list[dict]) -> list[dict]:
    """Preserve all references and reject ambiguous or malformed
    samples.
    """
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
            not isinstance(answer, str) or not answer.strip()
            for answer in answers
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
            for question_record in paragraph["qas"]:
                rows.append(
                    {
                        "id": question_record["id"],
                        "title": article.get("title", ""),
                        "context": paragraph["context"],
                        "question": question_record["question"],
                        "answers": [
                            answer["text"]
                            for answer in question_record["answers"]
                        ],
                        "is_impossible": question_record["is_impossible"],
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


NO_ANSWER = "unanswerable"
SCORER_VERSION = "squad2-official-text-v1"


def normalize_answer(text: str) -> str:
    """Lowercase, remove ASCII punctuation, articles, then extra
    spaces.
    """
    text = "".join(
        character
        for character in text.lower()
        if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_answer(text: str) -> str:
    """Accept an exact refusal marker; otherwise preserve answer
    content.

    Empty output is a parse failure, not a deliberate abstention. No
    substring matching, explanation extraction, punctuation
    truncation or case folding is applied before recognizing the
    marker.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty or non-string model output")
    answer = text.strip()
    return "" if answer == NO_ANSWER else answer


def score_answer(prediction: str, answers: list[str]) -> tuple[float, float]:
    """Return EM/F1 in [0, 1], maximizing independently over
    references.
    """
    golds = [
        normalize_answer(answer)
        for answer in answers
        if normalize_answer(answer)
    ]
    golds = golds or [""]
    predicted = normalize_answer(prediction)
    return (
        float(any(predicted == gold for gold in golds)),
        max(token_f1(predicted.split(), gold.split()) for gold in golds),
    )


MEASUREMENT = "squad_exact_match_online_service"
PROTOCOL = "openai_compatible_chat_and_completions"


def validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    config = {
        "service_url": string_value(
            parameters, "service_url", "http://127.0.0.1:8000"
        ),
        "api_key": string_value(parameters, "api_key", ""),
        "request_timeout": positive_number(parameters, "request_timeout", 60),
        "dataset_path": string_value(parameters, "dataset_path", ""),
        "split": string_value(parameters, "split", "validation"),
        "max_samples": (
            None
            if parameters.get("max_samples") is None
            else positive_integer(parameters, "max_samples", 200)
        ),
        "shuffle": boolean_value(parameters, "shuffle", False),
        "seed": non_negative_integer(parameters, "seed", 42),
        "few_shot": non_negative_integer(parameters, "few_shot", 0),
        "eval_mode": enum_value(parameters, "eval_mode", ("gen",), "gen"),
        "prompt_format": enum_value(
            parameters, "prompt_format", ("chat", "base"), "chat"
        ),
        "prompt_version": string_value(
            parameters, "prompt_version", "squad2-gen-v1"
        ),
        "max_tokens": positive_integer(parameters, "max_tokens", 64),
        "stop": string_list(parameters, "stop", []),
        "max_concurrency": positive_integer(parameters, "max_concurrency", 8),
    }
    fixed_value(parameters, "temperature", 0.0)
    if not config["dataset_path"]:
        raise ValueError("dataset_path must not be empty")
    if config["few_shot"] != 0:
        raise ValueError("squad2-gen-v1 supports only few_shot=0")
    validate_prompt_version("squad", config["prompt_version"])
    return config


def generate_answer(
    client: OpenAIClient, prompt: str, config: dict[str, Any]
) -> dict[str, Any]:
    if config["prompt_format"] == "chat":
        return client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=config["max_tokens"],
            stop=config["stop"],
        )
    return client.complete(
        prompt, max_tokens=config["max_tokens"], stop=config["stop"]
    )


def collect_sample(
    client: OpenAIClient,
    sample: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rendered = template_adapters.render_input(
        config, sample, [], lambda: render_squad_prompt(sample)
    )
    prompt = rendered["prompt"]
    record = {
        "id": sample["id"],
        "is_impossible": sample["is_impossible"],
        "prompt": prompt,
        "raw_output": None,
        "final_output": None,
        "prediction": None,
        "reference": sample["answers"],
        "latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "status": "service_failed",
        "error": None,
    }
    try:
        response = generate_answer(client, prompt, config)
        record.update(
            {
                **response_facts(response),
                "raw_output": response["text"],
                "input_tokens": response.get("input_tokens") or 0,
                "output_tokens": response.get("output_tokens") or 0,
                "latency_ms": response.get("latency_ms") or 0.0,
            }
        )
        try:
            record["final_output"] = extract_final_output(response["text"])
            record["prediction"] = template_adapters.parse_output(
                config, sample, rendered, record["final_output"], parse_answer
            )
            record["answer_parser"] = config.get("answer_parser") or "reading"
            record["parser_context"] = {**sample, **rendered}
            record["status"] = "success"
        except ValueError as exc:
            record["status"] = "parse_failed"
            record["error"] = str(exc)
    except ServiceError as exc:
        record["error"] = str(exc)
    return record


def aggregate(samples):
    count = len(samples)
    return {
        key: scalar(
            sum(sample_record[key] for sample_record in samples) / count
            if count
            else None,
            "ratio",
            count,
        )
        for key in ("exact_match", "token_f1")
    }


def process(raw_result: Mapping) -> dict:
    metadata = dict(raw_result.get("metadata") or {})
    metadata.update({"scorer_version": SCORER_VERSION})
    if raw_result.get("status") == "failed":
        return {
            "status": "failed",
            "metrics": {},
            "metadata": metadata,
            "error": raw_result.get("error")
            or {
                "type": "CollectionError",
                "message": "collection failed",
            },
        }
    if raw_result.get("status") != "success":
        raise ValueError("raw status must be success or failed")
    rows = raw_result.get("metrics", {}).get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("raw samples must be a non-empty list")
    samples = []
    seen = set()
    counts = {"success": 0, "parse_failed": 0, "service_failed": 0}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") not in counts:
            raise ValueError("invalid sample or sample status")
        sample_id = row.get("id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen
        ):
            raise ValueError("sample IDs must be non-empty and unique")
        seen.add(sample_id)
        answers = row.get("reference")
        if not isinstance(answers, list) or any(
            not isinstance(answer, str) or not answer.strip()
            for answer in answers
        ):
            raise ValueError("reference must be a list of answer strings")
        impossible = row.get("is_impossible")
        if not isinstance(impossible, bool) or impossible != (not answers):
            raise ValueError("is_impossible contradicts references")
        status = row["status"]
        counts[status] += 1
        exact_match_score, token_f1_score = 0.0, 0.0
        if status == "success":
            text = row.get("raw_output")
            if "final_output" in row:
                text = extract_final_output(text)
                if text != row["final_output"]:
                    raise ValueError("final_output does not match raw output")
            if row.get("answer_parser", "reading") == "reading":
                prediction = parse_answer(text)
            else:
                from luban_meter.core.template_contracts import (
                    answer_parsers,
                    validate_fields,
                )

                parser = answer_parsers.get(row["answer_parser"])()
                fields = {
                    **row["parser_context"], "raw_output": text
                }
                validate_fields(fields, parser.requires, "parser input")
                parsed = parser.parse(fields)
                validate_fields(parsed, parser.provides, "parser output")
                prediction = parsed["prediction"]
            if prediction != row.get("prediction"):
                raise ValueError("prediction does not match raw output")
            exact_match_score, token_f1_score = score_answer(
                prediction, answers
            )
        samples.append(
            {
                "exact_match": exact_match_score,
                "token_f1": token_f1_score,
                "is_impossible": impossible,
            }
        )
    failed = counts["service_failed"]
    status = "success"
    if failed:
        status = "failed" if failed == len(rows) else "partial_failed"
    metrics = {
        **aggregate(samples),
        "has_answer": aggregate(
            [
                sample_record
                for sample_record in samples
                if not sample_record["is_impossible"]
            ]
        ),
        "no_answer": aggregate(
            [
                sample_record
                for sample_record in samples
                if sample_record["is_impossible"]
            ]
        ),
        "total_samples": scalar(len(rows), "sample"),
        "scored_samples": scalar(len(rows), "sample"),
        "successful_samples": scalar(counts["success"], "sample"),
        "service_failed": scalar(failed, "sample"),
        "parse_failed": scalar(counts["parse_failed"], "sample"),
    }
    metadata["failure_policy"] = "zero_score_all_selected_samples"
    return {
        "status": status,
        "metrics": {"task_view": {"squad": metrics}},
        "metadata": metadata,
        "error": {
            "type": "ServiceFailures",
            "message": f"{failed} samples failed at the service",
        }
        if failed
        else None,
    }
