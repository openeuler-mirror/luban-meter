"""Arithmetic collection and scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from luban_meter.benchmarking.model_service_quality.common import (
    template_adapters,
)
from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
    ServiceError,
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
    extract_number,
)
from luban_meter.benchmarking.model_service_quality.common.prompts import (
    render_math_prompt,
    validate_prompt_version,
)
from luban_meter.benchmarking.model_service_quality.common.records import (
    number_value,
    object_list,
    scalar,
    token_count,
)

MEASUREMENT = "gsm8k_exact_match_online_service"
PROTOCOL = "openai_compatible_chat_and_completions"
SCORER_VERSION = "metrics-v1"


def validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    config = {
        "service_url": string_value(
            parameters, "service_url", "http://127.0.0.1:8000"
        ),
        "api_key": string_value(parameters, "api_key", ""),
        "request_timeout": positive_number(parameters, "request_timeout", 60),
        "dataset_path": string_value(parameters, "dataset_path", ""),
        "split": string_value(parameters, "split", "test"),
        "max_samples": positive_integer(parameters, "max_samples", 200),
        "shuffle": boolean_value(parameters, "shuffle", False),
        "seed": non_negative_integer(parameters, "seed", 42),
        "few_shot_path": string_value(parameters, "few_shot_path", ""),
        "few_shot": non_negative_integer(parameters, "few_shot", 8),
        "eval_mode": enum_value(parameters, "eval_mode", ("gen",), "gen"),
        "prompt_format": enum_value(
            parameters, "prompt_format", ("chat", "base"), "chat"
        ),
        "prompt_version": string_value(
            parameters, "prompt_version", "gsm8k-v1"
        ),
        "max_tokens": positive_integer(parameters, "max_tokens", 512),
        "stop": string_list(parameters, "stop", []),
        "max_concurrency": positive_integer(parameters, "max_concurrency", 8),
    }
    fixed_value(parameters, "temperature", 0.0)
    if not config["dataset_path"]:
        raise ValueError("dataset_path must not be empty")
    if config["few_shot"] > 0 and not config["few_shot_path"]:
        raise ValueError("few_shot_path is required when few_shot > 0")
    validate_prompt_version("gsm8k", config["prompt_version"])
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
    few_shot_samples: list[Mapping[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(sample.get("id", ""))
    reference = float(sample["reference_number"])
    rendered = template_adapters.render_input(
        config, sample, few_shot_samples,
        lambda: render_math_prompt(sample, few_shot_samples=few_shot_samples),
    )
    prompt = rendered["prompt"]
    record: dict[str, Any] = {
        "id": sample_id,
        "subject": "math",
        "mode": config["eval_mode"],
        "prompt_version": config["prompt_version"],
        "prompt_format": config["prompt_format"],
        "prompt": prompt,
        "raw_output": None,
        "parsed_output": None,
        "prediction": None,
        "reference": reference,
        "correct": None,
        "latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "status": "service_failed",
        "error": None,
    }
    try:
        response = generate_answer(client, prompt, config)
        raw_output = response["text"]
        prediction = template_adapters.parse_output(
            config, sample, rendered, raw_output, extract_number
        )
        record.update(
            {
                "raw_output": raw_output,
                "parsed_output": prediction,
                "prediction": prediction,
                "input_tokens": response.get("input_tokens") or 0,
                "output_tokens": response.get("output_tokens") or 0,
                "latency_ms": response.get("latency_ms") or 0.0,
            }
        )
        if prediction is None:
            record["status"] = "parse_failed"
            record["error"] = "no numeric answer found in output"
            return record
        record["correct"] = prediction == reference
        record["status"] = "success"
    except ServiceError as exc:
        record["error"] = str(exc)
    return record


def prepare_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the parsed numeric reference; fail fast on malformed
    rows.
    """
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question = record.get("question")
        answer = record.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"sample {index}: question must be a non-empty string"
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"sample {index}: answer must be a non-empty string"
            )
        reference = extract_number(answer)
        if reference is None:
            raise ValueError(f"sample {index}: no numeric reference answer")
        prepared.append({**record, "reference_number": reference})
    return prepared


SAMPLE_STATUSES = ("success", "parse_failed", "service_failed")


def process_sample(raw_sample: Mapping[str, Any]) -> dict[str, Any]:
    status = raw_sample.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"sample status must be one of {SAMPLE_STATUSES}")
    reference = number_value(raw_sample, "reference")
    if status in ("success", "parse_failed"):
        if status == "success":
            number_value(raw_sample, "prediction")
        token_count(raw_sample, "input_tokens")
        token_count(raw_sample, "output_tokens")
        correct = raw_sample.get("correct")
        if correct is not None and not isinstance(correct, bool):
            raise TypeError("sample correct must be a boolean")
    return {
        "status": status,
        "reference": reference,
        "correct": bool(raw_sample.get("correct"))
        if status == "success"
        else False,
    }


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_status = raw_result.get("status")
    metadata = raw_result.get("metadata")
    result_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

    if raw_status == "failed":
        error = raw_result.get("error") or {
            "type": "UnknownError",
            "message": "raw result failed without error detail",
        }
        result_metadata.setdefault("scorer_version", SCORER_VERSION)
        return {
            "status": "failed",
            "metrics": {},
            "metadata": result_metadata,
            "error": error,
        }
    if raw_status != "success":
        raise ValueError(f"raw status must be success or failed: {raw_status}")

    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")
    raw_samples = object_list(raw_metrics.get("samples"), "raw samples")
    if not raw_samples:
        raise ValueError("raw samples must not be empty")

    samples = [process_sample(raw_sample) for raw_sample in raw_samples]
    scored = [
        sample for sample in samples if sample["status"] != "service_failed"
    ]
    parse_failed = sum(
        1 for sample in samples if sample["status"] == "parse_failed"
    )
    service_failed = sum(
        1 for sample in samples if sample["status"] == "service_failed"
    )
    if not scored:
        raise ValueError("all samples failed at the service; nothing to score")

    correct = sum(1 for sample in scored if sample["correct"])
    result_status = "success" if service_failed == 0 else "partial_failed"

    result_metadata.setdefault("scorer_version", SCORER_VERSION)
    result_metadata.update(
        {
            "correct_samples": correct,
            "parse_failed_samples": parse_failed,
            "service_failed_samples": service_failed,
        }
    )
    return {
        "status": result_status,
        "metrics": {
            "task_view": {
                "gsm8k": {
                    "exact_match": scalar(
                        correct / len(scored), "ratio", len(scored)
                    ),
                    "total_samples": scalar(len(samples), "sample"),
                    "scored_samples": scalar(len(scored), "sample"),
                    "parse_failed": scalar(parse_failed, "sample"),
                    "service_failed": scalar(service_failed, "sample"),
                }
            }
        },
        "metadata": result_metadata,
        "error": (
            {
                "type": "ServiceFailures",
                "message": f"{service_failed} samples failed at the service",
            }
            if service_failed
            else None
        ),
    }
