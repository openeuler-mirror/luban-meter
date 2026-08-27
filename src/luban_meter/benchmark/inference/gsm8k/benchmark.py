"""Collect GSM8K exact-match samples from an OpenAI-compatible service."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from luban_meter.benchmark.inference.common.client import (
    OpenAIClient,
    ServiceError,
)
from luban_meter.benchmark.inference.common.dataset import (
    load_records,
    select_records,
)
from luban_meter.benchmark.inference.common.parameters import (
    boolean_value,
    enum_value,
    fixed_value,
    non_negative_integer,
    positive_integer,
    positive_number,
    string_list,
    string_value,
)
from luban_meter.benchmark.inference.common.parsers import extract_number
from luban_meter.benchmark.inference.common.prompts import (
    render_math_prompt,
    validate_prompt_version,
)

MEASUREMENT = "gsm8k_exact_match_online_service"
PROTOCOL = "openai_compatible_chat_and_completions"
SCORER_VERSION = "metrics-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM8K inference benchmark")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_request(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request")
    parameters = payload.get("parameters")
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be an object")
    return dict(request), dict(parameters)


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
        "prompt_version": string_value(parameters, "prompt_version", "gsm8k-v1"),
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
    prompt = render_math_prompt(sample, few_shot_samples=few_shot_samples)
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
        prediction = extract_number(raw_output)
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
    """Attach the parsed numeric reference; fail fast on malformed rows."""
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question = record.get("question")
        answer = record.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"sample {index}: question must be a non-empty string")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"sample {index}: answer must be a non-empty string")
        reference = extract_number(answer)
        if reference is None:
            raise ValueError(f"sample {index}: no numeric reference answer")
        prepared.append({**record, "reference_number": reference})
    return prepared


def run_benchmark(request: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    config = validate_parameters(parameters)
    model = request.get("model_name")
    if not isinstance(model, str) or not model:
        model = OpenAIClient(
            config["service_url"], api_key=config["api_key"],
            timeout=config["request_timeout"],
        ).discover_model()
    client = OpenAIClient(
        config["service_url"],
        model=model,
        api_key=config["api_key"],
        timeout=config["request_timeout"],
    )

    samples = prepare_samples(
        select_records(
            load_records(config["dataset_path"]),
            max_samples=config["max_samples"],
            shuffle=config["shuffle"],
            seed=config["seed"],
        )
    )
    if not samples:
        raise ValueError("dataset contains no samples")

    few_shot_samples: list[Mapping[str, Any]] = []
    if config["few_shot"] > 0:
        few_shot_samples = prepare_samples(
            load_records(config["few_shot_path"])
        )[: config["few_shot"]]

    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as executor:
        futures = [
            executor.submit(
                collect_sample, client, sample, few_shot_samples, config
            )
            for sample in samples
        ]
        sample_records = [future.result() for future in futures]

    counts = {
        "total": len(sample_records),
        "correct": sum(
            1
            for record in sample_records
            if record["status"] == "success" and record["correct"]
        ),
        "parse_failed": sum(
            1 for record in sample_records if record["status"] == "parse_failed"
        ),
        "service_failed": sum(
            1 for record in sample_records if record["status"] == "service_failed"
        ),
    }
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metrics": {"samples": sample_records, "counts": counts},
        "metadata": {
            "measurement": MEASUREMENT,
            "protocol": PROTOCOL,
            "service_url": config["service_url"],
            "model": model,
            "dataset": "GSM8K",
            "dataset_path": config["dataset_path"],
            "split": config["split"],
            "sample_count": counts["total"],
            "few_shot": config["few_shot"],
            "eval_mode": config["eval_mode"],
            "prompt_format": config["prompt_format"],
            "prompt_version": config["prompt_version"],
            "temperature": 0.0,
            "max_tokens": config["max_tokens"],
            "stop": config["stop"],
            "max_concurrency": config["max_concurrency"],
            "scorer_version": SCORER_VERSION,
        },
        "artifacts": {},
    }


def failure_result(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "failed",
        "metrics": {},
        "metadata": {"measurement": MEASUREMENT, "protocol": PROTOCOL},
        "artifacts": {},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main() -> None:
    args = parse_args()
    try:
        request, parameters = load_request(args.request)
        result = run_benchmark(request, parameters)
    except Exception as exc:  # noqa: BLE001
        result = failure_result(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
