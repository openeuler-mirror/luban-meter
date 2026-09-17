"""Collect SQuAD 2.0 exact-match samples from an OpenAI-compatible
service.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
    ServiceError,
)
from luban_meter.benchmarking.model_service_quality.common.dataset import (
    select_records,
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
from luban_meter.benchmarking.model_service_quality.common.prompts import (
    render_squad_prompt,
    validate_prompt_version,
)
from luban_meter.benchmarking.model_service_quality.squad.dataset import (
    load_samples,
)
from luban_meter.benchmarking.model_service_quality.squad.scoring import (
    SCORER_VERSION,
    parse_answer,
)

MEASUREMENT = "squad_exact_match_online_service"
PROTOCOL = "openai_compatible_chat_and_completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQuAD 2.0 model service quality scenario"
    )
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
        "stop": string_list(parameters, "stop", ["\n"]),
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
    prompt = render_squad_prompt(sample)
    record = {
        "id": sample["id"],
        "is_impossible": sample["is_impossible"],
        "prompt": prompt,
        "raw_output": None,
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
                "raw_output": response["text"],
                "input_tokens": response.get("input_tokens") or 0,
                "output_tokens": response.get("output_tokens") or 0,
                "latency_ms": response.get("latency_ms") or 0.0,
            }
        )
        try:
            record["prediction"] = parse_answer(response["text"])
            record["status"] = "success"
        except ValueError as exc:
            record["status"] = "parse_failed"
            record["error"] = str(exc)
    except ServiceError as exc:
        record["error"] = str(exc)
    return record


def collect_raw_result(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    config = validate_parameters(parameters)
    samples = select_records(
        load_samples(config["dataset_path"]),
        max_samples=config["max_samples"],
        shuffle=config["shuffle"],
        seed=config["seed"],
    )

    served_model_name = request.get("model_name")
    if not isinstance(served_model_name, str) or not served_model_name:
        served_model_name = OpenAIClient(
            config["service_url"],
            api_key=config["api_key"],
            timeout=config["request_timeout"],
        ).discover_served_model_name()
    client = OpenAIClient(
        config["service_url"],
        served_model_name=served_model_name,
        api_key=config["api_key"],
        timeout=config["request_timeout"],
    )

    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as executor:
        futures = [
            executor.submit(collect_sample, client, sample, config)
            for sample in samples
        ]
        sample_records = [future.result() for future in futures]

    counts = {
        "total": len(sample_records),
        "parse_failed": sum(
            1
            for record in sample_records
            if record["status"] == "parse_failed"
        ),
        "service_failed": sum(
            1
            for record in sample_records
            if record["status"] == "service_failed"
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
            "model": served_model_name,
            "dataset": "SQuAD 2.0",
            "dataset_version": "2.0",
            "shuffle": config["shuffle"],
            "seed": config["seed"],
            "max_samples": config["max_samples"],
            "failure_policy": "zero_score_all_selected_samples",
            "no_answer_marker": "unanswerable",
            "context_truncation": False,
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
        result = collect_raw_result(request, parameters)
    except Exception as exc:  # noqa: BLE001
        result = failure_result(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
