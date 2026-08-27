"""Shared four-choice benchmark flow for the C-Eval/CMMLU dataset family."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luban_meter.benchmark.inference.common.client import (
    OpenAIClient,
    ServiceError,
)
from luban_meter.benchmark.inference.common.dataset import (
    load_records,
    select_few_shot,
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
from luban_meter.benchmark.inference.common.parsers import extract_choice
from luban_meter.benchmark.inference.common.prompts import (
    CHOICE_LETTERS,
    render_choice_prompt,
    validate_prompt_version,
)

PROTOCOL = "openai_compatible_chat_and_completions"
SCORER_VERSION = "metrics-v1"


@dataclass(frozen=True)
class ChoiceSpec:
    """Dataset-family constants for a four-choice benchmark."""

    benchmark: str
    measurement: str
    dataset_label: str


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
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


def validate_choice_parameters(
    parameters: Mapping[str, Any], spec: ChoiceSpec
) -> dict[str, Any]:
    config = {
        "service_url": string_value(
            parameters, "service_url", "http://127.0.0.1:8000"
        ),
        "api_key": string_value(parameters, "api_key", ""),
        "request_timeout": positive_number(parameters, "request_timeout", 60),
        "dataset_path": string_value(parameters, "dataset_path", ""),
        "split": string_value(parameters, "split", "val"),
        "max_samples": positive_integer(parameters, "max_samples", 200),
        "shuffle": boolean_value(parameters, "shuffle", False),
        "seed": non_negative_integer(parameters, "seed", 42),
        "few_shot_path": string_value(parameters, "few_shot_path", ""),
        "few_shot": non_negative_integer(parameters, "few_shot", 5),
        "eval_mode": enum_value(parameters, "eval_mode", ("ppl", "gen"), "ppl"),
        "prompt_format": enum_value(
            parameters, "prompt_format", ("chat", "base"), "chat"
        ),
        "prompt_version": string_value(
            parameters, "prompt_version", f"{spec.benchmark}-v1"
        ),
        "max_tokens": positive_integer(parameters, "max_tokens", 8),
        "stop": string_list(parameters, "stop", ["\n"]),
        "max_concurrency": positive_integer(parameters, "max_concurrency", 8),
    }
    fixed_value(parameters, "temperature", 0.0)
    if config["eval_mode"] == "ppl" and config["prompt_format"] == "chat":
        raise ValueError(
            "eval_mode=ppl requires prompt_format=base: chat-template "
            "logprob scoring is not implemented"
        )
    if not config["dataset_path"]:
        raise ValueError("dataset_path must not be empty")
    if config["few_shot"] > 0 and not config["few_shot_path"]:
        raise ValueError("few_shot_path is required when few_shot > 0")
    validate_prompt_version(spec.benchmark, config["prompt_version"])
    return config


def score_choices(
    client: OpenAIClient, prompt: str, choices: list[str]
) -> tuple[dict[str, float], float]:
    """Mean continuation logprob per choice via echo+logprobs scoring.

    Returns the per-letter scores and the total scoring latency in ms.

    ``/v1/completions`` is called with ``echo=true`` and ``max_tokens=1`` so the
    returned logprobs cover every prompt token followed by exactly one newly
    generated token. The continuation is located via the full-text token offsets
    -- ``tokenize(prompt)`` vs ``tokenize(prompt + continuation)`` -- and only the
    continuation slice ``[prompt_count:full_count]`` is scored, which drops the
    trailing generated token. Tokenizing the continuation in isolation is
    intentionally avoided: it can disagree with the prompt+continuation boundary.
    """
    scores: dict[str, float] = {}
    latency_ms = 0.0
    prompt_count = len(client.tokenize(prompt))
    for letter, choice in zip(CHOICE_LETTERS, choices):
        continuation = f" {letter}. {choice}"
        full_count = len(client.tokenize(prompt + continuation))
        response = client.completion_logprobs(prompt + continuation)
        latency_ms += response.get("latency_ms") or 0.0
        logprobs = response["token_logprobs"]
        scored = [
            value
            for value in logprobs[prompt_count:full_count]
            if value is not None
        ]
        if not scored:
            raise ServiceError(f"no logprobs returned for choice {letter}")
        scores[letter] = sum(scored) / len(scored)
    return scores, latency_ms


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
    subject = str(sample.get("subject", "unknown"))
    answer = str(sample.get("answer", "")).strip().upper()
    prompt = render_choice_prompt(sample, few_shot_samples=few_shot_samples)
    record: dict[str, Any] = {
        "id": sample_id,
        "subject": subject,
        "mode": config["eval_mode"],
        "prompt_version": config["prompt_version"],
        "prompt_format": config["prompt_format"],
        "prompt": prompt,
        "choice_scores": None,
        "raw_output": None,
        "parsed_output": None,
        "prediction": None,
        "reference": answer,
        "correct": None,
        "latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "status": "service_failed",
        "error": None,
    }
    try:
        if config["eval_mode"] == "ppl":
            scores, latency_ms = score_choices(
                client, prompt, list(sample["choices"])
            )
            prediction = max(scores, key=lambda letter: scores[letter])
            record.update(
                {
                    "choice_scores": scores,
                    "prediction": prediction,
                    "parsed_output": prediction,
                    "latency_ms": latency_ms,
                }
            )
        else:
            response = generate_answer(client, prompt, config)
            raw_output = response["text"]
            prediction = extract_choice(raw_output)
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
                record["error"] = "no choice letter found in output"
                return record
        record["correct"] = prediction == answer
        record["status"] = "success"
    except ServiceError as exc:
        record["error"] = str(exc)
    return record


def probe_logprobs_support(client: OpenAIClient) -> None:
    """Fail fast when the service cannot echo prompt logprobs."""
    client.completion_logprobs("A")


def run_choice_benchmark(
    request: dict[str, Any], parameters: dict[str, Any], spec: ChoiceSpec
) -> dict[str, Any]:
    config = validate_choice_parameters(parameters, spec)
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

    records = load_records(config["dataset_path"])
    samples = select_records(
        records,
        max_samples=config["max_samples"],
        shuffle=config["shuffle"],
        seed=config["seed"],
    )
    if not samples:
        raise ValueError("dataset contains no samples")

    few_shot_records: list[dict[str, Any]] = []
    if config["few_shot"] > 0:
        few_shot_records = load_records(config["few_shot_path"])

    if config["eval_mode"] == "ppl":
        probe_logprobs_support(client)

    def few_shot_for(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        if not few_shot_records:
            return []
        return select_few_shot(
            few_shot_records, "subject", sample.get("subject"), config["few_shot"]
        )

    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as executor:
        futures = [
            executor.submit(
                collect_sample, client, sample, few_shot_for(sample), config
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
            "measurement": spec.measurement,
            "protocol": PROTOCOL,
            "service_url": config["service_url"],
            "model": model,
            "dataset": spec.dataset_label,
            "dataset_path": config["dataset_path"],
            "split": config["split"],
            "sample_count": counts["total"],
            "few_shot": config["few_shot"],
            "eval_mode": config["eval_mode"],
            "prompt_format": config["prompt_format"],
            "prompt_version": config["prompt_version"],
            "temperature": 0.0,
            "max_tokens": (
                config["max_tokens"] if config["eval_mode"] == "gen" else None
            ),
            "stop": config["stop"] if config["eval_mode"] == "gen" else [],
            "max_concurrency": config["max_concurrency"],
            "scorer_version": SCORER_VERSION,
        },
        "artifacts": {},
    }


def choice_failure_result(error: Exception, spec: ChoiceSpec) -> dict[str, Any]:
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "failed",
        "metrics": {},
        "metadata": {"measurement": spec.measurement, "protocol": PROTOCOL},
        "artifacts": {},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def choice_main(spec: ChoiceSpec, description: str) -> None:
    args = parse_args(description)
    try:
        request, parameters = load_request(args.request)
        result = run_choice_benchmark(request, parameters, spec)
    except Exception as exc:  # noqa: BLE001
        result = choice_failure_result(exc, spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
