"""Shared four-choice benchmark flow for the C-Eval/CMMLU dataset
family.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    extract_choice,
    extract_final_output,
)
from luban_meter.benchmarking.model_service_quality.common.prompts import (
    CHOICE_LETTERS,
    render_choice_prompt,
    validate_prompt_version,
)
from luban_meter.runtime.thinking import response_facts

PROTOCOL = "openai_compatible_chat_and_completions"
SCORER_VERSION = "metrics-v1"


@dataclass(frozen=True)
class ChoiceBenchmarkDefinition:
    """Dataset-family constants for a four-choice benchmark."""

    benchmark_name: str
    measurement: str
    dataset_label: str


def validate_choice_parameters(
    parameters: Mapping[str, Any],
    benchmark_definition: ChoiceBenchmarkDefinition,
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
        "eval_mode": enum_value(
            parameters, "eval_mode", ("ppl", "gen"), "ppl"
        ),
        "prompt_format": enum_value(
            parameters, "prompt_format", ("chat", "base"), "chat"
        ),
        "prompt_version": string_value(
            parameters,
            "prompt_version",
            f"{benchmark_definition.benchmark_name}-"
            f"v{3 if parameters.get('eval_mode') == 'gen' else 1}",
        ),
        "max_tokens": positive_integer(parameters, "max_tokens", 8),
        "stop": string_list(parameters, "stop", []),
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
    validate_prompt_version(
        benchmark_definition.benchmark_name, config["prompt_version"]
    )
    return config


def score_choices(
    client: OpenAIClient, prompt: str, choices: list[str]
) -> tuple[dict[str, float], float]:
    """Mean continuation logprob per choice via echo+logprobs scoring.

    Returns the per-letter scores and the total scoring latency in
    ms.

    ``/v1/completions`` is called with ``echo=true`` and
    ``max_tokens=1`` so the returned logprobs cover every prompt
    token followed by exactly one newly generated token. The
    continuation is located via the full-text token offsets --
    ``tokenize(prompt)`` vs ``tokenize(prompt + continuation)`` --
    and only the continuation slice ``[prompt_count:full_count]`` is
    scored, which drops the trailing generated token. Tokenizing the
    continuation in isolation is intentionally avoided: it can
    disagree with the prompt+continuation boundary.
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
    rendered = template_adapters.render_input(
        config,
        sample,
        few_shot_samples,
        lambda: render_choice_prompt(
            sample,
            few_shot_samples=few_shot_samples,
            strict_final_answer=not config["prompt_version"].endswith("-v1"),
            answer_only=config["prompt_version"].endswith("-v3"),
        ),
    )
    prompt = rendered["prompt"]
    record: dict[str, Any] = {
        "id": sample_id,
        "subject": subject,
        "mode": config["eval_mode"],
        "prompt_version": config["prompt_version"],
        "prompt_format": config["prompt_format"],
        "prompt": prompt,
        "choice_scores": None,
        "raw_output": None,
        "final_output": None,
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
            record.update(response_facts(response))
            raw_output = response["text"]
            record.update({
                "raw_output": raw_output,
                "input_tokens": response.get("input_tokens") or 0,
                "output_tokens": response.get("output_tokens") or 0,
                "latency_ms": response.get("latency_ms") or 0.0,
            })
            try:
                record["final_output"] = extract_final_output(raw_output)
                prediction = template_adapters.parse_output(
                    config, sample, rendered, record["final_output"],
                    extract_choice,
                )
            except ValueError as exc:
                record.update(status="parse_failed", error=str(exc))
                return record
            record.update(parsed_output=prediction, prediction=prediction)
            if prediction is None:
                record["status"] = "parse_failed"
                record["error"] = "no unambiguous final choice answer"
                return record
        record["correct"] = prediction == answer
        record["status"] = "success"
    except ServiceError as exc:
        record["error"] = str(exc)
    return record


def probe_logprobs_support(client: OpenAIClient) -> None:
    """Fail fast when the service cannot echo prompt logprobs."""
    client.completion_logprobs("A")
