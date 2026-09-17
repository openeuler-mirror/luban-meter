"""Collect LCSTS Chinese summarization samples.

Self-contained gen-mode benchmark for the LCSTS Part I
test split. The model generates a Chinese summary for
each source text; raw predictions and references are
stored for ROUGE scoring in calculate_metrics.py, matching
OpenCompass JiebaRougeEvaluator.
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
    load_records,
    select_records,
)
from luban_meter.benchmarking.model_service_quality.common.parameters import (
    boolean_value,
    enum_value,
    non_negative_integer,
    positive_integer,
    positive_number,
    string_list,
    string_value,
)
from luban_meter.benchmarking.model_service_quality.common.prompts import (
    render_lcsts_prompt,
    validate_prompt_version,
)

MEASUREMENT = "lcsts_rouge_online_service"
PROTOCOL = "openai_compatible_chat_and_completions"
SCORER_VERSION = "metrics-v1"

LCSTS_DEFAULT_PROMPT = "阅读以下文章，并给出简短的摘要：{content}\n摘要如下："


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LCSTS model service quality scenario",
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_request(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
    )
    request = payload.get("request")
    parameters = payload.get("parameters")
    if not isinstance(request, Mapping):
        raise TypeError(
            "request must be an object",
        )
    if not isinstance(parameters, Mapping):
        raise TypeError(
            "parameters must be an object",
        )
    return dict(request), dict(parameters)


def validate_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "service_url": string_value(
            parameters,
            "service_url",
            "http://127.0.0.1:8000",
        ),
        "api_key": string_value(
            parameters,
            "api_key",
            "",
        ),
        "request_timeout": positive_number(
            parameters,
            "request_timeout",
            60,
        ),
        "dataset_path": string_value(
            parameters,
            "dataset_path",
            "",
        ),
        "split": string_value(
            parameters,
            "split",
            "test",
        ),
        "max_samples": positive_integer(
            parameters,
            "max_samples",
            200,
        ),
        "shuffle": boolean_value(
            parameters,
            "shuffle",
            False,
        ),
        "seed": non_negative_integer(
            parameters,
            "seed",
            42,
        ),
        "few_shot_path": string_value(
            parameters,
            "few_shot_path",
            "",
        ),
        "few_shot": non_negative_integer(
            parameters,
            "few_shot",
            0,
        ),
        "eval_mode": enum_value(
            parameters,
            "eval_mode",
            ("gen",),
            "gen",
        ),
        "prompt_format": enum_value(
            parameters,
            "prompt_format",
            ("chat", "base"),
            "chat",
        ),
        "prompt_version": enum_value(
            parameters,
            "prompt_version",
            ("lcsts-v1",),
            "lcsts-v1",
        ),
        "prompt_template": (
            string_value(
                parameters,
                "prompt_template",
                LCSTS_DEFAULT_PROMPT,
            )
            or LCSTS_DEFAULT_PROMPT
        ),
        "max_tokens": non_negative_integer(
            parameters,
            "max_tokens",
            512,
        ),
        "stop": string_list(
            parameters,
            "stop",
            [],
        ),
        "max_concurrency": positive_integer(
            parameters,
            "max_concurrency",
            8,
        ),
    }
    validate_prompt_version(
        "lcsts",
        config["prompt_version"],
    )
    if config["few_shot"] > 0 and not config["few_shot_path"]:
        raise ValueError(
            "few_shot_path is required when few_shot > 0",
        )
    return config


def prepare_samples(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        content = record.get("content")
        abst = record.get("abst")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("sample content must be a non-empty string")
        if not isinstance(abst, str) or not abst.strip():
            raise ValueError("sample abst must be a non-empty string")
        prepared.append(
            {
                "content": content.strip(),
                "abst": abst.strip(),
            }
        )
    return prepared


def collect_sample(
    client: OpenAIClient,
    sample: Mapping[str, Any],
    few_shot_samples: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "content": sample["content"],
        "reference": sample["abst"],
    }
    try:
        prompt = render_lcsts_prompt(
            {"content": sample["content"]},
            few_shot_samples=few_shot_samples,
            template=config["prompt_template"],
        )
        if config["prompt_format"] == "chat":
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            result = client.chat(
                messages,
                max_tokens=config["max_tokens"],
                stop=config["stop"] or None,
            )
        else:
            result = client.complete(
                prompt,
                max_tokens=config["max_tokens"],
                stop=config["stop"] or None,
            )
        record.update(
            {
                "prediction": result["text"],
                "input_tokens": result.get(
                    "input_tokens",
                ),
                "output_tokens": result.get(
                    "output_tokens",
                ),
                "latency_ms": result["latency_ms"],
                "status": "success",
            }
        )
    except ServiceError as exc:
        record.update(
            {
                "prediction": "",
                "status": "service_failed",
                "error": str(exc),
            }
        )
    return record


def collect_raw_result(
    request: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    config = validate_parameters(parameters)
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

    samples = prepare_samples(
        select_records(
            load_records(config["dataset_path"]),
            max_samples=config["max_samples"],
            shuffle=config["shuffle"],
            seed=config["seed"],
        ),
    )
    if not samples:
        raise ValueError(
            "dataset contains no samples",
        )

    few_shot_samples: list[dict[str, Any]] = []
    if config["few_shot"] > 0:
        few_shot_samples = prepare_samples(
            load_records(config["few_shot_path"]),
        )[: config["few_shot"]]

    with ThreadPoolExecutor(
        max_workers=config["max_concurrency"],
    ) as executor:
        futures = [
            executor.submit(
                collect_sample,
                client,
                sample,
                few_shot_samples,
                config,
            )
            for sample in samples
        ]
        sample_records = [sample_future.result() for sample_future in futures]

    counts = {
        "total": len(sample_records),
        "service_failed": sum(
            1
            for sample_record in sample_records
            if sample_record["status"] == "service_failed"
        ),
    }
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metrics": {
            "samples": sample_records,
            "counts": counts,
        },
        "metadata": {
            "measurement": MEASUREMENT,
            "protocol": PROTOCOL,
            "service_url": config["service_url"],
            "model": served_model_name,
            "dataset": "LCSTS",
            "dataset_path": config["dataset_path"],
            "split": config["split"],
            "sample_count": counts["total"],
            "few_shot": config["few_shot"],
            "eval_mode": config["eval_mode"],
            "prompt_format": config["prompt_format"],
            "prompt_version": config["prompt_version"],
            "max_tokens": config["max_tokens"],
            "stop": config["stop"],
            "max_concurrency": config["max_concurrency"],
            "scorer_version": SCORER_VERSION,
        },
        "artifacts": {},
    }


def failure_result(
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "failed",
        "metrics": {},
        "metadata": {
            "measurement": MEASUREMENT,
            "protocol": PROTOCOL,
        },
        "artifacts": {},
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def main() -> None:
    args = parse_args()
    try:
        request, parameters = load_request(
            args.request,
        )
        result = collect_raw_result(
            request,
            parameters,
        )
    except Exception as exc:  # noqa: BLE001
        result = failure_result(exc)
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
