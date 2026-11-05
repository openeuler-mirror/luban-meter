"""Summarization collection and scoring."""

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
    render_lcsts_prompt,
    validate_prompt_version,
)
from luban_meter.benchmarking.model_service_quality.common.records import (
    object_list,
    scalar,
)
from luban_meter.log import get_logger
from luban_meter.runtime.thinking import response_facts

logger = get_logger(__name__)


MEASUREMENT = "lcsts_rouge_online_service"
PROTOCOL = "openai_compatible_chat_and_completions"
SCORER_VERSION = "metrics-v1"
LCSTS_DEFAULT_PROMPT = "阅读以下文章，并给出简短的摘要：{content}\n摘要如下："


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
        "raw_output": None,
        "final_output": None,
    }
    try:
        rendered = template_adapters.render_input(
            config, sample, few_shot_samples,
            lambda: render_lcsts_prompt(
                sample, few_shot_samples=few_shot_samples,
                template=config["prompt_template"],
            ),
        )
        prompt = rendered["prompt"]
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
                stop=config["stop"],
            )
        else:
            result = client.complete(
                prompt,
                max_tokens=config["max_tokens"],
                stop=config["stop"],
            )
        record.update(
            {
                **response_facts(result),
                "raw_output": result["text"],
                "input_tokens": result.get(
                    "input_tokens",
                ),
                "output_tokens": result.get(
                    "output_tokens",
                ),
                "latency_ms": result["latency_ms"],
            }
        )
        try:
            record["final_output"] = extract_final_output(result["text"])
            record["prediction"] = template_adapters.parse_output(
                config, sample, rendered, record["final_output"],
                lambda text: text,
            )
        except ValueError as exc:
            logger.warning("Answer parsing failed", exc_info=True)
            record.update(prediction="", status="parse_failed", error=str(exc))
            return record
        record["status"] = "success"
    except ServiceError as exc:
        logger.exception("Model service failed")
        record.update(
            {
                "prediction": "",
                "status": "service_failed",
                "error": str(exc),
            }
        )
    return record


SAMPLE_STATUSES = ("success", "parse_failed", "service_failed")


def process_sample(
    raw_sample: Mapping[str, Any],
) -> Mapping[str, Any]:
    status = raw_sample.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"sample status must be one of {SAMPLE_STATUSES}")
    return raw_sample


def postprocess(text: str) -> str:
    """Two-layer post-processing matching OpenCompass.

    1. ``lcsts_postprocess``: first line, strip prefixes
       and surrounding Chinese punctuation.
    2. ``general_postprocess``: truncate at first
       newline/period/comma, remove punctuation and
       articles, collapse whitespace.
    """
    from luban_meter.benchmarking.model_service_quality.common.parsers import (
        general_postprocess,
        lcsts_postprocess,
    )

    return general_postprocess(lcsts_postprocess(text))


def score_one(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1/2/L F-measures for one pair.

    Matches OpenCompass JiebaRougeEvaluator: jieba
    tokenize then ``rouge_chinese.Rouge().get_scores``.
    """
    import jieba
    from rouge_chinese import Rouge

    pred_tokens = " ".join(jieba.cut(postprocess(prediction)))
    ref_tokens = " ".join(jieba.cut(postprocess(reference)))
    if not pred_tokens.strip():
        pred_tokens = "空"
    if not ref_tokens.strip():
        ref_tokens = "空"
    scores = Rouge().get_scores(pred_tokens, ref_tokens)
    return {
        "rouge1": scores[0]["rouge-1"]["f"],
        "rouge2": scores[0]["rouge-2"]["f"],
        "rougeL": scores[0]["rouge-l"]["f"],
    }


def process(
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    raw_status = raw_result.get("status")
    metadata = raw_result.get("metadata")
    result_meta = dict(metadata) if isinstance(metadata, Mapping) else {}

    if raw_status == "failed":
        error = raw_result.get("error") or {
            "type": "UnknownError",
            "message": "raw result failed",
        }
        result_meta.setdefault("scorer_version", SCORER_VERSION)
        return {
            "status": "failed",
            "metrics": {},
            "metadata": result_meta,
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

    samples = [process_sample(sample_record) for sample_record in raw_samples]
    scored = [
        sample_record
        for sample_record in samples
        if sample_record.get("status") in ("success", "parse_failed")
    ]
    parse_failed = sum(
        sample["status"] == "parse_failed" for sample in samples
    )
    service_failed = sum(
        1
        for sample_record in samples
        if sample_record.get("status") == "service_failed"
    )
    if not scored:
        raise ValueError("all samples failed at the service")

    rouge = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
    }
    for sample in scored:
        if sample["status"] == "parse_failed":
            continue
        scores = score_one(
            sample["prediction"],
            sample["reference"],
        )
        for key in rouge:
            rouge[key] += scores[key]

    scored_sample_count = len(scored)
    avg = {
        metric_name: score_sum / scored_sample_count
        for metric_name, score_sum in rouge.items()
    }

    result_status = "success" if service_failed == 0 else "partial_failed"
    result_meta.setdefault("scorer_version", SCORER_VERSION)
    result_meta.update(
        {
            "service_failed_samples": service_failed,
            "parse_failed_samples": parse_failed,
        }
    )
    return {
        "status": result_status,
        "metrics": {
            "task_view": {
                "lcsts": {
                    "rouge1": scalar(
                        avg["rouge1"] * 100,
                        "score",
                        scored_sample_count,
                    ),
                    "rouge2": scalar(
                        avg["rouge2"] * 100,
                        "score",
                        scored_sample_count,
                    ),
                    "rougeL": scalar(
                        avg["rougeL"] * 100,
                        "score",
                        scored_sample_count,
                    ),
                    "total_samples": scalar(len(samples), "sample"),
                    "scored_samples": scalar(scored_sample_count, "sample"),
                    "service_failed": scalar(service_failed, "sample"),
                    "parse_failed": scalar(parse_failed, "sample"),
                }
            }
        },
        "metadata": result_meta,
        "error": (
            {
                "type": "ServiceFailures",
                "message": (f"{service_failed} samples failed at the service"),
            }
            if service_failed
            else None
        ),
    }
