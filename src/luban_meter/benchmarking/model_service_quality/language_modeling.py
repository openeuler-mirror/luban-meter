"""Language model collection and scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.benchmarking.model_service_quality.common import metrics
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
    string_value,
)
from luban_meter.benchmarking.model_service_quality.common.prompts import (
    render_wikitext_prompt,
    validate_prompt_version,
)
from luban_meter.benchmarking.model_service_quality.common.records import (
    number_value,
    object_list,
    scalar,
    token_count,
)
from luban_meter.log import get_logger

logger = get_logger(__name__)


MEASUREMENT = "wikitext_logloss_online_service"
PROTOCOL = "openai_compatible_completions_logprobs"
SCORER_VERSION = "metrics-v1"


def _nullable_positive_integer(
    parameters: Mapping[str, Any], name: str
) -> int | None:
    """Accept None (full load) or a positive integer."""
    value = parameters.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be null or a positive integer")
    return value


def _nullable_stride(
    parameters: Mapping[str, Any],
    max_context_length: int,
) -> int | None:
    """Accept None or int in [1, max_context_length]."""
    value = parameters.get("stride")
    if value is None:
        return None
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > max_context_length
    ):
        raise ValueError(
            f"stride must be null or in [1, {max_context_length}]"
        )
    return value


def validate_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    max_context_length = positive_integer(
        parameters, "max_context_length", 2048
    )
    config: dict[str, Any] = {
        "service_url": string_value(
            parameters,
            "service_url",
            "http://127.0.0.1:8000",
        ),
        "api_key": string_value(parameters, "api_key", ""),
        "request_timeout": positive_number(parameters, "request_timeout", 60),
        "dataset_path": string_value(parameters, "dataset_path", ""),
        "split": string_value(parameters, "split", "validation"),
        "max_samples": _nullable_positive_integer(parameters, "max_samples"),
        "shuffle": boolean_value(parameters, "shuffle", False),
        "seed": non_negative_integer(parameters, "seed", 42),
        "eval_mode": enum_value(parameters, "eval_mode", ("loss",), "loss"),
        "prompt_format": enum_value(
            parameters, "prompt_format", ("base",), "base"
        ),
        "prompt_version": string_value(
            parameters, "prompt_version", "wikitext-v1"
        ),
        "max_context_length": max_context_length,
        "stride": _nullable_stride(parameters, max_context_length),
        "max_concurrency": positive_integer(parameters, "max_concurrency", 8),
    }
    fixed_value(parameters, "temperature", 0.0)
    fixed_value(parameters, "few_shot", 0)
    if not config["dataset_path"]:
        raise ValueError("dataset_path must not be empty")
    validate_prompt_version("wikitext", config["prompt_version"])
    return config


def validate_records(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate wikitext dataset records before scoring."""
    seen_ids: set[str] = set()
    for record_index, dataset_record in enumerate(records):
        record_id = dataset_record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"record {record_index}: id must be non-empty")
        if record_id in seen_ids:
            raise ValueError(
                f"record {record_index}: duplicate id {record_id!r}"
            )
        seen_ids.add(record_id)
        text = dataset_record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(
                f"record {record_index} ({record_id}): text must be "
                "a non-empty string"
            )
        raw_text = dataset_record.get("raw_text")
        if not isinstance(raw_text, str) or not raw_text:
            raise ValueError(
                f"record {record_index} ({record_id}): raw_text must be "
                "a non-empty string"
            )
        for field in ("bytes", "words"):
            value = dataset_record.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"record {record_index} ({record_id}): {field} must "
                    "be a non-negative integer"
                )


def _find_char_position(
    client: OpenAIClient,
    text: str,
    target: int,
    lower_character_bound: int = 0,
    upper_character_bound: int | None = None,
) -> int:
    """Binary search for smallest c where
    len(tokenize(text[:c])) >= target."""
    if target <= 0:
        return 0
    if upper_character_bound is None:
        upper_character_bound = len(text)
    while lower_character_bound < upper_character_bound:
        middle_character_index = (
            lower_character_bound + upper_character_bound
        ) // 2
        prefix_token_count = len(
            client.tokenize(text[:middle_character_index])
        )
        if prefix_token_count < target:
            lower_character_bound = middle_character_index + 1
        else:
            upper_character_bound = middle_character_index
    return lower_character_bound


def compute_windows(
    client: OpenAIClient,
    text: str,
    all_tokens: list[int],
    max_context_length: int,
    stride: int,
) -> list[tuple[str, str]]:
    """Split text into (context, prediction) pairs.

    Window 0: predict tokens[0:MCL], context = "".
    Window k (k >= 1): predict
        tokens[base:base+stride], context = preceding
        min(MCL, base) tokens.
    Each token is predicted exactly once.
    """
    token_count = len(all_tokens)
    if token_count <= max_context_length:
        return [("", text)]

    boundaries: set[int] = {0, max_context_length}
    base = max_context_length
    while base < token_count:
        chunk = min(stride, token_count - base)
        context_start_index = max(0, base - max_context_length)
        boundaries.update({context_start_index, base, base + chunk})
        base += stride

    sorted_boundaries = sorted(boundaries)
    character_positions: dict[int, int] = {}
    lower_character_bound = 0
    for token_index in sorted_boundaries:
        if token_index <= 0:
            character_positions[token_index] = 0
        elif token_index >= token_count:
            character_positions[token_index] = len(text)
        else:
            character_position = _find_char_position(
                client,
                text,
                token_index,
                lower_character_bound=lower_character_bound,
            )
            character_positions[token_index] = character_position
            lower_character_bound = character_position

    windows: list[tuple[str, str]] = []
    windows.append(("", text[: character_positions[max_context_length]]))

    base = max_context_length
    while base < token_count:
        chunk = min(stride, token_count - base)
        context_start_index = max(0, base - max_context_length)
        context_text = text[
            character_positions[context_start_index] : character_positions[
                base
            ]
        ]
        prediction_text = text[
            character_positions[base] : character_positions[base + chunk]
        ]
        windows.append((context_text, prediction_text))
        base += stride

    return windows


def score_window(
    client: OpenAIClient,
    context_text: str,
    prediction_text: str,
) -> dict[str, Any]:
    """Score one window via echo+logprobs offset-slicing."""
    prompt_count = len(client.tokenize(context_text)) if context_text else 0
    combined = context_text + prediction_text
    full_count = len(client.tokenize(combined))
    response = client.completion_logprobs(combined)
    logprobs = response["token_logprobs"]
    scored = [
        token_logprob
        for token_logprob in logprobs[prompt_count:full_count]
        if token_logprob is not None
    ]
    return {
        "context": context_text,
        "prediction": prediction_text,
        "sum_logprob": sum(scored) if scored else 0.0,
        "token_count": len(scored),
        "latency_ms": response.get("latency_ms", 0.0),
        "input_tokens": (response.get("input_tokens") or 0),
        "status": "success",
        "error": None,
    }


def collect_sample(
    client: OpenAIClient,
    sample: Mapping[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Score a single document with rolling windows."""
    sample_id = str(sample.get("id", ""))
    text = render_wikitext_prompt(sample)
    raw_text = str(sample.get("raw_text", ""))
    byte_count = int(sample.get("bytes", 0))
    word_count = int(sample.get("words", 0))
    max_context_length = config["max_context_length"]
    stride = config["stride"] or max_context_length

    record: dict[str, Any] = {
        "id": sample_id,
        "mode": "loss",
        "prompt_format": "base",
        "prompt_version": config["prompt_version"],
        "text": text,
        "raw_text": raw_text,
        "bytes": byte_count,
        "words": word_count,
        "windows": [],
        "sum_logprob": 0.0,
        "token_count": 0,
        "mean_loss": None,
        "latency_ms": 0.0,
        "input_tokens": 0,
        "status": "service_failed",
        "error": None,
    }
    try:
        all_tokens = client.tokenize(text)
        text_windows = compute_windows(
            client, text, all_tokens, max_context_length, stride
        )
        document_logprob_sum = 0.0
        document_token_count = 0
        document_latency_ms = 0.0
        document_input_tokens = 0
        for context_text, prediction_text in text_windows:
            window_result = score_window(client, context_text, prediction_text)
            record["windows"].append(window_result)
            document_logprob_sum += window_result["sum_logprob"]
            document_token_count += window_result["token_count"]
            document_latency_ms += window_result["latency_ms"]
            document_input_tokens += window_result["input_tokens"]
        record["sum_logprob"] = document_logprob_sum
        record["token_count"] = document_token_count
        if document_token_count > 0:
            record["mean_loss"] = metrics.mean_loss(
                document_logprob_sum, document_token_count
            )
        record["latency_ms"] = document_latency_ms
        record["input_tokens"] = document_input_tokens
        record["status"] = "success"
    except ServiceError as exc:
        logger.exception("Model service failed")
        record["error"] = str(exc)
    return record


def probe_logprobs_support(
    client: OpenAIClient,
) -> None:
    """Fail fast when the service cannot echo logprobs."""
    client.completion_logprobs("A")


SAMPLE_STATUSES = ("success", "service_failed")


def process_sample(
    raw_sample: Mapping[str, Any],
) -> dict[str, Any]:
    status = raw_sample.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"sample status must be one of {SAMPLE_STATUSES}")
    if status == "success":
        number_value(raw_sample, "sum_logprob")
        token_count(raw_sample, "token_count")
        token_count(raw_sample, "bytes")
        token_count(raw_sample, "words")
    return {
        "status": status,
        "sum_logprob": (
            number_value(raw_sample, "sum_logprob")
            if status == "success"
            else 0.0
        ),
        "token_count": (
            token_count(raw_sample, "token_count")
            if status == "success"
            else 0
        ),
        "bytes": (
            token_count(raw_sample, "bytes") if status == "success" else 0
        ),
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
        if sample_record["status"] == "success"
    ]
    service_failed = sum(
        1
        for sample_record in samples
        if sample_record["status"] == "service_failed"
    )
    if not scored:
        raise ValueError("all samples failed at the service")

    total_logprob = sum(
        sample_record["sum_logprob"] for sample_record in scored
    )
    total_token_count = sum(
        sample_record["token_count"] for sample_record in scored
    )
    total_byte_count = sum(sample_record["bytes"] for sample_record in scored)

    if total_token_count <= 0:
        raise ValueError("total scored tokens must be positive")
    mean_token_loss = -total_logprob / total_token_count
    ppl = math.exp(mean_token_loss)
    bpb = (
        -total_logprob / total_byte_count / math.log(2)
        if total_byte_count > 0
        else 0.0
    )

    result_status = "success" if service_failed == 0 else "partial_failed"
    result_meta.setdefault("scorer_version", SCORER_VERSION)
    result_meta.update(
        {
            "service_failed_samples": service_failed,
        }
    )
    return {
        "status": result_status,
        "metrics": {
            "task_view": {
                "wikitext": {
                    "mean_loss": scalar(
                        mean_token_loss, "nats/token", total_token_count
                    ),
                    "perplexity": scalar(ppl, "ppl", total_token_count),
                    "bits_per_byte": scalar(
                        bpb, "bits/byte", total_byte_count
                    ),
                    "count": scalar(total_token_count, "token"),
                    "count_bytes": scalar(total_byte_count, "byte"),
                    "total_samples": scalar(len(samples), "sample"),
                    "scored_samples": scalar(len(scored), "sample"),
                    "service_failed": scalar(service_failed, "sample"),
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
