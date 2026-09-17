"""Compute WikiText loss metrics from raw per-sample records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.result.report_spec import bar, table

REPORT = {
    "tables": [
        table(
            "WikiText Loss",
            "/task_view/wikitext",
            {
                "/mean_loss": "Mean Loss (nats/tok)",
                "/perplexity": "Perplexity",
                "/bits_per_byte": "Bits/Byte",
                "/count": "Scored Tokens",
                "/count_bytes": "Total Bytes",
                "/total_samples": "Total Samples",
                "/scored_samples": "Scored Samples",
                "/service_failed": "Service Failed",
            },
            charts=[
                bar("/perplexity"),
                bar("/bits_per_byte"),
            ],
        )
    ]
}

SCORER_VERSION = "metrics-v1"
SAMPLE_STATUSES = ("success", "service_failed")


def scalar(
    value: float, unit: str, count: int | None = None
) -> dict[str, Any]:
    metric: dict[str, Any] = {"value": value, "unit": unit}
    if count is not None:
        metric["count"] = count
    return metric


def token_count(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def number_value(record: Mapping[str, Any], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"sample {name} must be a number")
    return float(value)


def object_list(value: Any, name: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise TypeError(f"{name} must be a list of objects")
    return list(value)


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
            "report": REPORT,
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
