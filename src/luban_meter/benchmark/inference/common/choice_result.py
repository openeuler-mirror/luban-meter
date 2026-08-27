"""Shared accuracy aggregation for four-choice benchmark raw results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCORER_VERSION = "metrics-v1"
SAMPLE_STATUSES = ("success", "parse_failed", "service_failed")


def scalar(value: float, unit: str, count: int | None = None) -> dict[str, Any]:
    metric: dict[str, Any] = {"value": value, "unit": unit}
    if count is not None:
        metric["count"] = count
    return metric


def token_count(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def object_list(value: Any, name: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise TypeError(f"{name} must be a list of objects")
    return list(value)


def process_sample(raw_sample: Mapping[str, Any]) -> dict[str, Any]:
    status = raw_sample.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"sample status must be one of {SAMPLE_STATUSES}")
    reference = raw_sample.get("reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError("sample reference must be a non-empty string")
    prediction = raw_sample.get("prediction")
    if status in ("success", "parse_failed"):
        if status == "success" and not isinstance(prediction, str):
            raise ValueError("successful sample prediction must be a string")
        token_count(raw_sample, "input_tokens")
        token_count(raw_sample, "output_tokens")
        correct = raw_sample.get("correct")
        if correct is not None and not isinstance(correct, bool):
            raise TypeError("sample correct must be a boolean")
    choice_scores = raw_sample.get("choice_scores")
    if choice_scores is not None and not isinstance(choice_scores, Mapping):
        raise TypeError("choice_scores must be an object")
    return {
        "subject": raw_sample.get("subject") or "unknown",
        "status": status,
        "correct": bool(raw_sample.get("correct")) if status == "success" else False,
    }


def accuracy_metric(correct: int, total: int) -> dict[str, Any]:
    return scalar(correct / total, "ratio", total)


def process_choice_result(
    raw_result: Mapping[str, Any], benchmark: str
) -> dict[str, Any]:
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
    scored = [sample for sample in samples if sample["status"] != "service_failed"]
    parse_failed = sum(1 for sample in samples if sample["status"] == "parse_failed")
    service_failed = sum(
        1 for sample in samples if sample["status"] == "service_failed"
    )
    if not scored:
        raise ValueError("all samples failed at the service; nothing to score")

    correct = sum(1 for sample in scored if sample["correct"])
    by_subject: dict[str, dict[str, int]] = {}
    for sample in scored:
        bucket = by_subject.setdefault(
            str(sample["subject"]), {"correct": 0, "total": 0}
        )
        bucket["total"] += 1
        if sample["correct"]:
            bucket["correct"] += 1

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
                benchmark: {
                    "accuracy": accuracy_metric(correct, len(scored)),
                    "accuracy_by_subject": {
                        subject: accuracy_metric(bucket["correct"], bucket["total"])
                        for subject, bucket in sorted(by_subject.items())
                    },
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
