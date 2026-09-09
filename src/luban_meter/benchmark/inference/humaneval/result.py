"""Calculate HumanEval pass@1 from raw per-candidate records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.benchmark.inference.common.metrics import pass_at_k

SCORER_VERSION = "humaneval-scorer-v1"
SAMPLE_STATUSES = ("success", "parse_failed", "service_failed", "sandbox_failed")
EXECUTION_STATUSES = (
    "passed",
    "failed_test",
    "syntax_error",
    "runtime_error",
    "timeout",
    "sandbox_error",
)


def scalar(value: float, unit: str, count: int | None = None) -> dict[str, Any]:
    metric: dict[str, Any] = {"value": value, "unit": unit}
    if count is not None:
        metric["count"] = count
    return metric


def object_list(value: Any, name: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise TypeError(f"{name} must be a list of objects")
    return list(value)


def process_sample(raw_sample: Mapping[str, Any]) -> dict[str, Any]:
    task_id = raw_sample.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("sample task_id must be a non-empty string")
    sample_id = raw_sample.get("sample_id")
    if sample_id != 0:
        raise ValueError("pass@1 requires sample_id=0 for every task")
    status = raw_sample.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"sample status must be one of {SAMPLE_STATUSES}")
    passed = raw_sample.get("passed")
    if not isinstance(passed, bool):
        raise TypeError("sample passed must be a boolean")
    execution_status = raw_sample.get("execution_status")
    if execution_status is not None and execution_status not in EXECUTION_STATUSES:
        raise ValueError(
            f"execution_status must be one of {EXECUTION_STATUSES} or null"
        )
    if status != "success" and passed:
        raise ValueError("failed collection records cannot be marked passed")
    if status == "success" and execution_status is None:
        raise ValueError("successful collection records need execution_status")
    return {
        "task_id": task_id,
        "status": status,
        "execution_status": execution_status,
        "passed": passed,
    }


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_status = raw_result.get("status")
    metadata = raw_result.get("metadata")
    result_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    if raw_status == "failed":
        result_metadata.setdefault("scorer_version", SCORER_VERSION)
        return {
            "status": "failed",
            "metrics": {},
            "metadata": result_metadata,
            "error": raw_result.get("error")
            or {
                "type": "UnknownError",
                "message": "raw result failed without error detail",
            },
        }
    if raw_status != "success":
        raise ValueError(f"raw status must be success or failed: {raw_status}")

    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")
    raw_samples = object_list(raw_metrics.get("samples"), "raw samples")
    if not raw_samples:
        raise ValueError("raw samples must not be empty")
    samples = [process_sample(sample) for sample in raw_samples]
    task_ids = [sample["task_id"] for sample in samples]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("pass@1 requires exactly one record per task_id")

    total = len(samples)
    passed = sum(1 for sample in samples if sample["passed"])
    pass_at_1 = (
        sum(pass_at_k(1, int(sample["passed"]), 1) for sample in samples) / total
    )
    parse_failed = sum(1 for sample in samples if sample["status"] == "parse_failed")
    service_failed = sum(
        1 for sample in samples if sample["status"] == "service_failed"
    )
    sandbox_failed = sum(
        1 for sample in samples if sample["status"] == "sandbox_failed"
    )
    timeouts = sum(1 for sample in samples if sample["execution_status"] == "timeout")
    infrastructure_failed = service_failed + sandbox_failed
    if infrastructure_failed == total:
        result_status = "failed"
    elif infrastructure_failed:
        result_status = "partial_failed"
    else:
        result_status = "success"

    result_metadata.setdefault("scorer_version", SCORER_VERSION)
    result_metadata.update(
        {
            "passed_samples": passed,
            "parse_failed_samples": parse_failed,
            "service_failed_samples": service_failed,
            "sandbox_failed_samples": sandbox_failed,
        }
    )
    error = None
    if infrastructure_failed:
        error = {
            "type": "InfrastructureFailures",
            "message": (
                f"{service_failed} service and {sandbox_failed} sandbox "
                "failures were counted as failed candidates"
            ),
        }
    return {
        "status": result_status,
        "metrics": {
            "task_view": {
                "humaneval": {
                    "pass_at_1": scalar(pass_at_1, "ratio", total),
                    "total_tasks": scalar(total, "sample"),
                    "passed": scalar(passed, "sample"),
                    "parse_failed": scalar(parse_failed, "sample"),
                    "service_failed": scalar(service_failed, "sample"),
                    "sandbox_failed": scalar(sandbox_failed, "sample"),
                    "execution_timeout": scalar(timeouts, "sample"),
                }
            }
        },
        "metadata": result_metadata,
        "error": error,
    }
