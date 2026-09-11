"""The shared v2 envelope; Benchmark metrics remain a dictionary."""

from collections.abc import Mapping
from typing import Any

RESULT_SCHEMA = "luban-meter.result/v2"
SUITE_SCHEMA = "luban-meter.suite-result/v2"
STATUSES = {"success", "partial_failed", "failed"}


def validate_result(payload: Mapping[str, Any]) -> None:
    """Validate the envelope without prescribing the metric hierarchy."""
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("只支持 luban-meter.result/v2 结果")
    for key in ("run_id", "module", "benchmark", "config"):
        if not isinstance(payload.get(key), str):
            raise ValueError(f"结果字段 {key} 必须是字符串")
    status = payload.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError("结果 status 无效")
    for key in (
        "model",
        "environment",
        "parameters",
        "metrics",
        "artifacts",
        "metadata",
    ):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"结果字段 {key} 必须是字典")
    if payload.get("error") is not None and not isinstance(
        payload["error"], Mapping
    ):
        raise ValueError("结果 error 必须是字典或 null")
