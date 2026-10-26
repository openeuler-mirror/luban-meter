"""Shared readers for persisted results; no metric recalculation."""

import json
from pathlib import Path

STATUSES = {
    "success",
    "partial_failed",
    "failed",
    "interrupted",
    "skipped",
    "discarded",
}


def validate_result(payload):
    if not isinstance(payload, dict) or "schema_version" in payload:
        raise ValueError(
            "unsupported legacy/versioned result; expected an unversioned "
            "run/task result. Keep the original JSON; "
            "no metrics were converted"
        )
    if payload.get("kind") not in ("run", "task"):
        raise ValueError("result.kind must be run or task")
    if payload.get("status") not in STATUSES:
        raise ValueError("invalid result status")
    tasks = payload.get("tasks") if payload["kind"] == "run" else [payload]
    if not isinstance(tasks, list):
        raise ValueError("result.tasks must be a list")
    for task in tasks:
        if not isinstance(task, dict) or task.get("status") not in STATUSES:
            raise ValueError("invalid task result")
        if not isinstance(task.get("cases"), list):
            raise ValueError("task.cases must be a list")
        if any(
            not isinstance(task.get(key), str) for key in ("name", "category")
        ):
            raise ValueError("task name and category must be strings")
        for case in task["cases"]:
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("id"), str)
                or case.get("status") not in STATUSES
            ):
                raise ValueError("invalid Case result")
            if not isinstance(case.get("metrics", {}), dict):
                raise ValueError("Case metrics must be an object")
    return tasks


def metric_rows(value, path="", unit="", count=None):
    if isinstance(value, dict):
        unit = value.get("unit", unit)
        count = value.get("sample_count", value.get("count", count))
        if "value" in value:
            yield {
                "metric": path,
                "value": value["value"],
                "unit": unit,
                "count": count,
            }
        else:
            for name, item in value.items():
                if name not in ("unit", "count", "sample_count", "weight_sum"):
                    yield from metric_rows(item, f"{path}/{name}", unit, count)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from metric_rows(item, f"{path}/{index}", unit, count)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield {"metric": path, "value": value, "unit": unit, "count": count}


def load_result(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_result(payload)
    return payload


def metric_groups(value, path="", unit="", count=None):
    """Read scalar/distribution nodes with the HTML inheritance rules."""
    if isinstance(value, dict):
        unit = value.get("unit", unit)
        count = value.get("sample_count", value.get("count", count))
        if "value" in value or any(
            k in value for k in ("mean", "median", "p50", "p95")
        ):
            node = dict(value)
            node.setdefault("unit", unit)
            if count is not None:
                node.setdefault("count", count)
            yield path, node
        else:
            for key, item in value.items():
                if key not in (
                    "unit", "count", "sample_count", "weight_sum",
                    "lengths", "environment", "metadata", "error", "reason",
                    "condition",
                ):
                    yield from metric_groups(
                        item, f"{path}/{key}".strip("/"), unit, count
                    )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from metric_groups(item, f"{path}/{index}", unit, count)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, {"value": value, "unit": unit, "count": count}
