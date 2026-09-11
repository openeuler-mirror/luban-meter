"""Read v2 results and extract metrics without Benchmark-specific code."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luban_meter.result.schema import (
    RESULT_SCHEMA,
    STATUSES,
    SUITE_SCHEMA,
    validate_result,
)


def numeric(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def pointer(parts: list[str]) -> str:
    return "/" + "/".join(
        p.replace("~", "~0").replace("/", "~1") for p in parts
    )


def lookup(
    value: Any,
    path: str,
    *,
    unwrap: bool = True,
    unit: str = "",
) -> tuple[Any, str]:
    """Resolve a JSON Pointer and the nearest enclosing metric unit."""
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise ValueError("指标路径必须是 JSON Pointer，例如 /score/value")
    parts = path[1:].split("/") if path else []
    for part in parts:
        if isinstance(value, Mapping):
            if isinstance(value.get("unit"), str):
                unit = value["unit"]
            if part == "count" and unit:
                unit = "count"
            value = value.get(part.replace("~1", "/").replace("~0", "~"))
        elif isinstance(value, list) and part.isdigit():
            index = int(part)
            value = value[index] if index < len(value) else None
        else:
            return None, unit
    if isinstance(value, Mapping):
        if isinstance(value.get("unit"), str):
            unit = value["unit"]
        if unwrap:
            value = value.get("value", value)
    return value, unit


def metric_rows(value: Any, parts: list[str] | None = None, unit: str = ""):
    """Yield numeric leaves, retaining nulls and omitting free-form text."""
    parts = parts or []
    if isinstance(value, Mapping):
        if isinstance(value.get("unit"), str):
            unit = value["unit"]
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            if key == "unit" and isinstance(child, str):
                continue
            child_unit = "count" if key == "count" and unit else unit
            yield from metric_rows(child, [*parts, str(key)], child_unit)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from metric_rows(child, [*parts, str(index)], unit)
    elif value is None or isinstance(value, (int, float)):
        if not isinstance(value, bool):
            yield {
                "metric": pointer(parts),
                "value": value if numeric(value) else None,
                "unit": unit,
            }


@dataclass
class Task:
    name: str
    data: dict[str, Any]
    source: Path | None
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    run_id: str
    name: str
    status: str
    source: Path
    tasks: list[Task]
    kind: str = "run"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取结果 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"结果必须是 JSON 字典：{path}")
    return value


def from_payload(
    payload: dict[str, Any],
    source: Path,
    name: str | None = None,
) -> Report:
    schema = payload.get("schema_version")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata 必须是字典")
    if schema == RESULT_SCHEMA:
        validate_result(payload)
        return Report(
            payload["run_id"],
            name or metadata.get("display_name") or payload["run_id"],
            payload["status"],
            source,
            [Task(payload["benchmark"], payload, source)],
        )
    if schema != SUITE_SCHEMA:
        raise ValueError(f"只支持 v2 结果，收到：{schema!r}")
    if not isinstance(payload.get("suite_id"), str):
        raise ValueError("suite_id 必须是字符串")
    status = payload.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError("Suite status 无效")
    entries = payload.get("tasks")
    if not isinstance(entries, list):
        raise ValueError("Suite tasks 必须是列表")
    tasks = []
    names = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Suite task 必须是字典")
        task_name = entry.get("name")
        if not isinstance(task_name, str) or task_name in names:
            raise ValueError("Suite task 名称缺失或重复")
        names.add(task_name)
        for key in ("module", "benchmark", "status"):
            if not isinstance(entry.get(key), str):
                raise ValueError(f"Suite task.{key} 必须是字符串")
        if entry["status"] not in STATUSES | {"skipped"}:
            raise ValueError("Suite task.status 无效")
        if entry.get("error") is not None and not isinstance(
            entry["error"], Mapping
        ):
            raise ValueError("Suite task.error 必须是字典或 null")
        run_id = entry.get("run_id")
        child = None
        if isinstance(run_id, str) and Path(run_id).name == run_id:
            child = source.parent / "tasks" / run_id / "result.json"
        output = entry.get("output")
        if output is None:
            data = dict(entry)
            data["metrics"] = {}
            note = (
                []
                if entry.get("status") == "skipped"
                else ["任务缺少内嵌结果"]
            )
        else:
            if not isinstance(output, dict):
                raise ValueError("Suite task.output 必须是 v2 结果字典")
            validate_result(output)
            for key in ("module", "benchmark", "run_id", "status"):
                if output[key] != entry.get(key):
                    raise ValueError(f"Suite task.output 身份不一致：{key}")
            data, note = output, []
        tasks.append(Task(task_name, data, child, note))
    return Report(
        payload["suite_id"],
        name or metadata.get("display_name") or payload["suite_id"],
        payload["status"],
        source,
        tasks,
        "suite",
    )


def load_report(path: Path, name: str | None = None) -> Report:
    path = path.expanduser().resolve()
    if path.is_dir():
        matches = [
            p
            for p in (path / "suite_result.json", path / "result.json")
            if p.is_file()
        ]
        if len(matches) != 1:
            raise ValueError(f"请指定唯一结果文件：{path}")
        path = matches[0]
    return from_payload(read_json(path), path, name)
