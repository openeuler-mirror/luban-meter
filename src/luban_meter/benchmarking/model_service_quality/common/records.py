"""Shared quality metric fields and raw-record validation."""

from collections.abc import Mapping, Sequence
from typing import Any


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
