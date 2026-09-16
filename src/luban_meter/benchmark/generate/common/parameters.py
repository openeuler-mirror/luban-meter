"""Parameter parsing helpers for generate benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def positive_integer(
    parameters: Mapping[str, Any], name: str, default: int
) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def non_negative_integer(
    parameters: Mapping[str, Any], name: str, default: int
) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def positive_integer_list(
    parameters: Mapping[str, Any], name: str, default: Sequence[int]
) -> list[int]:
    value = parameters.get(name, default)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in value
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty list of positive integers"
        )
    result = list(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def positive_number_list(
    parameters: Mapping[str, Any], name: str, default: Sequence[float]
) -> list[float]:
    value = parameters.get(name, default)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or item <= 0
            for item in value
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty list of positive numbers"
        )
    result = [float(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def positive_number(
    parameters: Mapping[str, Any], name: str, default: float
) -> float:
    value = parameters.get(name, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def string_value(
    parameters: Mapping[str, Any], name: str, default: str
) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def fixed_value(
    parameters: Mapping[str, Any], name: str, expected: Any
) -> Any:
    value = parameters.get(name, expected)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r} for exact-length cases")
    return value


def optional_positive_number(
    parameters: Mapping[str, Any], name: str
) -> float | None:
    value = parameters.get(name)
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)
