"""Configuration parameter validation shared by inference benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def positive_integer(parameters: Mapping[str, Any], name: str, default: int) -> int:
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


def string_value(parameters: Mapping[str, Any], name: str, default: str) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def non_empty_string(parameters: Mapping[str, Any], name: str, default: str) -> str:
    value = string_value(parameters, name, default)
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def boolean_value(parameters: Mapping[str, Any], name: str, default: bool) -> bool:
    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def enum_value(
    parameters: Mapping[str, Any], name: str, allowed: Sequence[str], default: str
) -> str:
    value = string_value(parameters, name, default)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}")
    return value


def string_list(parameters: Mapping[str, Any], name: str, default: Sequence[str]) -> list[str]:
    value = parameters.get(name, list(default))
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, str) for item in value)
    ):
        raise TypeError(f"{name} must be a list of strings")
    return list(value)


def fixed_value(parameters: Mapping[str, Any], name: str, expected: Any) -> Any:
    value = parameters.get(name, expected)
    if value != expected:
        raise ValueError(f"{name} is fixed to {expected!r}")
    return expected
