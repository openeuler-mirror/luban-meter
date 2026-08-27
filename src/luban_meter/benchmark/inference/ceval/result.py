"""Calculate C-Eval accuracy from raw per-sample records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from luban_meter.benchmark.inference.common.choice_result import (
    process_choice_result,
)


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    return process_choice_result(raw_result, "ceval")
