"""Calculate CMMLU accuracy from raw per-sample records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from luban_meter.benchmarking.model_service_quality.common import choice_result


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    return choice_result.process_choice_result(raw_result, "cmmlu")
