"""Collect CMMLU accuracy samples from an OpenAI-compatible service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from luban_meter.benchmarking.model_service_quality.common import choice
from luban_meter.benchmarking.model_service_quality.common.choice import (  # noqa: F401
    ChoiceBenchmarkDefinition,
    collect_sample,
    generate_answer,
    load_request,
    probe_logprobs_support,
    score_choices,
)
from luban_meter.benchmarking.model_service_quality.common.prompts import (  # noqa: F401
    CHOICE_LETTERS,
)

SPEC = ChoiceBenchmarkDefinition(
    benchmark_name="cmmlu",
    measurement="cmmlu_choice_accuracy_online_service",
    dataset_label="CMMLU",
)
MEASUREMENT = SPEC.measurement
PROTOCOL = choice.PROTOCOL
SCORER_VERSION = choice.SCORER_VERSION


def parse_args():
    return choice.parse_args("CMMLU model service quality benchmark")


def validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return choice.validate_choice_parameters(parameters, SPEC)


def failure_result(error: Exception) -> dict[str, Any]:
    return choice.build_choice_failure_result(error, SPEC)


def collect_raw_result(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    return choice.collect_choice_raw_result(request, parameters, SPEC)


def main() -> None:
    choice.run_choice_cli(SPEC, "CMMLU model service quality benchmark")


if __name__ == "__main__":
    main()
