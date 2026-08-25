"""Collect CMMLU accuracy samples from an OpenAI-compatible service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from luban_meter.benchmark.inference.common import choice
from luban_meter.benchmark.inference.common.choice import (  # noqa: F401
    ChoiceSpec,
    collect_sample,
    generate_answer,
    load_request,
    probe_logprobs_support,
    score_choices,
)
from luban_meter.benchmark.inference.common.prompts import (  # noqa: F401
    CHOICE_LETTERS,
)

SPEC = ChoiceSpec(
    benchmark="cmmlu",
    measurement="cmmlu_choice_accuracy_online_service",
    dataset_label="CMMLU",
)
MEASUREMENT = SPEC.measurement
PROTOCOL = choice.PROTOCOL
SCORER_VERSION = choice.SCORER_VERSION


def parse_args():
    return choice.parse_args("CMMLU inference benchmark")


def validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return choice.validate_choice_parameters(parameters, SPEC)


def failure_result(error: Exception) -> dict[str, Any]:
    return choice.choice_failure_result(error, SPEC)


def run_benchmark(request: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    return choice.run_choice_benchmark(request, parameters, SPEC)


def main() -> None:
    choice.choice_main(SPEC, "CMMLU inference benchmark")


if __name__ == "__main__":
    main()
