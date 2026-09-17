"""Discover scenario collectors and metric processors by category."""

from __future__ import annotations

import re
from pathlib import Path

from luban_meter.core.config import load_scenario_parameters
from luban_meter.core.errors import ConfigurationError, UnknownScenarioError
from luban_meter.core.run_contracts import (
    ResolvedRun,
    RunRequest,
    ScenarioDefinition,
)

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_CATEGORY_DESCRIPTIONS = {
    "generate": "Large-model generation benchmarks",
    "model_service_quality": "Model service quality benchmarks",
}
_CATEGORY_DIRECTORIES = {
    "generate": "generation_performance",
    "model_service_quality": "model_service_quality",
}
_SCENARIO_DIRECTORIES = {
    "serving-online": "online_serving",
    "vllm-engine-offline": "offline_vllm_engine",
    "vllm_metrics": "vllm_service_metrics",
}
_ENTRY_FILENAMES = (
    ("collect_raw.py", "calculate_metrics.py"),
    ("benchmark.py", "result.py"),
)


class ScenarioRegistry:
    def __init__(self, scenario_root: Path | None = None) -> None:
        self._scenario_root = (
            scenario_root
            if scenario_root is not None
            else Path(__file__).parents[1] / "benchmarking"
        )

    def list_categories(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(_CATEGORY_DESCRIPTIONS.items()))

    def _category_directory(self, category_name: str) -> Path:
        self._validate_name("module", category_name)
        if category_name not in _CATEGORY_DIRECTORIES:
            raise ConfigurationError(
                f"unknown functional module: {category_name}"
            )
        canonical = self._scenario_root / _CATEGORY_DIRECTORIES[category_name]
        return (
            canonical
            if canonical.is_dir()
            else self._scenario_root / category_name
        )

    @staticmethod
    def _entry_paths(scenario_directory: Path) -> tuple[Path, Path] | None:
        for collector_name, processor_name in _ENTRY_FILENAMES:
            collector_path = scenario_directory / collector_name
            processor_path = scenario_directory / processor_name
            if collector_path.is_file() and processor_path.is_file():
                return collector_path, processor_path
        return None

    def list_scenarios(self, category_name: str) -> tuple[str, ...]:
        category_directory = self._category_directory(category_name)
        logical_names = {
            directory: name
            for name, directory in _SCENARIO_DIRECTORIES.items()
        }
        scenario_names = []
        for scenario_directory in category_directory.glob("*"):
            if self._entry_paths(scenario_directory) is not None:
                scenario_names.append(
                    logical_names.get(
                        scenario_directory.name, scenario_directory.name
                    )
                )
        return tuple(sorted(scenario_names))

    def resolve(self, request: RunRequest) -> ResolvedRun:
        self._validate_name("benchmark", request.scenario_name)
        category_directory = self._category_directory(request.category_name)
        directory_name = _SCENARIO_DIRECTORIES.get(
            request.scenario_name, request.scenario_name
        )
        entry_paths = self._entry_paths(category_directory / directory_name)
        if entry_paths is None and directory_name != request.scenario_name:
            entry_paths = self._entry_paths(
                category_directory / request.scenario_name
            )
        if entry_paths is None:
            raise UnknownScenarioError(
                request.category_name, request.scenario_name
            )
        collector_path, processor_path = entry_paths
        return ResolvedRun(
            request=request,
            scenario_definition=ScenarioDefinition(
                category_name=request.category_name,
                scenario_name=request.scenario_name,
                collector_path=collector_path,
                processor_path=processor_path,
            ),
            parameters=load_scenario_parameters(request.config_path),
        )

    @staticmethod
    def _validate_name(field_name: str, value: str) -> None:
        if not _SAFE_NAME.fullmatch(value):
            raise ConfigurationError(
                f"{field_name} must match {_SAFE_NAME.pattern!r}: {value!r}"
            )
