"""Shared request, execution, and result data objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandSpec:
    argv: Sequence[str]
    cwd: Path | None = None
    timeout: float | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    category_name: str = field(metadata={"json_name": "module"})
    scenario_name: str = field(metadata={"json_name": "benchmark"})
    config_path: Path = field(metadata={"json_name": "config"})
    model_path: Path | None
    model_name: str | None
    output_dir: Path
    timeout: int = 3600
    display_name: str | None = None


@dataclass(frozen=True)
class ScenarioDefinition:
    """Convention-resolved Benchmark implementation."""

    category_name: str
    scenario_name: str
    collector_path: Path
    processor_path: Path


@dataclass(frozen=True)
class ResolvedRun:
    request: RunRequest
    scenario_definition: ScenarioDefinition
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class RawRunArtifacts:
    raw_result: Path
    stdout_log: Path
    stderr_log: Path
    artifact_dir: Path


@dataclass
class RunResult:
    schema_version: str
    run_id: str
    status: str
    category_name: str = field(metadata={"json_name": "module"})
    scenario_name: str = field(metadata={"json_name": "benchmark"})
    config_path: str = field(metadata={"json_name": "config"})
    model_info: Mapping[str, Any] = field(
        default_factory=dict, metadata={"json_name": "model"}
    )
    environment: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
