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


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    module: str
    benchmark: str
    config: Path
    model_path: Path | None
    model_name: str | None
    output_dir: Path
    timeout: int = 3600


@dataclass(frozen=True)
class BenchmarkSpec:
    """Convention-resolved Benchmark implementation."""

    module: str
    benchmark: str
    benchmark_entry: Path
    result_handler: Path


@dataclass(frozen=True)
class ResolvedRun:
    request: RunRequest
    benchmark: BenchmarkSpec
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class RawRunArtifacts:
    raw_result: Path
    stdout_log: Path
    stderr_log: Path
    artifact_dir: Path


@dataclass
class BenchmarkResult:
    schema_version: str
    run_id: str
    status: str
    module: str
    benchmark: str
    config: str
    model: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
