"""Data objects for one Benchmark Suite."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SuiteTask:
    name: str
    module: str
    benchmark: str
    config: Path
    timeout: int | None = None


@dataclass(frozen=True)
class SuiteDefinition:
    name: str
    source: Path
    tasks: tuple[SuiteTask, ...]


@dataclass(frozen=True)
class SuiteRequest:
    suite_id: str
    suite: str
    model_path: Path | None
    model_name: str | None
    output_dir: Path
    timeout: int = 3600
    fail_fast: bool = False
    task_configs: Mapping[str, Path] = field(default_factory=dict)
    display_name: str | None = None


@dataclass(frozen=True)
class SuiteTaskResult:
    name: str
    module: str
    benchmark: str
    status: str
    run_id: str | None = None
    result: str | None = None
    output: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SuiteResult:
    schema_version: str
    suite_id: str
    name: str
    status: str
    tasks: tuple[SuiteTaskResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
