"""Data objects for one vendor-scoped Suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    vendor: str
    source: Path
    tasks: tuple[SuiteTask, ...]


@dataclass(frozen=True)
class SuiteRequest:
    suite_id: str
    vendor: str
    suite: str
    model_path: Path | None
    model_name: str | None
    output_dir: Path
    timeout: int = 3600
    fail_fast: bool = False


@dataclass(frozen=True)
class SuiteTaskResult:
    name: str
    module: str
    benchmark: str
    status: str
    run_id: str | None = None
    result: str | None = None


@dataclass(frozen=True)
class SuiteResult:
    schema_version: str
    suite_id: str
    name: str
    vendor: str
    status: str
    tasks: tuple[SuiteTaskResult, ...]
