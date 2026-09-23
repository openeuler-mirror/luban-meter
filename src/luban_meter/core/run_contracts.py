"""Shared request, execution, and result data objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


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
