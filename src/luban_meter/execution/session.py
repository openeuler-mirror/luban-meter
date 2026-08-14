"""ExecutionSession protocol and shared local artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from luban_meter.core.models import (
    RawRunArtifacts,
    ResolvedRun,
)
from luban_meter.utils.json_io import to_jsonable, write_json_atomic


class ExecutionSession(Protocol):
    def execute(self, run: ResolvedRun) -> RawRunArtifacts: ...

    def close(self) -> None: ...


def prepare_run_directory(
    run: ResolvedRun,
) -> tuple[Path, Path, Path]:
    run_dir = run.request.output_dir / run.request.run_id
    raw_dir = run_dir / "raw"
    artifact_dir = raw_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        run_dir / "request.json",
        {
            "request": to_jsonable(run.request),
            "parameters": to_jsonable(run.parameters),
        },
    )
    return run_dir, raw_dir, artifact_dir


def write_command_logs(raw_dir: Path, stdout: str, stderr: str) -> None:
    (raw_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (raw_dir / "stderr.log").write_text(stderr, encoding="utf-8")


def collected_artifacts(raw_dir: Path, artifact_dir: Path) -> RawRunArtifacts:
    return RawRunArtifacts(
        raw_result=raw_dir / "raw_result.json",
        stdout_log=raw_dir / "stdout.log",
        stderr_log=raw_dir / "stderr.log",
        artifact_dir=artifact_dir,
    )
