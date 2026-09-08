"""Docker-only execution for HumanEval candidates."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from luban_meter.benchmark.inference.humaneval.sandbox_runner import RESULT_PREFIX
from luban_meter.utils.docker_sandbox import (
    DockerSandbox,
    DockerSandboxConfig,
    DockerSandboxUnavailable,
)

SandboxUnavailable = DockerSandboxUnavailable


@dataclass(frozen=True)
class SandboxConfig(DockerSandboxConfig):
    """HumanEval execution limits layered on the shared Docker sandbox."""

    timeout_seconds: float = 3.0
    startup_grace_seconds: float = 5.0
    output_limit_bytes: int = 64 * 1024

    def validate(self) -> None:
        super().validate()
        if self.timeout_seconds <= 0 or self.startup_grace_seconds <= 0:
            raise ValueError("sandbox timeouts must be positive")
        if self.output_limit_bytes < 1024:
            raise ValueError("sandbox output limit is too small")


@dataclass
class ExecutionResult:
    status: str
    passed: bool
    execution_ms: float
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DockerSandboxExecutor:
    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.sandbox = DockerSandbox(config)

    def preflight(self) -> dict[str, Any]:
        return self.sandbox.preflight()

    @staticmethod
    def _drain(
        stream: Any,
        limit: int,
        target: bytearray,
        truncated: list[bool],
    ) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            remaining = limit - len(target)
            if remaining > 0:
                target.extend(chunk[:remaining])
            if len(chunk) > max(0, remaining):
                truncated[0] = True

    def execute(self, program: str) -> ExecutionResult:
        if not isinstance(program, str) or not program:
            raise ValueError("program must be a non-empty string")
        name = f"luban-humaneval-{uuid.uuid4().hex}"
        command = self.sandbox.build_run_command(
            name=name,
            hostname="humaneval-sandbox",
        )
        payload = json.dumps(
            {
                "program": program,
                "timeout_seconds": self.config.timeout_seconds,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.sandbox.environment(),
            )
        except OSError as exc:
            return ExecutionResult(
                status="sandbox_error",
                passed=False,
                execution_ms=0.0,
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            )

        stdout = bytearray()
        stderr = bytearray()
        stdout_truncated = [False]
        stderr_truncated = [False]
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(
                target=self._drain,
                args=(
                    process.stdout,
                    self.config.output_limit_bytes,
                    stdout,
                    stdout_truncated,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=(
                    process.stderr,
                    self.config.output_limit_bytes,
                    stderr,
                    stderr_truncated,
                ),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        try:
            assert process.stdin is not None
            try:
                process.stdin.write(payload)
                process.stdin.close()
            except BrokenPipeError:
                pass
            process.wait(
                timeout=(
                    self.config.timeout_seconds + self.config.startup_grace_seconds
                )
            )
        except subprocess.TimeoutExpired:
            self.sandbox.remove_container(name)
            process.kill()
            process.wait()
            for thread in threads:
                thread.join(timeout=1)
            return ExecutionResult(
                status="timeout",
                passed=False,
                execution_ms=(time.perf_counter() - started) * 1000.0,
                stdout=stdout.decode("utf-8", "replace"),
                stderr=stderr.decode("utf-8", "replace"),
                stdout_truncated=stdout_truncated[0],
                stderr_truncated=stderr_truncated[0],
                error_type="TimeoutError",
                error_message="sandbox exceeded the host-side timeout",
            )
        for thread in threads:
            thread.join(timeout=1)

        stdout_text = stdout.decode("utf-8", "replace")
        stderr_text = stderr.decode("utf-8", "replace")
        for line in reversed(stdout_text.splitlines()):
            if not line.startswith(RESULT_PREFIX):
                continue
            try:
                result = json.loads(line[len(RESULT_PREFIX) :])
                return ExecutionResult(
                    status=str(result["status"]),
                    passed=bool(result["passed"]),
                    execution_ms=float(result.get("execution_ms") or 0.0),
                    stdout=str(result.get("stdout") or ""),
                    stderr=str(result.get("stderr") or ""),
                    stdout_truncated=bool(result.get("stdout_truncated")),
                    stderr_truncated=bool(result.get("stderr_truncated")),
                    error_type=result.get("error_type"),
                    error_message=result.get("error_message"),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                break
        return ExecutionResult(
            status="sandbox_error",
            passed=False,
            execution_ms=(time.perf_counter() - started) * 1000.0,
            stdout=stdout_text,
            stderr=stderr_text,
            stdout_truncated=stdout_truncated[0],
            stderr_truncated=stderr_truncated[0],
            error_type="SandboxProtocolError",
            error_message=(
                f"sandbox returned exit code {process.returncode} without a result"
            ),
        )
