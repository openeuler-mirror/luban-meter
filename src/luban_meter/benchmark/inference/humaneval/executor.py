"""Docker-only execution for HumanEval candidates."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from luban_meter.benchmark.inference.humaneval.sandbox_runner import RESULT_PREFIX


class SandboxUnavailable(RuntimeError):
    """Raised when the configured Docker sandbox cannot be used safely."""


@dataclass(frozen=True)
class SandboxConfig:
    docker_host: str
    image: str
    docker_binary: str = "docker"
    runtime: str = "runc"
    timeout_seconds: float = 3.0
    startup_grace_seconds: float = 5.0
    memory_mb: int = 256
    cpus: float = 1.0
    pids_limit: int = 32
    tmpfs_mb: int = 64
    output_limit_bytes: int = 64 * 1024

    def validate(self) -> None:
        if not self.docker_host.startswith("unix://"):
            raise ValueError("docker_host must be a unix:// socket")
        if not self.image:
            raise ValueError("sandbox image must not be empty")
        if self.timeout_seconds <= 0 or self.startup_grace_seconds <= 0:
            raise ValueError("sandbox timeouts must be positive")
        if self.memory_mb < 32:
            raise ValueError("sandbox memory_mb must be at least 32")
        if self.cpus <= 0:
            raise ValueError("sandbox cpus must be positive")
        if self.pids_limit < 2:
            raise ValueError("sandbox pids_limit must be at least 2")
        if self.tmpfs_mb < 1 or self.output_limit_bytes < 1024:
            raise ValueError("sandbox tmpfs/output limits are too small")


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
        config.validate()
        self.config = config

    def _prefix(self) -> list[str]:
        return [
            self.config.docker_binary,
            "-H",
            self.config.docker_host,
        ]

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = os.environ.copy()
        environment.pop("DOCKER_HOST", None)
        return environment

    def _control(self, arguments: list[str], timeout: float = 15.0) -> str:
        try:
            completed = subprocess.run(
                [*self._prefix(), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"Docker command failed: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise SandboxUnavailable(f"Docker command failed: {detail}")
        return completed.stdout.strip()

    def preflight(self) -> dict[str, Any]:
        info_text = self._control(["info", "--format", "{{json .}}"])
        try:
            info = json.loads(info_text)
        except json.JSONDecodeError as exc:
            raise SandboxUnavailable("Docker info returned invalid JSON") from exc
        security_options = info.get("SecurityOptions") or []
        if not any("seccomp" in str(option) for option in security_options):
            raise SandboxUnavailable("Docker daemon must enable seccomp")
        cgroup_version = str(info.get("CgroupVersion") or "")
        if cgroup_version not in {"1", "2"}:
            raise SandboxUnavailable("Docker daemon returned no cgroup version")

        image_text = self._control(
            ["image", "inspect", "--format", "{{json .}}", self.config.image]
        )
        try:
            image = json.loads(image_text)
        except json.JSONDecodeError as exc:
            raise SandboxUnavailable(
                "sandbox image inspect returned invalid JSON"
            ) from exc
        return {
            "runtime": "docker",
            "server_version": info.get("ServerVersion"),
            "docker_root_dir": info.get("DockerRootDir"),
            "cgroup_driver": info.get("CgroupDriver"),
            "cgroup_version": cgroup_version,
            "security_options": list(security_options),
            "image": self.config.image,
            "image_id": image.get("Id"),
            "oci_runtime": self.config.runtime,
        }

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

    def _remove_container(self, name: str) -> None:
        try:
            subprocess.run(
                [*self._prefix(), "rm", "-f", name],
                check=False,
                capture_output=True,
                timeout=10,
                env=self._environment(),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def execute(self, program: str) -> ExecutionResult:
        if not isinstance(program, str) or not program:
            raise ValueError("program must be a non-empty string")
        name = f"luban-humaneval-{uuid.uuid4().hex}"
        command = [
            *self._prefix(),
            "run",
            "--interactive",
            "--rm",
            "--pull",
            "never",
            "--name",
            name,
            "--hostname",
            "humaneval-sandbox",
            "--runtime",
            self.config.runtime,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            (f"/tmp:rw,noexec,nosuid,nodev,size={self.config.tmpfs_mb}m,mode=1777"),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.config.pids_limit),
            "--memory",
            f"{self.config.memory_mb}m",
            "--memory-swap",
            f"{self.config.memory_mb}m",
            "--cpus",
            str(self.config.cpus),
            "--user",
            "65534:65534",
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "fsize=1048576:1048576",
            "--ipc",
            "none",
            "--log-driver",
            "none",
            self.config.image,
        ]
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
                env=self._environment(),
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
            self._remove_container(name)
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
