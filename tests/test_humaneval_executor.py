"""Unit and optional integration tests for the HumanEval Docker sandbox."""

from __future__ import annotations

import io
import json
import os
import subprocess
from typing import Any

import pytest

from luban_meter.benchmark.inference.humaneval.executor import (
    DockerSandboxExecutor,
    SandboxConfig,
    SandboxUnavailable,
)
from luban_meter.benchmark.inference.humaneval.sandbox_runner import (
    RESULT_PREFIX,
    execute_payload,
)


def sandbox_config() -> SandboxConfig:
    return SandboxConfig(
        docker_host="unix:///test/docker.sock",
        image="test-humaneval:latest",
    )


def test_sandbox_runner_outcomes() -> None:
    passed = execute_payload({"program": "assert 1 + 1 == 2", "timeout_seconds": 1})
    assert passed["status"] == "passed"
    assert passed["passed"] is True

    failed = execute_payload({"program": "assert False", "timeout_seconds": 1})
    assert failed["status"] == "failed_test"

    syntax = execute_payload(
        {"program": "def broken(:\n    pass", "timeout_seconds": 1}
    )
    assert syntax["status"] == "syntax_error"

    runtime = execute_payload(
        {"program": "raise RuntimeError('x')", "timeout_seconds": 1}
    )
    assert runtime["status"] == "runtime_error"


def test_sandbox_runner_timeout_and_output_limit() -> None:
    timeout = execute_payload(
        {"program": "while True:\n    pass", "timeout_seconds": 0.01}
    )
    assert timeout["status"] == "timeout"
    output = execute_payload({"program": "print('x' * 20000)", "timeout_seconds": 1})
    assert output["status"] == "passed"
    assert output["stdout_truncated"] is True
    assert len(output["stdout"]) <= 8192


def test_sandbox_config_rejects_unsafe_host() -> None:
    with pytest.raises(ValueError):
        SandboxConfig(
            docker_host="tcp://remote:2375",
            image="test",
        ).validate()


def test_preflight_requires_seccomp(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = DockerSandboxExecutor(sandbox_config())

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        if "info" in command:
            stdout = json.dumps(
                {
                    "ServerVersion": "24.0.9",
                    "CgroupVersion": "1",
                    "CgroupDriver": "cgroupfs",
                    "SecurityOptions": [],
                }
            )
        else:
            stdout = json.dumps({"Id": "sha256:test"})
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SandboxUnavailable, match="seccomp"):
        executor.preflight()


class _FakeProcess:
    def __init__(self, command: list[str], *_args: Any, **_kwargs: Any) -> None:
        self.command = command
        self.stdin = io.BytesIO()
        payload = {
            "status": "passed",
            "passed": True,
            "execution_ms": 1.25,
            "stdout": "",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "error_type": None,
            "error_message": None,
        }
        self.stdout = io.BytesIO(
            (RESULT_PREFIX + json.dumps(payload) + "\n").encode("utf-8")
        )
        self.stderr = io.BytesIO()
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def test_executor_uses_hardened_docker_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeProcess] = []

    def fake_popen(command: list[str], *args: Any, **kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(command, *args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    result = DockerSandboxExecutor(sandbox_config()).execute("assert True")
    assert result.passed is True
    command = created[0].command
    for option in (
        "--interactive",
        "--network",
        "--read-only",
        "--cap-drop",
        "--security-opt",
        "--pids-limit",
        "--memory",
        "--cpus",
        "--user",
        "--ipc",
        "--log-driver",
    ):
        assert option in command
    assert "--privileged" not in command
    assert "--volume" not in command
    assert "-v" not in command
    assert command[-1] == "test-humaneval:latest"


@pytest.mark.skipif(
    os.environ.get("LUBAN_METER_RUN_DOCKER_TESTS") != "1",
    reason="set LUBAN_METER_RUN_DOCKER_TESTS=1 for the real sandbox test",
)
def test_real_docker_sandbox() -> None:
    host = os.environ.get("LUBAN_METER_DOCKER_HOST", "unix:///var/run/docker.sock")
    image = os.environ.get(
        "LUBAN_METER_HUMANEVAL_IMAGE",
        "luban-meter-humaneval-sandbox:v1",
    )
    executor = DockerSandboxExecutor(SandboxConfig(docker_host=host, image=image))
    environment = executor.preflight()
    assert environment["image_id"]
    assert executor.execute("assert 2 + 2 == 4").passed is True
