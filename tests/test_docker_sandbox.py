"""Tests for reusable Docker sandbox construction."""

from __future__ import annotations

from luban_meter.utils.docker_sandbox import (
    DockerSandbox,
    DockerSandboxConfig,
)


def sandbox() -> DockerSandbox:
    return DockerSandbox(
        DockerSandboxConfig(
            docker_host="unix:///test/docker.sock",
            image="test-sandbox:latest",
        )
    )


def test_build_run_command_uses_hardened_defaults() -> None:
    command = sandbox().build_run_command(
        name="test-container",
        hostname="test-sandbox",
    )

    assert command[:4] == [
        "docker",
        "-H",
        "unix:///test/docker.sock",
        "run",
    ]
    assert command[-1] == "test-sandbox:latest"
    for option in (
        "--interactive",
        "--rm",
        "--network",
        "--read-only",
        "--tmpfs",
        "--cap-drop",
        "--security-opt",
        "--pids-limit",
        "--memory",
        "--memory-swap",
        "--cpus",
        "--user",
        "--ulimit",
        "--ipc",
        "--log-driver",
    ):
        assert option in command
    assert "--privileged" not in command
    assert "--volume" not in command
    assert "-v" not in command


def test_environment_ignores_ambient_docker_host(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://unsafe:2375")

    environment = sandbox().environment()

    assert "DOCKER_HOST" not in environment
