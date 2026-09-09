"""Reusable helpers for constructing hardened Docker sandboxes."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any


class DockerSandboxUnavailable(RuntimeError):
    """Raised when a configured Docker sandbox cannot be used safely."""


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Business-neutral Docker sandbox configuration."""

    docker_host: str
    image: str
    docker_binary: str = "docker"
    runtime: str = "runc"
    memory_mb: int = 256
    cpus: float = 1.0
    pids_limit: int = 32
    tmpfs_mb: int = 64

    def validate(self) -> None:
        if not self.docker_host.startswith("unix://"):
            raise ValueError("docker_host must be a unix:// socket")
        if not self.image:
            raise ValueError("sandbox image must not be empty")
        if self.memory_mb < 32:
            raise ValueError("sandbox memory_mb must be at least 32")
        if self.cpus <= 0:
            raise ValueError("sandbox cpus must be positive")
        if self.pids_limit < 2:
            raise ValueError("sandbox pids_limit must be at least 2")
        if self.tmpfs_mb < 1:
            raise ValueError("sandbox tmpfs_mb must be positive")


class DockerSandbox:
    """Build and inspect disposable, hardened Docker containers."""

    def __init__(self, config: DockerSandboxConfig) -> None:
        config.validate()
        self.config = config

    def client_prefix(self) -> list[str]:
        return [
            self.config.docker_binary,
            "-H",
            self.config.docker_host,
        ]

    @staticmethod
    def environment() -> dict[str, str]:
        environment = os.environ.copy()
        environment.pop("DOCKER_HOST", None)
        return environment

    def _control(self, arguments: list[str], timeout: float = 15.0) -> str:
        try:
            completed = subprocess.run(
                [*self.client_prefix(), *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self.environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DockerSandboxUnavailable(
                f"Docker command failed: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise DockerSandboxUnavailable(f"Docker command failed: {detail}")
        return completed.stdout.strip()

    def preflight(self) -> dict[str, Any]:
        """Verify the daemon, isolation features, and configured image."""
        info_text = self._control(["info", "--format", "{{json .}}"])
        try:
            info = json.loads(info_text)
        except json.JSONDecodeError as exc:
            raise DockerSandboxUnavailable(
                "Docker info returned invalid JSON"
            ) from exc
        security_options = info.get("SecurityOptions") or []
        if not any("seccomp" in str(option) for option in security_options):
            raise DockerSandboxUnavailable("Docker daemon must enable seccomp")
        cgroup_version = str(info.get("CgroupVersion") or "")
        if cgroup_version not in {"1", "2"}:
            raise DockerSandboxUnavailable(
                "Docker daemon returned no cgroup version"
            )

        image_text = self._control(
            ["image", "inspect", "--format", "{{json .}}", self.config.image]
        )
        try:
            image = json.loads(image_text)
        except json.JSONDecodeError as exc:
            raise DockerSandboxUnavailable(
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

    def build_run_command(self, *, name: str, hostname: str) -> list[str]:
        """Build the command for one disposable sandbox container."""
        return [
            *self.client_prefix(),
            "run",
            "--interactive",
            "--rm",
            "--pull",
            "never",
            "--name",
            name,
            "--hostname",
            hostname,
            "--runtime",
            self.config.runtime,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            (
                "/tmp:rw,noexec,nosuid,nodev,"
                f"size={self.config.tmpfs_mb}m,mode=1777"
            ),
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

    def remove_container(self, name: str) -> None:
        """Best-effort cleanup for a container that exceeded its deadline."""
        try:
            subprocess.run(
                [*self.client_prefix(), "rm", "-f", name],
                check=False,
                capture_output=True,
                timeout=10,
                env=self.environment(),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
