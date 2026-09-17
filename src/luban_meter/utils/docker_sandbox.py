"""Reusable helpers for constructing hardened Docker sandboxes."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCOPE_ENV = "LUBAN_METER_SANDBOX_SCOPE"
REGISTRY_ENV = "LUBAN_METER_SANDBOX_REGISTRY"
SCOPE_LABEL = "io.luban-meter.sandbox-scope"


class DockerSandboxUnavailableError(RuntimeError):
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
        self.config_path = config

    def client_prefix(self) -> list[str]:
        return [
            self.config_path.docker_binary,
            "-H",
            self.config_path.docker_host,
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
            raise DockerSandboxUnavailableError(
                f"Docker command failed: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise DockerSandboxUnavailableError(
                f"Docker command failed: {detail}"
            )
        return completed.stdout.strip()

    def preflight(self) -> dict[str, Any]:
        """Verify the daemon, isolation features, and configured image."""
        info_text = self._control(["info", "--format", "{{json .}}"])
        try:
            info = json.loads(info_text)
        except json.JSONDecodeError as exc:
            raise DockerSandboxUnavailableError(
                "Docker info returned invalid JSON"
            ) from exc
        security_options = info.get("SecurityOptions") or []
        if not any("seccomp" in str(option) for option in security_options):
            raise DockerSandboxUnavailableError(
                "Docker daemon must enable seccomp"
            )
        cgroup_version = str(info.get("CgroupVersion") or "")
        if cgroup_version not in {"1", "2"}:
            raise DockerSandboxUnavailableError(
                "Docker daemon returned no cgroup version"
            )

        image_text = self._control(
            [
                "image",
                "inspect",
                "--format",
                "{{json .}}",
                self.config_path.image,
            ]
        )
        try:
            image = json.loads(image_text)
        except json.JSONDecodeError as exc:
            raise DockerSandboxUnavailableError(
                "sandbox image inspect returned invalid JSON"
            ) from exc
        return {
            "runtime": "docker",
            "server_version": info.get("ServerVersion"),
            "docker_root_dir": info.get("DockerRootDir"),
            "cgroup_driver": info.get("CgroupDriver"),
            "cgroup_version": cgroup_version,
            "security_options": list(security_options),
            "image": self.config_path.image,
            "image_id": image.get("Id"),
            "oci_runtime": self.config_path.runtime,
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
            "--label",
            f"{SCOPE_LABEL}={os.environ.get(SCOPE_ENV, name)}",
            "--hostname",
            hostname,
            "--runtime",
            self.config_path.runtime,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            (
                f"/tmp:rw,noexec,nosuid,nodev,size={self.config_path.tmpfs_mb}m,mode=1777"
            ),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.config_path.pids_limit),
            "--memory",
            f"{self.config_path.memory_mb}m",
            "--memory-swap",
            f"{self.config_path.memory_mb}m",
            "--cpus",
            str(self.config_path.cpus),
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
            self.config_path.image,
        ]

    def remove_container(self, name: str) -> None:
        """Remove an exact named container and verify removal; fail
        visibly.
        """
        # Listing first distinguishes an already auto-removed container
        # from
        # daemon unavailability; never suppress a Docker failure.
        ids = self._control(
            ["ps", "-aq", "--no-trunc", "--filter", f"name=^/{name}$"]
        ).split()
        for container_id in ids:
            self._control(["rm", "-f", container_id], timeout=10)
        if self._control(["ps", "-aq", "--filter", f"name=^/{name}$"]):
            raise DockerSandboxUnavailableError(
                f"container still exists: {name}"
            )

    def create_container(self, *, name: str, hostname: str) -> str:
        """Create without starting, so interrupted creates cannot
        execute code.
        """
        arguments = self.build_run_command(name=name, hostname=hostname)[
            len(self.client_prefix()) :
        ]
        arguments[0] = "create"
        arguments.remove("--rm")
        container_id = self._control(arguments)
        if not container_id:
            raise DockerSandboxUnavailableError(
                "Docker create returned no container ID"
            )
        return container_id


class DockerSandboxScope:
    """Parent-owned registry. Only the parent calls cleanup after child
    exit.
    """

    def __init__(self, directory: Path, config: DockerSandboxConfig) -> None:
        self.token = uuid.uuid4().hex
        self.directory = directory / self.token
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sandbox = DockerSandbox(config)
        (self.directory / "owner.json").write_text(
            json.dumps(
                {
                    "scope": self.token,
                    "docker_host": config.docker_host,
                    "docker_binary": config.docker_binary,
                }
            ),
            encoding="utf-8",
        )

    def environment(self) -> dict[str, str]:
        return {
            SCOPE_ENV: self.token,
            REGISTRY_ENV: str(self.directory),
        }

    def cleanup(self, grace_seconds: float = 3.0) -> None:
        """Reconcile interrupted creates; report ambiguity rather than
        success.
        """
        pending = {
            pending_path.name
            for pending_path in self.directory.glob("*.pending")
        }
        observed: set[str] = set()
        deadline = time.monotonic() + grace_seconds
        try:
            while True:
                rows = self.sandbox._control(
                    [
                        "ps",
                        "-a",
                        "--no-trunc",
                        "--filter",
                        f"label={SCOPE_LABEL}={self.token}",
                        "--format",
                        "{{.ID}} {{.Names}}",
                    ]
                )
                for row in rows.splitlines():
                    container_id, name = row.split()
                    self.sandbox._control(
                        ["rm", "-f", container_id], timeout=10
                    )
                    observed.add(name + ".pending")
                unresolved = pending - observed
                if not rows and not unresolved:
                    break
                if time.monotonic() >= deadline:
                    raise DockerSandboxUnavailableError(
                        "sandbox cleanup unconfirmed; interrupted creates or "
                        f"remaining containers; registry: {self.directory}"
                    )
                time.sleep(0.1)
        except Exception as exc:
            (self.directory / "cleanup-error.txt").write_text(
                str(exc), encoding="utf-8"
            )
            raise
        (self.directory / "cleanup-ok").touch()
