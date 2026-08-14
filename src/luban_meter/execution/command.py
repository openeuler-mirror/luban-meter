"""Run commands on the current server without invoking a shell."""

from __future__ import annotations

import subprocess
from time import monotonic

from luban_meter.core.errors import ExecutionError
from luban_meter.core.models import CommandResult, CommandSpec


class LocalCommandRunner:
    def run(self, command: CommandSpec) -> CommandResult:
        started = monotonic()

        try:
            completed = subprocess.run(
                list(command.argv),
                cwd=command.cwd,
                capture_output=True,
                text=True,
                timeout=command.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(
                f"command timed out after {command.timeout} seconds: "
                f"{list(command.argv)!r}"
            ) from exc
        except OSError as exc:
            raise ExecutionError(
                f"could not start command {list(command.argv)!r}: {exc}"
            ) from exc

        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=monotonic() - started,
        )
