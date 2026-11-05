"""Run commands on the current server without invoking a shell."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from time import monotonic

from luban_meter.core.errors import ExecutionError
from luban_meter.core.run_contracts import CommandResult, CommandSpec
from luban_meter.log import get_logger, log_command

logger = get_logger(__name__)


class LocalCommandRunner:
    def run(self, command: CommandSpec) -> CommandResult:
        # A normal SIGTERM must unwind the caller's resource cleanup.
        handle_term = threading.current_thread() is threading.main_thread()
        previous = signal.getsignal(signal.SIGTERM) if handle_term else None

        def interrupted(signum, frame):
            raise ExecutionError(f"command interrupted by signal {signum}")

        if handle_term:
            signal.signal(signal.SIGTERM, interrupted)
        try:
            return self._run(command)
        finally:
            if handle_term:
                signal.signal(signal.SIGTERM, previous)

    def _run(self, command: CommandSpec) -> CommandResult:
        started = monotonic()
        logger.info("Starting command: argv=%s cwd=%s timeout=%s",
                    log_command(command.argv), command.cwd, command.timeout)

        try:
            process = subprocess.Popen(
                list(command.argv),
                cwd=command.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=None
                if command.env is None
                else {**os.environ, **command.env},
                start_new_session=os.name == "posix",
            )
            try:
                stdout, stderr = process.communicate(timeout=command.timeout)
            finally:
                # Descendants can retain pipes or continue creating
                # resources even
                # after the immediate benchmark process has exited.
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        logger.debug("Command process already exited")
                elif process.poll() is None:
                    process.kill()
                process.wait()
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
        except subprocess.TimeoutExpired as exc:
            logger.exception("Command timed out")
            raise ExecutionError(
                f"command timed out after {command.timeout} seconds: "
                f"{log_command(command.argv)!r}"
            ) from exc
        except OSError as exc:
            logger.exception("Command launch failed")
            raise ExecutionError(
                f"could not start command {log_command(command.argv)!r}: {exc}"
            ) from exc

        return CommandResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=monotonic() - started,
        )
