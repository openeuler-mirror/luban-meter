"""Run commands on the current server without invoking a shell."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from time import monotonic

from luban_meter.core.errors import ExecutionError
from luban_meter.core.run_contracts import CommandResult, CommandSpec


class LocalCommandRunner:
    def run(self, command: CommandSpec) -> CommandResult:
        # A normal SIGTERM must unwind HostSession's finally, just like
        # Ctrl-C.
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
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait()
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
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
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=monotonic() - started,
        )
