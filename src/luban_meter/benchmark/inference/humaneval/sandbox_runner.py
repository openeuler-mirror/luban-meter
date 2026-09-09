"""Execute one HumanEval program inside the sandbox container."""

from __future__ import annotations

import contextlib
import io
import json
import signal
import sys
import time
from collections.abc import Mapping
from typing import Any

RESULT_PREFIX = "LUBAN_METER_RESULT="
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_CAPTURE_CHARS = 8192


class _ExecutionTimeout(Exception):
    pass


class _LimitedWriter(io.TextIOBase):
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._size = 0
        self.truncated = False

    def write(self, value: str) -> int:
        text = str(value)
        remaining = max(0, self._limit - self._size)
        if remaining:
            piece = text[:remaining]
            self._parts.append(piece)
            self._size += len(piece)
        if len(text) > remaining:
            self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        return "".join(self._parts)


def _error(status: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": status,
        "passed": False,
        "error_type": type(error).__name__,
        "error_message": str(error)[:1000],
    }


def execute_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    program = payload.get("program")
    timeout = payload.get("timeout_seconds", 3.0)
    if not isinstance(program, str) or not program:
        raise ValueError("program must be a non-empty string")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ValueError("timeout_seconds must be positive")

    stdout = _LimitedWriter(MAX_CAPTURE_CHARS)
    stderr = _LimitedWriter(MAX_CAPTURE_CHARS)
    started = time.perf_counter()
    try:
        compiled = compile(program, "<humaneval>", "exec")
    except SyntaxError as exc:
        result = _error("syntax_error", exc)
    else:
        previous_handler = signal.getsignal(signal.SIGALRM)

        def handle_timeout(_signum: int, _frame: object) -> None:
            raise _ExecutionTimeout("execution timed out")

        signal.signal(signal.SIGALRM, handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                # This process runs only inside the locked-down sandbox image.
                exec(compiled, {"__name__": "__main__"})  # noqa: S102
        except _ExecutionTimeout as exc:
            result = _error("timeout", exc)
        except AssertionError as exc:
            result = _error("failed_test", exc)
        # Generated code may raise SystemExit or other BaseException subclasses.
        except BaseException as exc:  # noqa: BLE001
            result = _error("runtime_error", exc)
        else:
            result = {
                "status": "passed",
                "passed": True,
                "error_type": None,
                "error_message": None,
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    result.update(
        {
            "execution_ms": (time.perf_counter() - started) * 1000.0,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "stdout_truncated": stdout.truncated,
            "stderr_truncated": stderr.truncated,
        }
    )
    return result


def main() -> None:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        result = _error("sandbox_error", ValueError("sandbox input is too large"))
    else:
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("sandbox payload must be an object")
            result = execute_payload(payload)
        # Always return the versioned sandbox protocol, even on invalid input.
        except BaseException as exc:  # noqa: BLE001
            result = _error("sandbox_error", exc)
    sys.stdout.write(RESULT_PREFIX + json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
