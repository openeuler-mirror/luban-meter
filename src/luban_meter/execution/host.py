"""Execute a Benchmark tool directly on the current host."""

from __future__ import annotations

import json
import sys

from luban_meter.core.errors import ExecutionError
from luban_meter.core.models import (
    CommandSpec,
    RawRunArtifacts,
    ResolvedRun,
)
from luban_meter.execution.command import LocalCommandRunner
from luban_meter.execution.device_monitor import DeviceMonitorDaemon
from luban_meter.execution.session import (
    collected_artifacts,
    prepare_run_directory,
    write_command_logs,
)


class HostSession:
    def __init__(
        self,
        runner: LocalCommandRunner | None = None,
    ) -> None:
        self._runner = runner or LocalCommandRunner()

    def execute(self, run: ResolvedRun) -> RawRunArtifacts:
        run_dir, raw_dir, artifact_dir = prepare_run_directory(run)
        raw_result = raw_dir / "raw_result.json"

        # Start hardware monitoring daemon before benchmark execution
        daemon = DeviceMonitorDaemon(interval=1.0)
        daemon.start()

        result = self._runner.run(
            CommandSpec(
                argv=(
                    sys.executable,
                    str(run.benchmark.benchmark_entry),
                    "--request",
                    str(run_dir / "request.json"),
                    "--output",
                    str(raw_result),
                ),
                timeout=run.request.timeout,
            )
        )
        write_command_logs(raw_dir, result.stdout, result.stderr)

        # Stop daemon and inject monitoring summary into raw_result.json
        daemon.stop()
        self._inject_monitoring_summary(raw_result, daemon.summary())

        if result.returncode != 0:
            raise ExecutionError(
                f"benchmark exited with code {result.returncode}; "
                f"see {raw_dir / 'stderr.log'}"
            )
        if not raw_result.is_file():
            raise ExecutionError(
                f"benchmark did not produce required output: {raw_result}"
            )
        return collected_artifacts(raw_dir, artifact_dir)

    def close(self) -> None:
        return None

    @staticmethod
    def _inject_monitoring_summary(raw_result, summary) -> None:
        """Read raw_result.json, inject device monitoring summary, write back."""
        if summary is None:
            return
        try:
            with raw_result.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(data, dict):
            return

        from dataclasses import asdict

        data["device_monitoring"] = asdict(summary)
        try:
            with raw_result.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass