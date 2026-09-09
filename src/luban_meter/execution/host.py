"""Execute a Benchmark tool directly on the current host.

Hardware monitoring is optional: if a Prometheus exporter URL is provided,
a background daemon samples GPU/CPU metrics during the benchmark run.
If no URL is provided, monitoring is skipped entirely.
"""

from __future__ import annotations

import json
import sys
from typing import Any

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
        monitor_url: str | None = None,
        monitor_interval: float = 1.0,
    ) -> None:
        self._runner = runner or LocalCommandRunner()
        self._monitor_url = monitor_url
        self._monitor_interval = monitor_interval

    def execute(self, run: ResolvedRun) -> RawRunArtifacts:
        run_dir, raw_dir, artifact_dir = prepare_run_directory(run)
        raw_result = raw_dir / "raw_result.json"

        # Start hardware monitoring daemon (only if exporter URL is provided)
        daemon: DeviceMonitorDaemon | None = None
        if self._monitor_url:
            daemon = DeviceMonitorDaemon(
                exporter_url=self._monitor_url,
                interval=self._monitor_interval,
            )
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
        if daemon is not None:
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

        monitoring = asdict(summary)

        # Generate charts from timeseries data
        timeseries = monitoring.get("timeseries", [])
        artifact_dir = raw_result.parent / "artifacts"
        if timeseries:
            try:
                from luban_meter.result.charts import generate_monitoring_charts
                charts = generate_monitoring_charts(timeseries, artifact_dir)
                monitoring["charts"] = charts
            except Exception:
                pass  # matplotlib may not be installed

        # Put hardware_environment at the top of the result for easy access
        hw_env = monitoring.pop("hardware_environment", None)
        if hw_env:
            new_data = {"hardware_environment": hw_env}
            new_data.update(data)
            data = new_data

        data["device_monitoring"] = monitoring
        try:
            with raw_result.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
