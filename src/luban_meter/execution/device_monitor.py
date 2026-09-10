"""Hardware monitoring daemon launched during benchmark execution.

Starts a background thread that periodically samples device metrics (GPU
utilization, memory, power, temperature) and CPU/memory metrics from a
Prometheus-compatible exporter endpoint, retaining both per-sample
**timeseries** data and aggregate averages + percentiles.

Monitoring is **optional**: if no exporter URL is provided, the daemon is
not started and no monitoring data is injected into results.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from luban_meter.benchmark.generate.common.device_monitor import (
    DeviceInfo,
    DeviceSample,
    Snapshot,
    collect_hardware_environment,
    detect_devices,
    print_hardware_info,
    sample_exporter,
)


@dataclass
class DeviceAvg:
    """Per-device aggregate metrics over the sampling period."""

    index: int
    name: str
    vendor: str
    utilization_avg: float | None = None
    utilization_p50: float | None = None
    utilization_p90: float | None = None
    utilization_p99: float | None = None
    memory_used_avg_mb: float | None = None
    memory_total_mb: float | None = None
    power_avg_watts: float | None = None
    power_p50_watts: float | None = None
    power_p90_watts: float | None = None
    power_p99_watts: float | None = None
    temperature_avg_celsius: float | None = None
    temperature_p50_celsius: float | None = None
    temperature_p90_celsius: float | None = None
    temperature_p99_celsius: float | None = None


@dataclass
class CpuAvg:
    """CPU/memory aggregate metrics over the sampling period."""

    utilization_avg: float | None = None
    utilization_p50: float | None = None
    utilization_p90: float | None = None
    utilization_p99: float | None = None
    memory_used_avg_mb: float | None = None
    memory_total_mb: float | None = None


@dataclass
class MonitoringSummary:
    """Aggregated hardware monitoring results with timeseries data."""

    vendor: str
    tool: str
    device_count: int
    sample_count: int
    duration_seconds: float
    interval_seconds: float
    devices: list[DeviceAvg] = field(default_factory=list)
    cpu: CpuAvg | None = None
    total_power_avg_watts: float | None = None
    total_energy_wh: float | None = None
    error_count: int = 0
    timeseries: list[dict[str, Any]] = field(default_factory=list)
    hardware_environment: dict[str, Any] | None = None


class DeviceMonitorDaemon:
    """Background thread that samples device metrics periodically via HTTP.

    Usage::

        daemon = DeviceMonitorDaemon(
            exporter_url="http://host:9400",
            interval=1.0,
        )
        daemon.start()
        # ... run benchmark ...
        daemon.stop()
        summary = daemon.summary()
    """

    def __init__(
        self,
        exporter_url: str,
        interval: float = 1.0,
    ) -> None:
        self._exporter_url = exporter_url
        self._interval = max(interval, 0.1)
        self._devices: list[DeviceInfo] = []
        self._vendor: str = ""
        self._snapshots: list[Snapshot] = []
        self._error_count = 0
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._hardware_env: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Detect devices from exporter and start the sampling thread."""
        self._devices = detect_devices(exporter_url=self._exporter_url)
        if not self._devices:
            print("[Device Monitor] No compute devices detected at the exporter endpoint.")
            self._vendor = ""
            return

        print_hardware_info(self._devices)
        self._vendor = self._devices[0].vendor
        self._hardware_env = collect_hardware_environment(exporter_url=self._exporter_url)
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampling thread to stop and wait for it."""
        self._end_time = time.monotonic()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def summary(self) -> MonitoringSummary | None:
        """Aggregate collected samples and return a summary with timeseries.

        Returns None if no devices were detected or no samples collected.
        """
        if not self._devices or not self._vendor or not self._snapshots:
            return None

        duration = self._end_time - self._start_time if self._end_time > 0 else 0.0
        device_avgs: list[DeviceAvg] = []

        for dev_info in self._devices:
            idx = dev_info.index
            dev_samples = [
                s for snap in self._snapshots for s in snap.devices if s.index == idx
            ]
            if not dev_samples:
                device_avgs.append(
                    DeviceAvg(
                        index=idx,
                        name=dev_info.name,
                        vendor=dev_info.vendor,
                    )
                )
                continue

            util_vals = [s.utilization_percent for s in dev_samples]
            power_vals = [s.power_watts for s in dev_samples]
            temp_vals = [s.temperature_celsius for s in dev_samples]

            device_avgs.append(
                DeviceAvg(
                    index=idx,
                    name=dev_info.name,
                    vendor=dev_info.vendor,
                    utilization_avg=_avg(util_vals),
                    utilization_p50=_percentile(util_vals, 50),
                    utilization_p90=_percentile(util_vals, 90),
                    utilization_p99=_percentile(util_vals, 99),
                    memory_used_avg_mb=_avg([s.memory_used_mb for s in dev_samples]),
                    memory_total_mb=dev_samples[0].memory_total_mb,
                    power_avg_watts=_avg(power_vals),
                    power_p50_watts=_percentile(power_vals, 50),
                    power_p90_watts=_percentile(power_vals, 90),
                    power_p99_watts=_percentile(power_vals, 99),
                    temperature_avg_celsius=_avg(temp_vals),
                    temperature_p50_celsius=_percentile(temp_vals, 50),
                    temperature_p90_celsius=_percentile(temp_vals, 90),
                    temperature_p99_celsius=_percentile(temp_vals, 99),
                )
            )

        total_power = _avg(
            [_sum_power(snap.devices) for snap in self._snapshots]
        )
        total_energy = (
            round(total_power * duration / 3600.0, 6)
            if total_power is not None and duration > 0
            else None
        )

        # CPU averages
        cpu_avg: CpuAvg | None = None
        cpu_samples = [snap.cpu for snap in self._snapshots if snap.cpu is not None]
        if cpu_samples:
            cpu_util_vals = [c.utilization_percent for c in cpu_samples]
            cpu_avg = CpuAvg(
                utilization_avg=_avg(cpu_util_vals),
                utilization_p50=_percentile(cpu_util_vals, 50),
                utilization_p90=_percentile(cpu_util_vals, 90),
                utilization_p99=_percentile(cpu_util_vals, 99),
                memory_used_avg_mb=_avg([c.memory_used_mb for c in cpu_samples]),
                memory_total_mb=cpu_samples[0].memory_total_mb,
            )

        # Build timeseries for JSON output
        timeseries: list[dict[str, Any]] = []
        for snap in self._snapshots:
            entry: dict[str, Any] = {
                "elapsed": snap.elapsed_seconds,
                "devices": [
                    {
                        "index": d.index,
                        "utilization": d.utilization_percent,
                        "power": d.power_watts,
                        "temperature": d.temperature_celsius,
                        "memory_used": d.memory_used_mb,
                    }
                    for d in snap.devices
                ],
            }
            if snap.cpu is not None:
                entry["cpu"] = {
                    "utilization": snap.cpu.utilization_percent,
                    "memory_used": snap.cpu.memory_used_mb,
                }
            timeseries.append(entry)

        return MonitoringSummary(
            vendor=self._vendor,
            tool=self._devices[0].tool,
            device_count=len(self._devices),
            sample_count=len(self._snapshots),
            duration_seconds=round(duration, 3),
            interval_seconds=self._interval,
            devices=device_avgs,
            cpu=cpu_avg,
            total_power_avg_watts=total_power,
            total_energy_wh=total_energy,
            error_count=self._error_count,
            timeseries=timeseries,
            hardware_environment=self._hardware_env,
        )

    def _run(self) -> None:
        """Main sampling loop (runs in background thread)."""
        snapshot_index = 0
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_time
            snapshot = sample_exporter(
                self._exporter_url,
                self._devices,
                snapshot_index,
                elapsed,
            )
            if snapshot.error:
                self._error_count += 1
            self._snapshots.append(snapshot)
            snapshot_index += 1
            self._stop_event.wait(self._interval)


def _avg(values: list[float | None]) -> float | None:
    """Compute the arithmetic mean of non-None values, or None if all None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def _percentile(values: list[float | None], p: float) -> float | None:
    """Compute the *p*-th percentile of non-None values, or None."""
    valid = sorted(v for v in values if v is not None)
    if not valid:
        return None
    k = (len(valid) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(valid) - 1)
    if f == c:
        return round(valid[f], 2)
    return round(valid[f] + (valid[c] - valid[f]) * (k - f), 2)


def _sum_power(devices: list[DeviceSample]) -> float | None:
    """Sum power across all devices in a snapshot, or None if no data."""
    powers = [d.power_watts for d in devices if d.power_watts is not None]
    if not powers:
        return None
    return round(sum(powers), 2)
