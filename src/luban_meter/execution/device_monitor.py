"""Hardware monitoring daemon launched during benchmark execution.

Starts a background thread that periodically samples device metrics (GPU/NPU
utilization, memory, power, temperature) and aggregates averages on stop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from luban_meter.benchmark.generate.common.device_monitor import (
    DeviceInfo,
    DeviceSample,
    _sample_once,
    detect_devices,
    print_hardware_info,
)


@dataclass
class DeviceAvg:
    """Per-device average metrics over the sampling period."""

    index: int
    name: str
    vendor: str
    utilization_avg: float | None = None
    memory_used_avg_mb: float | None = None
    memory_total_mb: float | None = None
    power_avg_watts: float | None = None
    temperature_avg_celsius: float | None = None


@dataclass
class MonitoringSummary:
    """Aggregated hardware monitoring results."""

    vendor: str
    tool: str
    device_count: int
    sample_count: int
    duration_seconds: float
    interval_seconds: float
    devices: list[DeviceAvg] = field(default_factory=list)
    total_power_avg_watts: float | None = None
    total_energy_wh: float | None = None
    error_count: int = 0


class DeviceMonitorDaemon:
    """Background thread that samples device metrics periodically.

    Usage::

        daemon = DeviceMonitorDaemon(interval=1.0)
        daemon.start()
        # ... run benchmark ...
        daemon.stop()
        summary = daemon.summary()
    """

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = max(interval, 0.1)
        self._devices: list[DeviceInfo] = []
        self._vendor: str = ""
        self._samples: list[list[DeviceSample]] = []
        self._error_count = 0
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Print hardware info and start the background sampling thread."""
        self._devices = detect_devices()
        if not self._devices:
            print("[Device Monitor] No supported compute devices detected.")
            self._vendor = ""
            return

        print_hardware_info()
        self._vendor = self._devices[0].vendor
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
        """Aggregate collected samples and return a summary.

        Returns None if no devices were detected or no samples collected.
        """
        if not self._devices or not self._vendor or not self._samples:
            return None

        duration = self._end_time - self._start_time if self._end_time > 0 else 0.0
        device_avgs: list[DeviceAvg] = []

        for dev_info in self._devices:
            idx = dev_info.index
            dev_samples = [
                s for snap in self._samples for s in snap if s.index == idx
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

            util = _avg([s.utilization_percent for s in dev_samples])
            mem_used = _avg([s.memory_used_mb for s in dev_samples])
            mem_total = dev_samples[0].memory_total_mb
            power = _avg([s.power_watts for s in dev_samples])
            temp = _avg([s.temperature_celsius for s in dev_samples])

            device_avgs.append(
                DeviceAvg(
                    index=idx,
                    name=dev_info.name,
                    vendor=dev_info.vendor,
                    utilization_avg=util,
                    memory_used_avg_mb=mem_used,
                    memory_total_mb=mem_total,
                    power_avg_watts=power,
                    temperature_avg_celsius=temp,
                )
            )

        total_power = _avg(
            [_sum_power(snap) for snap in self._samples]
        )
        total_energy = (
            round(total_power * duration / 3600.0, 6)
            if total_power is not None and duration > 0
            else None
        )

        return MonitoringSummary(
            vendor=self._vendor,
            tool=self._devices[0].tool,
            device_count=len(self._devices),
            sample_count=len(self._samples),
            duration_seconds=round(duration, 3),
            interval_seconds=self._interval,
            devices=device_avgs,
            total_power_avg_watts=total_power,
            total_energy_wh=total_energy,
            error_count=self._error_count,
        )

    def _run(self) -> None:
        """Main sampling loop (runs in background thread)."""
        snapshot_index = 0
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_time
            snapshot = _sample_once(self._vendor, snapshot_index, elapsed)
            if snapshot.error:
                self._error_count += 1
            else:
                self._samples.append(snapshot.devices)
            snapshot_index += 1
            self._stop_event.wait(self._interval)


def _avg(values: list[float | None]) -> float | None:
    """Compute the arithmetic mean of non-None values, or None if all None."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def _sum_power(devices: list[DeviceSample]) -> float | None:
    """Sum power across all devices in a snapshot, or None if no data."""
    powers = [d.power_watts for d in devices if d.power_watts is not None]
    if not powers:
        return None
    return round(sum(powers), 2)