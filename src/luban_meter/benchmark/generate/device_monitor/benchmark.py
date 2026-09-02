"""Collect device (GPU/NPU) utilization, memory, power, and temperature.

This is the CLI entry point for the device-monitor benchmark.
The shared detection and monitoring logic lives in
``luban_meter.benchmark.generate.common.device_monitor``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import threading
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from luban_meter.benchmark.generate.common.device_monitor import (
    Snapshot,
    _sample_once,
    detect_devices,
    print_hardware_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Device metrics collector"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_request(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request")
    parameters = payload.get("parameters")
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be an object")
    return dict(request), dict(parameters)


def positive_number(
    parameters: Mapping[str, Any], name: str, default: float
) -> float:
    value = parameters.get(name, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def run_benchmark(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    collect_interval = positive_number(parameters, "collect_interval", 1.0)
    collect_duration = positive_number(parameters, "collect_duration", 60.0)

    # Print hardware info before collection
    print_hardware_info()

    # Detect devices
    devices = detect_devices()
    device_info = [asdict(d) for d in devices]

    if not devices:
        return {
            "schema_version": "luban-meter.raw/v1",
            "status": "success",
            "metadata": {
                "measurement": "device_monitor",
                "device_count": 0,
                "devices": [],
                "collect_interval": collect_interval,
                "collect_duration": collect_duration,
                "message": "No supported compute devices detected on this host.",
            },
            "metrics": {
                "snapshots": [],
            },
        }

    vendor = devices[0].vendor
    snapshots: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    snapshot_index = 0
    lock = threading.Lock()
    stop_event = threading.Event()

    def _collect_loop() -> None:
        nonlocal snapshot_index
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start_time
            if elapsed >= collect_duration:
                break
            snapshot_start = time.perf_counter()
            snapshot = _sample_once(vendor, snapshot_index, elapsed)
            with lock:
                snapshots.append({
                    "index": snapshot.index,
                    "elapsed_seconds": snapshot.elapsed_seconds,
                    "devices": [asdict(d) for d in snapshot.devices],
                    "error": (
                        {"type": snapshot.error["type"], "message": snapshot.error["message"]}
                        if snapshot.error else None
                    ),
                })
            snapshot_index += 1
            remaining = collect_interval - (time.perf_counter() - snapshot_start)
            if remaining > 0:
                stop_event.wait(remaining)

    collector = threading.Thread(target=_collect_loop, daemon=True)
    collector.start()
    collector.join(timeout=collect_duration + 10.0)
    stop_event.set()

    total_elapsed = round(time.perf_counter() - start_time, 6)
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metadata": {
            "measurement": "device_monitor",
            "device_count": len(devices),
            "devices": device_info,
            "vendor": vendor,
            "collect_interval": collect_interval,
            "collect_duration": collect_duration,
            "snapshot_count": len(snapshots),
            "total_elapsed_seconds": total_elapsed,
        },
        "metrics": {
            "snapshots": snapshots,
        },
    }


def main() -> None:
    args = parse_args()
    request_data, parameters = load_request(args.request)
    result = run_benchmark(request_data, parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()