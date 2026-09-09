"""Generate hardware monitoring charts from timeseries data.

Produces PNG images showing GPU utilization, power, temperature, and
memory usage trends over the benchmark lifecycle.  Each chart contains
one line per GPU device.  None values in the data create gaps in the
line rather than errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def generate_monitoring_charts(
    timeseries: list[dict[str, Any]],
    output_dir: Path,
    device_count: int = 1,
) -> list[str]:
    """Generate PNG charts from timeseries data.

    Returns a list of generated file names (relative to *output_dir*).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    if not timeseries:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    elapsed = [t.get("elapsed", 0) for t in timeseries]

    # Extract per-device timeseries
    device_data: dict[int, dict[str, list]] = {}
    for t in timeseries:
        for dev in t.get("devices", []):
            idx = dev.get("index", 0)
            if idx not in device_data:
                device_data[idx] = {
                    "utilization": [],
                    "power": [],
                    "temperature": [],
                    "memory_used": [],
                }
            device_data[idx]["utilization"].append(dev.get("utilization"))
            device_data[idx]["power"].append(dev.get("power"))
            device_data[idx]["temperature"].append(dev.get("temperature"))
            device_data[idx]["memory_used"].append(dev.get("memory_used"))

    # CPU timeseries
    cpu_util = [t.get("cpu", {}).get("utilization") if t.get("cpu") else None for t in timeseries]
    cpu_mem = [t.get("cpu", {}).get("memory_used") if t.get("cpu") else None for t in timeseries]

    def _plot(
        title: str,
        ylabel: str,
        data_dict: dict[int, list],
        key: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx in sorted(data_dict):
            values = data_dict[idx][key]
            # Convert None to NaN so matplotlib creates gaps, not errors
            y = [v if v is not None else np.nan for v in values]
            ax.plot(elapsed, y, label=f"GPU {idx}", linewidth=1.2)
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(filename)

    # GPU charts
    if device_data:
        _plot("GPU Utilization", "Utilization (%)", device_data, "utilization", "gpu_utilization.png")
        _plot("GPU Power", "Power (W)", device_data, "power", "gpu_power.png")
        _plot("GPU Temperature", "Temperature (C)", device_data, "temperature", "gpu_temperature.png")
        _plot("GPU Memory Used", "Memory (MB)", device_data, "memory_used", "gpu_memory.png")

    # CPU chart
    has_cpu_data = any(v is not None for v in cpu_util)
    if has_cpu_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        y = [v if v is not None else np.nan for v in cpu_util]
        ax.plot(elapsed, y, label="CPU Utilization", color="tab:blue", linewidth=1.2)
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Utilization (%)")
        ax.set_title("CPU Utilization")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        filename = "cpu_utilization.png"
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        generated.append(filename)

    # Memory chart
    has_mem_data = any(v is not None for v in cpu_mem)
    if has_mem_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        y = [v if v is not None else np.nan for v in cpu_mem]
        ax.plot(elapsed, y, label="Memory Used", color="tab:orange", linewidth=1.2)
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("System Memory Used")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        filename = "memory_usage.png"
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        generated.append(filename)

    return generated
