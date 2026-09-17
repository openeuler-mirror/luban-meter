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

    elapsed = [snapshot.get("elapsed", 0) for snapshot in timeseries]

    # Extract per-device timeseries
    device_data: dict[int, dict[str, list]] = {}
    for snapshot in timeseries:
        for device_sample in snapshot.get("devices", []):
            device_index = device_sample.get("index", 0)
            if device_index not in device_data:
                device_data[device_index] = {
                    "utilization": [],
                    "power": [],
                    "temperature": [],
                    "memory_used": [],
                }
            device_data[device_index]["utilization"].append(
                device_sample.get("utilization")
            )
            device_data[device_index]["power"].append(
                device_sample.get("power")
            )
            device_data[device_index]["temperature"].append(
                device_sample.get("temperature")
            )
            device_data[device_index]["memory_used"].append(
                device_sample.get("memory_used")
            )

    # CPU timeseries
    cpu_utilization = [
        snapshot.get("cpu", {}).get("utilization")
        if snapshot.get("cpu")
        else None
        for snapshot in timeseries
    ]
    cpu_memory_used = [
        snapshot.get("cpu", {}).get("memory_used")
        if snapshot.get("cpu")
        else None
        for snapshot in timeseries
    ]

    def _plot(
        title: str,
        ylabel: str,
        data_dict: dict[int, list],
        key: str,
        filename: str,
    ) -> None:
        figure, axes = plt.subplots(figsize=(10, 4))
        for device_index in sorted(data_dict):
            values = data_dict[device_index][key]
            # Convert None to NaN so matplotlib creates gaps, not errors
            plot_values = [
                sample_value if sample_value is not None else np.nan
                for sample_value in values
            ]
            axes.plot(
                elapsed,
                plot_values,
                label=f"GPU {device_index}",
                linewidth=1.2,
            )
        axes.set_xlabel("Elapsed (s)")
        axes.set_ylabel(ylabel)
        axes.set_title(title)
        axes.legend(loc="upper right", fontsize=8)
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        generated.append(filename)

    # GPU charts
    if device_data:
        _plot(
            "GPU Utilization",
            "Utilization (%)",
            device_data,
            "utilization",
            "gpu_utilization.png",
        )
        _plot("GPU Power", "Power (W)", device_data, "power", "gpu_power.png")
        _plot(
            "GPU Temperature",
            "Temperature (C)",
            device_data,
            "temperature",
            "gpu_temperature.png",
        )
        _plot(
            "GPU Memory Used",
            "Memory (MB)",
            device_data,
            "memory_used",
            "gpu_memory.png",
        )

    # CPU chart
    has_cpu_data = any(
        sample_value is not None for sample_value in cpu_utilization
    )
    if has_cpu_data:
        figure, axes = plt.subplots(figsize=(10, 4))
        plot_values = [
            sample_value if sample_value is not None else np.nan
            for sample_value in cpu_utilization
        ]
        axes.plot(
            elapsed,
            plot_values,
            label="CPU Utilization",
            color="tab:blue",
            linewidth=1.2,
        )
        axes.set_xlabel("Elapsed (s)")
        axes.set_ylabel("Utilization (%)")
        axes.set_title("CPU Utilization")
        axes.legend(loc="upper right", fontsize=8)
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        filename = "cpu_utilization.png"
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)
        generated.append(filename)

    # Memory chart
    has_mem_data = any(
        sample_value is not None for sample_value in cpu_memory_used
    )
    if has_mem_data:
        figure, axes = plt.subplots(figsize=(10, 4))
        plot_values = [
            sample_value if sample_value is not None else np.nan
            for sample_value in cpu_memory_used
        ]
        axes.plot(
            elapsed,
            plot_values,
            label="Memory Used",
            color="tab:orange",
            linewidth=1.2,
        )
        axes.set_xlabel("Elapsed (s)")
        axes.set_ylabel("Memory (MB)")
        axes.set_title("System Memory Used")
        axes.legend(loc="upper right", fontsize=8)
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        filename = "memory_usage.png"
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)
        generated.append(filename)

    return generated
