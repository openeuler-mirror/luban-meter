"""Shared device (GPU/NPU/CPU) detection and monitoring via Prometheus exporters.

Collects hardware metrics from a Prometheus-compatible exporter endpoint
(e.g. DCGM exporter for NVIDIA GPUs, node_exporter for CPU/memory).

The exporter URL is provided by the user. If not provided, monitoring is
disabled.

Supported exporters:
  - DCGM exporter  (NVIDIA GPU)  — default port 9400
  - node_exporter   (CPU/memory)  — default port 9100

Provides:
  - collect_hardware_environment()  — static hardware environment summary
  - sample_exporter()                — take a single snapshot of metrics
  - Snapshot / DeviceSample / CpuSample / DeviceInfo — data classes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    vendor: str
    tool: str
    index: int
    name: str


@dataclass
class DeviceSample:
    index: int
    name: str
    vendor: str
    utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    power_watts: float | None = None
    temperature_celsius: float | None = None


@dataclass
class CpuSample:
    utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None


@dataclass
class Snapshot:
    index: int
    elapsed_seconds: float
    devices: list[DeviceSample] = field(default_factory=list)
    cpu: CpuSample | None = None
    error: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# HTTP fetch helper
# ---------------------------------------------------------------------------

def _fetch_metrics(url: str, timeout: float = 5.0) -> str | None:
    """Fetch Prometheus text-format metrics from *url*/metrics."""
    if requests is None:
        return None
    if not url.startswith("http"):
        url = f"http://{url}"
    metrics_url = url.rstrip("/").removesuffix("/metrics") + "/metrics"
    try:
        resp = requests.get(metrics_url, timeout=timeout)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
    except Exception:
        pass
    return None


def _parse_prom_value(text: str, metric_name: str, labels: dict[str, str] | None = None) -> list[tuple[dict[str, str], float]]:
    """Parse Prometheus text format for a specific metric.

    Returns a list of (label_dict, value) tuples.
    Only numeric values are returned; string values are skipped.
    """
    results: list[tuple[dict[str, str], float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: metric_name{label1="val1",label2="val2"} value
        m = re.match(r'^(\w+)(?:\{([^}]*)\})?\s+(\S+)$', line)
        if not m:
            continue
        name, label_str, val_str = m.group(1), m.group(2), m.group(3)
        if name != metric_name:
            continue
        label_dict: dict[str, str] = {}
        if label_str:
            for lm in re.finditer(r'(\w+)="([^"]*)"', label_str):
                label_dict[lm.group(1)] = lm.group(2)
        if labels and not all(label_dict.get(k) == v for k, v in labels.items()):
            continue
        try:
            results.append((label_dict, float(val_str)))
        except ValueError:
            pass  # skip string values like "NVIDIA H20"
    return results


def _parse_prom_str_value(text: str, metric_name: str, labels: dict[str, str] | None = None) -> list[tuple[dict[str, str], str]]:
    """Parse Prometheus text format for string-valued metrics.

    Returns a list of (label_dict, string_value) tuples.
    """
    results: list[tuple[dict[str, str], str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^(\w+)(?:\{([^}]*)\})?\s+"?([^"]+)"?$', line)
        if not m:
            continue
        name, label_str, val_str = m.group(1), m.group(2), m.group(3)
        if name != metric_name:
            continue
        label_dict: dict[str, str] = {}
        if label_str:
            for lm in re.finditer(r'(\w+)="([^"]*)"', label_str):
                label_dict[lm.group(1)] = lm.group(2)
        if labels and not all(label_dict.get(k) == v for k, v in labels.items()):
            continue
        results.append((label_dict, val_str))
    return results


def _parse_prom_simple(text: str, metric_name: str) -> float | None:
    """Parse a single value from Prometheus text format."""
    results = _parse_prom_value(text, metric_name)
    if results:
        return results[0][1]
    return None


# ---------------------------------------------------------------------------
# Device detection (from exporter)
# ---------------------------------------------------------------------------

def detect_devices(
    metrics_text: str | None = None,
    exporter_url: str | None = None,
) -> list[DeviceInfo]:
    """Detect GPUs from DCGM exporter metrics."""
    if metrics_text is None and exporter_url:
        metrics_text = _fetch_metrics(exporter_url)
    if not metrics_text:
        return []

    # Retain support for explicit device-name fields.
    name_entries = _parse_prom_str_value(metrics_text, "DCGM_FI_DEV_NAME")
    if not name_entries:
        # Fall back: infer device count from DCGM_FI_DEV_GPU_UTIL
        util_entries = _parse_prom_value(metrics_text, "DCGM_FI_DEV_GPU_UTIL")
        if not util_entries:
            return []
        devices: list[DeviceInfo] = []
        for labels, _ in util_entries:
            gpu = labels.get("gpu", "0")
            try:
                idx = int(gpu)
            except ValueError:
                continue
            devices.append(DeviceInfo(
                vendor="nvidia",
                tool="dcgm-exporter",
                index=idx,
                name=labels.get("modelName") or f"GPU-{idx}",
            ))
        return devices

    devices: list[DeviceInfo] = []
    for labels, name_val in name_entries:
        gpu = labels.get("gpu", "0")
        try:
            idx = int(gpu)
        except ValueError:
            continue
        devices.append(DeviceInfo(
            vendor="nvidia",
            tool="dcgm-exporter",
            index=idx,
            name=labels.get("modelName") or name_val or f"GPU-{idx}",
        ))
    return devices


def print_hardware_info(devices: list[DeviceInfo] | None = None, exporter_url: str | None = None) -> None:
    """Print detected hardware info to stdout."""
    if devices is None:
        metrics_text = _fetch_metrics(exporter_url) if exporter_url else None
        devices = detect_devices(metrics_text)
    if not devices:
        print("[Device Monitor] No supported compute devices detected at the exporter endpoint.")
        return
    vendor = devices[0].vendor
    tool = devices[0].tool
    print(f"[Device Monitor] Detected {len(devices)} x {devices[0].name} ({vendor}) via {tool}")
    for dev in devices:
        print(f"  [{dev.vendor}] Device {dev.index}: {dev.name}")
    print()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_gpu_from_exporter(metrics_text: str, devices: list[DeviceInfo]) -> list[DeviceSample]:
    """Parse GPU metrics from DCGM exporter text."""
    samples: list[DeviceSample] = []
    for dev in devices:
        idx = dev.index
        labels = {"gpu": str(idx)}
        sample = DeviceSample(index=idx, name=dev.name, vendor=dev.vendor)

        util = _parse_prom_value(metrics_text, "DCGM_FI_DEV_GPU_UTIL", labels)
        if util:
            sample.utilization_percent = util[0][1]

        mem_used = _parse_prom_value(metrics_text, "DCGM_FI_DEV_FB_USED", labels)
        if mem_used:
            sample.memory_used_mb = mem_used[0][1]

        mem_total = _parse_prom_value(metrics_text, "DCGM_FI_DEV_FB_TOTAL", labels)
        if mem_total:
            sample.memory_total_mb = mem_total[0][1]

        power = _parse_prom_value(metrics_text, "DCGM_FI_DEV_POWER_USAGE", labels)
        if power:
            sample.power_watts = power[0][1]

        temp = _parse_prom_value(metrics_text, "DCGM_FI_DEV_GPU_TEMP", labels)
        if temp:
            sample.temperature_celsius = temp[0][1]

        samples.append(sample)
    return samples


def _sample_cpu_from_exporter(metrics_text: str) -> CpuSample:
    """Parse CPU/memory metrics from node_exporter text."""
    cpu_sample = CpuSample()

    # CPU utilization: node_cpu_seconds_total{mode="idle"} vs total
    idle_vals = _parse_prom_value(metrics_text, "node_cpu_seconds_total", {"mode": "idle"})
    all_vals = _parse_prom_value(metrics_text, "node_cpu_seconds_total")
    if idle_vals and all_vals:
        idle_total = sum(v for _, v in idle_vals)
        all_total = sum(v for _, v in all_vals)
        if all_total > 0:
            cpu_sample.utilization_percent = round((1 - idle_total / all_total) * 100, 2)

    # Memory: node_memory_MemTotal_bytes / node_memory_MemAvailable_bytes
    mem_total = _parse_prom_simple(metrics_text, "node_memory_MemTotal_bytes")
    mem_avail = _parse_prom_simple(metrics_text, "node_memory_MemAvailable_bytes")
    if mem_total is not None:
        cpu_sample.memory_total_mb = round(mem_total / (1024 * 1024), 1)
    if mem_total is not None and mem_avail is not None:
        cpu_sample.memory_used_mb = round((mem_total - mem_avail) / (1024 * 1024), 1)

    return cpu_sample


# ---------------------------------------------------------------------------
# Hardware environment (static info)
# ---------------------------------------------------------------------------

def collect_hardware_environment(
    metrics_text: str | None = None,
    exporter_url: str | None = None,
) -> dict[str, Any]:
    """Collect static hardware environment info from exporter metrics."""
    if metrics_text is None and exporter_url:
        metrics_text = _fetch_metrics(exporter_url)
    if not metrics_text:
        return {}

    env: dict[str, Any] = {}
    devices = detect_devices(metrics_text)
    if devices:
        env["vendor"] = devices[0].vendor
        env["tool"] = devices[0].tool
        env["devices"] = [
            {"index": d.index, "name": d.name}
            for d in devices
        ]

    # CPU model from node_exporter
    cpu_model = _parse_prom_simple(metrics_text, "node_cpu_model")
    if cpu_model is None:
        # Some exporters use node_uname_info
        uname = _parse_prom_value(metrics_text, "node_uname_info")
        if uname:
            cpu_model = uname[0][0].get("release", None)
    cpu_cores = _parse_prom_value(metrics_text, "node_cpu_count")
    if cpu_model or cpu_cores:
        env["cpu"] = {
            "model": cpu_model if isinstance(cpu_model, str) else None,
            "cores": int(cpu_cores[0][1]) if cpu_cores else None,
            "sockets": 1,
        }

    # Memory
    mem_total = _parse_prom_simple(metrics_text, "node_memory_MemTotal_bytes")
    if mem_total is not None:
        env["memory_total_gb"] = round(mem_total / (1024 ** 3), 1)

    return env


# ---------------------------------------------------------------------------
# Single snapshot
# ---------------------------------------------------------------------------

def sample_exporter(
    exporter_url: str,
    devices: list[DeviceInfo],
    snapshot_index: int,
    elapsed: float,
) -> Snapshot:
    """Take a single snapshot of all devices via Prometheus exporter."""
    snapshot = Snapshot(index=snapshot_index, elapsed_seconds=round(elapsed, 6))
    metrics_text = _fetch_metrics(exporter_url)
    if not metrics_text:
        snapshot.error = {"type": "FetchError", "message": f"Failed to fetch metrics from {exporter_url}"}
        return snapshot

    try:
        snapshot.devices = _sample_gpu_from_exporter(metrics_text, devices)
    except Exception as exc:
        snapshot.error = {"type": type(exc).__name__, "message": str(exc)}

    try:
        snapshot.cpu = _sample_cpu_from_exporter(metrics_text)
    except Exception:
        pass

    return snapshot
