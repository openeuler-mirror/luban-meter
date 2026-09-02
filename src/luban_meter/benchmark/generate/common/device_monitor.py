"""Shared device (GPU/NPU) detection and monitoring utilities.

Auto-detects available monitoring tools on the current host:
  - nvidia-smi   (NVIDIA GPU)
  - npu-smi      (Huawei Ascend)
  - rocm-smi     (AMD / Hygon DCU)
  - cnmon        (Cambricon)
  - mt-gpu-smi   (Moore Threads)
  - biren-smi    (Biren)
  - tops-smi     (Enflame)

Provides:
  - detect_devices()       — auto-detect available hardware
  - print_hardware_info()  — print formatted hardware info to stdout
  - _sample_once()         — take a single snapshot of device metrics
  - Snapshot / DeviceSample / DeviceInfo — data classes
"""

from __future__ import annotations

import csv
import io
import math
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, asdict
from typing import Any


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
class Snapshot:
    index: int
    elapsed_seconds: float
    devices: list[DeviceSample] = field(default_factory=list)
    error: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    """Convert a value to float, returning None if not possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        value = value.strip().strip('"').strip("'")
        if not value or value in ("N/A", "N/A (Not Supported)", "[N/A]"):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _run_tool(tool: str, args: Sequence[str], use_sudo: bool = False) -> str | None:
    """Run a monitoring tool and return stdout, or None if not found/error."""
    tool_path = tool
    if use_sudo:
        resolved = shutil.which(tool)
        if resolved is None:
            return None
        tool_path = resolved
        cmd = ["sudo", "-n", tool_path, *args]
    else:
        cmd = [tool, *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        pass
    return None


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _detect_nvidia() -> list[DeviceInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    out = _run_tool("nvidia-smi", [
        "--query-gpu=index,name",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for line in io.StringIO(out):
        line = line.strip()
        if not line:
            continue
        parts = line.split(", ", 1)
        if len(parts) == 2:
            try:
                idx = int(parts[0])
                devices.append(DeviceInfo("nvidia", "nvidia-smi", idx, parts[1]))
            except ValueError:
                pass
    return devices


def _detect_ascend() -> list[DeviceInfo]:
    """Detect Huawei Ascend NPUs via npu-smi."""
    out = _run_tool("npu-smi", ["info"])
    if not out:
        out = _run_tool("npu-smi", ["info"], use_sudo=True)
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for line in out.splitlines():
        if line.count("|") != 4:
            continue
        # Skip row 2 lines: index followed by empty column then pipe
        if re.match(r"\|\s*\d+\s+\|", line):
            continue
        m = re.match(r"\|\s*(\d+)\s+(\S+)", line)
        if m:
            idx = int(m.group(1))
            name = m.group(2)
            if not any(d.index == idx for d in devices):
                devices.append(DeviceInfo("ascend", "npu-smi", idx, name))
    return devices


def _detect_rocm() -> list[DeviceInfo]:
    """Detect AMD / Hygon DCU via rocm-smi."""
    out = _run_tool("rocm-smi", ["--showid", "--showproductname", "--csv"])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for line in io.StringIO(out):
        line = line.strip()
        if not line or line.startswith("device") or "Device" in line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                idx = int(parts[0].strip())
                name = parts[1].strip().strip('"')
                devices.append(DeviceInfo("rocm", "rocm-smi", idx, name))
            except ValueError:
                pass
    return devices


def _detect_cambricon() -> list[DeviceInfo]:
    """Detect Cambricon MLU via cnmon."""
    out = _run_tool("cnmon", ["info", "--summary"])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for match in re.finditer(r"Device\s+(\d+)", out):
        idx = int(match.group(1))
        devices.append(DeviceInfo("cambricon", "cnmon", idx, f"MLU-{idx}"))
    return devices


def _detect_moore_threads() -> list[DeviceInfo]:
    """Detect Moore Threads GPU via mt-gpu-smi."""
    out = _run_tool("mt-gpu-smi", ["--query", "-d", "device"])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for match in re.finditer(r"Device\s+(\d+)", out):
        idx = int(match.group(1))
        devices.append(DeviceInfo("moore_threads", "mt-gpu-smi", idx, f"MTT-{idx}"))
    return devices


def _detect_biren() -> list[DeviceInfo]:
    """Detect Biren GPU via biren-smi."""
    out = _run_tool("biren-smi", ["--query"])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for match in re.finditer(r"Device\s+(\d+)", out):
        idx = int(match.group(1))
        devices.append(DeviceInfo("biren", "biren-smi", idx, f"BR-{idx}"))
    return devices


def _detect_enflame() -> list[DeviceInfo]:
    """Detect Enflame CloudBlazer via tops-smi."""
    out = _run_tool("tops-smi", ["--show"])
    if not out:
        return []
    devices: list[DeviceInfo] = []
    for match in re.finditer(r"Device\s+(\d+)", out):
        idx = int(match.group(1))
        devices.append(DeviceInfo("enflame", "tops-smi", idx, f"Tops-{idx}"))
    return devices


_DETECTORS: list[tuple[str, Any]] = [
    ("nvidia", _detect_nvidia),
    ("ascend", _detect_ascend),
    ("rocm", _detect_rocm),
    ("cambricon", _detect_cambricon),
    ("moore_threads", _detect_moore_threads),
    ("biren", _detect_biren),
    ("enflame", _detect_enflame),
]


def detect_devices() -> list[DeviceInfo]:
    """Auto-detect all available compute devices on the host."""
    for vendor, detector in _DETECTORS:
        devices = detector()
        if devices:
            return devices
    return []


def print_hardware_info() -> None:
    """Detect hardware and print a formatted summary to stdout."""
    devices = detect_devices()
    if not devices:
        print("[Device Monitor] No supported compute devices detected on this host.")
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

def _parse_nvidia_smi(snapshot: Snapshot) -> None:
    """Parse nvidia-smi CSV output for all devices."""
    out = _run_tool("nvidia-smi", [
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return
    for line in io.StringIO(out):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            idx = int(parts[0])
            sample = DeviceSample(index=idx, name=parts[1], vendor="nvidia")
            sample.utilization_percent = _to_float(parts[2])
            sample.memory_used_mb = _to_float(parts[3])
            sample.memory_total_mb = _to_float(parts[4])
            sample.power_watts = _to_float(parts[5])
            sample.temperature_celsius = _to_float(parts[6])
            snapshot.devices.append(sample)
        except (ValueError, IndexError):
            pass


def _parse_npu_smi(snapshot: Snapshot) -> None:
    """Parse npu-smi info table output for all Ascend devices."""
    out = _run_tool("npu-smi", ["info"])
    if not out:
        out = _run_tool("npu-smi", ["info"], use_sudo=True)
    if not out:
        return

    device_rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        if line.count("|") != 4:
            continue
        # Try row 2 first (chip + AICore + HBM): second column is empty
        m2 = re.match(r"\|\s*\d+\s+\|", line)
        if m2 and device_rows:
            parts = line.split("|")
            if len(parts) >= 4:
                metrics_part = parts[3]
                nums = re.findall(r"([\d.]+)", metrics_part)
                aicore = float(nums[0]) if len(nums) > 0 else None
                hbm_used = float(nums[3]) if len(nums) > 3 else None
                hbm_total = float(nums[4]) if len(nums) > 4 else None
            else:
                aicore, hbm_used, hbm_total = None, None, None
            device_rows[-1].update({"aicore": aicore, "hbm_used": hbm_used, "hbm_total": hbm_total})
            continue

        # Match device row 1: index + name (row 2 already skipped by above)
        m = re.match(r"\|\s*(\d+)\s+(\S+)", line)
        if m:
            idx = int(m.group(1))
            name = m.group(2)
            parts = line.split("|")
            if len(parts) >= 4:
                metrics_part = parts[3]
                nums = re.findall(r"([\d.]+)", metrics_part)
                power = float(nums[0]) if len(nums) > 0 else None
                temp = float(nums[1]) if len(nums) > 1 else None
            else:
                power, temp = None, None
            device_rows.append({"index": idx, "name": name, "power": power, "temp": temp})

    for dev in device_rows:
        sample = DeviceSample(
            index=dev["index"],
            name=dev.get("name", f"Ascend-{dev['index']}"),
            vendor="ascend",
        )
        sample.utilization_percent = dev.get("aicore")
        sample.memory_used_mb = dev.get("hbm_used")
        sample.memory_total_mb = dev.get("hbm_total")
        sample.power_watts = dev.get("power")
        sample.temperature_celsius = dev.get("temp")
        snapshot.devices.append(sample)


def _parse_rocm_smi(snapshot: Snapshot) -> None:
    """Parse rocm-smi output for all AMD/DCU devices."""
    out = _run_tool("rocm-smi", [
        "--showid", "--showuse", "--showmeminfo", "vram",
        "--showpower", "--showtemp",
        "--csv", "--noreport",
    ])
    if not out:
        return
    reader = csv.DictReader(io.StringIO(out))
    for row in reader:
        try:
            idx = int(row.get("device", 0))
            sample = DeviceSample(index=idx, name=f"DCU-{idx}", vendor="rocm")
            sample.utilization_percent = _to_float(row.get("GPU Use (%)"))
            sample.memory_used_mb = _to_float(row.get("VRAM Total Used (MB)"))
            sample.memory_total_mb = _to_float(row.get("VRAM Total (MB)"))
            sample.power_watts = _to_float(row.get("Power Draw (W)"))
            sample.temperature_celsius = _to_float(row.get("Temperature (C)"))
            snapshot.devices.append(sample)
        except (ValueError, TypeError):
            pass


def _parse_cnmon(snapshot: Snapshot) -> None:
    """Parse cnmon output for Cambricon MLU devices."""
    out = _run_tool("cnmon", ["info", "--summary"])
    if not out:
        return
    current_device = -1
    for line in out.splitlines():
        m = re.search(r"Device\s+(\d+)", line)
        if m:
            current_device = int(m.group(1))
            continue
        if current_device < 0:
            continue
        existing = [d for d in snapshot.devices if d.index == current_device]
        if existing:
            sample = existing[0]
        else:
            sample = DeviceSample(
                index=current_device,
                name=f"MLU-{current_device}",
                vendor="cambricon",
            )
            snapshot.devices.append(sample)

        m = re.search(r"Utilization\s*:\s*(\d+\.?\d*)%", line)
        if m:
            sample.utilization_percent = float(m.group(1))
        m = re.search(r"Memory\s*Used\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_used_mb = float(m.group(1))
        m = re.search(r"Memory\s*Total\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_total_mb = float(m.group(1))
        m = re.search(r"Power\s*:\s*(\d+\.?\d*)\s*W", line, re.IGNORECASE)
        if m:
            sample.power_watts = float(m.group(1))
        m = re.search(r"Temperature\s*:\s*(\d+\.?\d*)", line, re.IGNORECASE)
        if m:
            sample.temperature_celsius = float(m.group(1))


def _parse_mt_gpu_smi(snapshot: Snapshot) -> None:
    """Parse mt-gpu-smi output for Moore Threads devices."""
    out = _run_tool("mt-gpu-smi", ["--query", "-d", "device_stats"])
    if not out:
        return
    current_device = -1
    for line in out.splitlines():
        m = re.search(r"Device\s+(\d+)", line)
        if m:
            current_device = int(m.group(1))
            sample = DeviceSample(index=current_device, name=f"MTT-{current_device}", vendor="moore_threads")
            snapshot.devices.append(sample)
            continue
        if current_device < 0:
            continue
        dev_samples = [d for d in snapshot.devices if d.index == current_device]
        if not dev_samples:
            continue
        sample = dev_samples[-1]
        m = re.search(r"GPU\s+Util\s*:\s*(\d+\.?\d*)%", line, re.IGNORECASE)
        if m:
            sample.utilization_percent = float(m.group(1))
        m = re.search(r"Memory\s+Used\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_used_mb = float(m.group(1))
        m = re.search(r"Memory\s+Total\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_total_mb = float(m.group(1))
        m = re.search(r"Power\s*:\s*(\d+\.?\d*)\s*W", line, re.IGNORECASE)
        if m:
            sample.power_watts = float(m.group(1))
        m = re.search(r"Temperature\s*:\s*(\d+\.?\d*)", line, re.IGNORECASE)
        if m:
            sample.temperature_celsius = float(m.group(1))


def _parse_biren_smi(snapshot: Snapshot) -> None:
    """Parse biren-smi output for Biren devices."""
    out = _run_tool("biren-smi", ["--query"])
    if not out:
        return
    current_device = -1
    for line in out.splitlines():
        m = re.search(r"Device\s+(\d+)", line)
        if m:
            current_device = int(m.group(1))
            sample = DeviceSample(index=current_device, name=f"BR-{current_device}", vendor="biren")
            snapshot.devices.append(sample)
            continue
        if current_device < 0:
            continue
        dev_samples = [d for d in snapshot.devices if d.index == current_device]
        if not dev_samples:
            continue
        sample = dev_samples[-1]
        m = re.search(r"Utilization\s*:\s*(\d+\.?\d*)%", line, re.IGNORECASE)
        if m:
            sample.utilization_percent = float(m.group(1))
        m = re.search(r"Memory\s+Used\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_used_mb = float(m.group(1))
        m = re.search(r"Memory\s+Total\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_total_mb = float(m.group(1))
        m = re.search(r"Power\s*:\s*(\d+\.?\d*)\s*W", line, re.IGNORECASE)
        if m:
            sample.power_watts = float(m.group(1))
        m = re.search(r"Temperature\s*:\s*(\d+\.?\d*)", line, re.IGNORECASE)
        if m:
            sample.temperature_celsius = float(m.group(1))


def _parse_tops_smi(snapshot: Snapshot) -> None:
    """Parse tops-smi output for Enflame devices."""
    out = _run_tool("tops-smi", ["--show"])
    if not out:
        return
    current_device = -1
    for line in out.splitlines():
        m = re.search(r"Device\s+(\d+)", line)
        if m:
            current_device = int(m.group(1))
            sample = DeviceSample(index=current_device, name=f"Tops-{current_device}", vendor="enflame")
            snapshot.devices.append(sample)
            continue
        if current_device < 0:
            continue
        dev_samples = [d for d in snapshot.devices if d.index == current_device]
        if not dev_samples:
            continue
        sample = dev_samples[-1]
        m = re.search(r"Utilization\s*:\s*(\d+\.?\d*)%", line, re.IGNORECASE)
        if m:
            sample.utilization_percent = float(m.group(1))
        m = re.search(r"Memory\s+Used\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_used_mb = float(m.group(1))
        m = re.search(r"Memory\s+Total\s*:\s*(\d+\.?\d*)\s*MB", line, re.IGNORECASE)
        if m:
            sample.memory_total_mb = float(m.group(1))
        m = re.search(r"Power\s*:\s*(\d+\.?\d*)\s*W", line, re.IGNORECASE)
        if m:
            sample.power_watts = float(m.group(1))
        m = re.search(r"Temperature\s*:\s*(\d+\.?\d*)", line, re.IGNORECASE)
        if m:
            sample.temperature_celsius = float(m.group(1))


_VENDOR_PARSERS: dict[str, Any] = {
    "nvidia": _parse_nvidia_smi,
    "ascend": _parse_npu_smi,
    "rocm": _parse_rocm_smi,
    "cambricon": _parse_cnmon,
    "moore_threads": _parse_mt_gpu_smi,
    "biren": _parse_biren_smi,
    "enflame": _parse_tops_smi,
}


def _sample_once(vendor: str, snapshot_index: int, elapsed: float) -> Snapshot:
    """Take a single snapshot of all devices using the detected vendor tool."""
    snapshot = Snapshot(index=snapshot_index, elapsed_seconds=round(elapsed, 6))
    parser = _VENDOR_PARSERS.get(vendor)
    if parser is None:
        snapshot.error = {"type": "UnknownVendor", "message": f"No parser for vendor: {vendor}"}
    else:
        try:
            parser(snapshot)
        except Exception as exc:
            snapshot.error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
    return snapshot