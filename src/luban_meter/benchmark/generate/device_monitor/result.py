"""Aggregate device monitoring snapshots into per-device summary statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.benchmark.generate.common.statistics import summarize


def _object_list(value: Any, name: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise TypeError(f"{name} must be a list of objects")
    return list(value)


def _extract_metric_values(
    snapshots: Sequence[Mapping[str, Any]],
    device_index: int,
    field: str,
) -> list[float]:
    """Extract a numeric field from all snapshots for a specific device."""
    values: list[float] = []
    for snap in snapshots:
        devices = snap.get("devices")
        if not isinstance(devices, Sequence):
            continue
        for dev in devices:
            if not isinstance(dev, Mapping):
                continue
            if dev.get("index") == device_index:
                val = dev.get(field)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    values.append(float(val))
                break
    return values


def _aggregate_device_metrics(
    snapshots: Sequence[Mapping[str, Any]],
    device_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute summary statistics for one device across all snapshots."""
    device_index = device_info["index"]
    device_name = device_info["name"]
    vendor = device_info["vendor"]

    result: dict[str, Any] = {
        "index": device_index,
        "name": device_name,
        "vendor": vendor,
    }

    metric_fields = {
        "utilization_percent": "%",
        "memory_used_mb": "MB",
        "memory_total_mb": "MB",
        "power_watts": "W",
        "temperature_celsius": "°C",
    }

    for field, unit in metric_fields.items():
        values = _extract_metric_values(snapshots, device_index, field)
        if not values:
            continue
        # memory_total is a constant, not a time-series - keep as scalar
        if field == "memory_total_mb":
            result[field] = {
                "value": values[-1],
                "unit": unit,
            }
        else:
            result[field] = summarize(values, unit)

    return result


def _aggregate_power_energy(
    snapshots: Sequence[Mapping[str, Any]],
    device_index: int,
    duration_seconds: float,
) -> dict[str, Any] | None:
    """Estimate total energy consumption (Watt-hours) for a device."""
    power_values = _extract_metric_values(snapshots, device_index, "power_watts")
    if not power_values or duration_seconds <= 0:
        return None
    avg_power = sum(power_values) / len(power_values)
    # Energy in Wh = avg_power (W) * duration (hours)
    energy_wh = avg_power * (duration_seconds / 3600)
    return {
        "avg_power_watts": round(avg_power, 3),
        "total_energy_wh": round(energy_wh, 6),
        "duration_seconds": duration_seconds,
    }


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")

    raw_snapshots = _object_list(raw_metrics.get("snapshots"), "snapshots")
    raw_metadata = raw_result.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("metadata must be an object")

    device_count = raw_metadata.get("device_count", 0)
    devices_raw = raw_metadata.get("devices", [])
    vendor = raw_metadata.get("vendor", "unknown")
    duration_seconds = float(raw_metadata.get("collect_duration", 0.0))

    if device_count == 0 or not isinstance(devices_raw, Sequence) or not devices_raw:
        # No devices detected - return empty result
        return {
            "status": "success",
            "metrics": {
                "device_count": 0,
                "devices": [],
                "message": "No supported compute devices detected on this host.",
            },
            "metadata": {
                "measurement": "device_monitor_aggregation",
                "device_count": 0,
                "vendor": vendor,
            },
            "error": None,
        }

    # Filter out error snapshots
    good_snapshots = [
        snap for snap in raw_snapshots
        if snap.get("error") is None
    ]

    if not good_snapshots:
        raise ValueError("all snapshots contain errors")

    # Aggregate per-device metrics
    devices_metrics: list[dict[str, Any]] = []
    for dev_info in devices_raw:
        if not isinstance(dev_info, Mapping):
            continue
        dev_result = _aggregate_device_metrics(good_snapshots, dev_info)
        # Add energy estimate
        energy = _aggregate_power_energy(
            good_snapshots, dev_info["index"], duration_seconds
        )
        if energy is not None:
            dev_result["energy_estimate"] = energy
        devices_metrics.append(dev_result)

    # Compute system-level summary
    system_summary: dict[str, Any] = {
        "total_devices": len(devices_metrics),
        "vendor": vendor,
        "total_snapshots": len(good_snapshots),
        "total_snapshot_count": len(raw_snapshots),
    }

    # Average utilization across all devices
    all_util: list[float] = []
    for dev in devices_metrics:
        util = dev.get("utilization_percent")
        if isinstance(util, Mapping) and util.get("mean") is not None:
            all_util.append(float(util["mean"]))
    if all_util:
        system_summary["avg_utilization_percent"] = round(
            sum(all_util) / len(all_util), 3
        )

    # Total power across all devices
    all_power_avg: list[float] = []
    for dev in devices_metrics:
        energy = dev.get("energy_estimate")
        if isinstance(energy, Mapping) and energy.get("avg_power_watts") is not None:
            all_power_avg.append(float(energy["avg_power_watts"]))
    if all_power_avg:
        system_summary["total_power_watts"] = round(sum(all_power_avg), 3)
        system_summary["total_energy_wh"] = round(
            sum(
                float(dev["energy_estimate"]["total_energy_wh"])
                for dev in devices_metrics
                if isinstance(dev.get("energy_estimate"), Mapping)
                and dev["energy_estimate"].get("total_energy_wh") is not None
            ),
            6,
        )

    metadata = dict(raw_metadata)
    metadata.update(
        {
            "measurement": "device_monitor_aggregation",
            "successful_snapshot_count": len(good_snapshots),
            "total_snapshot_count": len(raw_snapshots),
        }
    )

    return {
        "status": "success",
        "metrics": {
            "device_count": len(devices_metrics),
            "vendor": vendor,
            "devices": devices_metrics,
            "system_summary": system_summary,
        },
        "metadata": metadata,
        "error": None,
    }