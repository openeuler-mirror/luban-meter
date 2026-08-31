"""Parse Prometheus Exposition Format text into structured data."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"                         # metric name
    r"(?:\{(?P<labels>[^}]*)\})?"                                   # optional labels
    r"\s+(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"      # numeric value
    r"(?:\s+(?P<timestamp>[0-9]+(?:\.[0-9]+)?))?"                   # optional timestamp
    r"\s*$"
)
_LABEL_PAIR = re.compile(
    r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)'   # label key
    r'="(?P<value>(?:[^"\\]|\\.)*)"'      # quoted label value
)
_HISTOGRAM_SUFFIXES = {"_bucket", "_sum", "_count"}


def _parse_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for match in _LABEL_PAIR.finditer(raw):
        labels[match.group("key")] = match.group("value")
    return labels


def _histogram_base(name: str) -> str | None:
    for suffix in _HISTOGRAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _collect_histogram_types(text: str) -> set[str]:
    """Scan TYPE lines to discover which metrics are histograms."""
    histogram_types: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("# TYPE "):
            parts = line[6:].split()
            if len(parts) >= 2 and parts[1] == "histogram":
                histogram_types.add(parts[0])
    return histogram_types


def parse_prometheus_text(text: str) -> dict[str, Any]:
    """Parse Prometheus Exposition Format text.

    Returns a dict keyed by metric name, each value containing:
      - type: "counter" | "gauge" | "histogram" | "untyped"
      - help: help text or None
      - samples: list of {"labels": {...}, "value": float, "_metric_name": str}
    """
    histogram_types = _collect_histogram_types(text)

    metrics: dict[str, Any] = {}
    histogram_samples: dict[str, list[dict[str, Any]]] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            # Record HELP/TYPE metadata
            if line.startswith("# HELP "):
                parts = line[7:].split(None, 1)
                if parts:
                    name = parts[0]
                    help_text = parts[1] if len(parts) > 1 else ""
                    metrics.setdefault(name, {})["help"] = help_text
                    metrics.setdefault(name, {}).setdefault("samples", [])
            elif line.startswith("# TYPE "):
                parts = line[6:].split()
                if len(parts) >= 2:
                    name = parts[0]
                    mtype = parts[1]
                    metrics.setdefault(name, {})["type"] = mtype
                    metrics.setdefault(name, {}).setdefault("samples", [])
            continue

        match = _METRIC_LINE.match(line)
        if not match:
            continue

        name = match.group("name")
        raw_labels = match.group("labels")
        value = float(match.group("value"))

        labels = _parse_labels(raw_labels) if raw_labels else {}
        sample = {"labels": labels, "value": value, "_metric_name": name}

        hist_base = _histogram_base(name)
        if hist_base is not None and hist_base in histogram_types:
            histogram_samples.setdefault(hist_base, []).append(sample)
            continue

        metrics.setdefault(name, {}).setdefault("samples", []).append(sample)
        metrics.setdefault(name, {}).setdefault("type", "untyped")

    # Merge histogram sub-metrics back into parent
    for base_name, samples in histogram_samples.items():
        if base_name in metrics:
            metrics[base_name]["samples"].extend(samples)
        else:
            metrics[base_name] = {
                "type": "histogram",
                "help": None,
                "samples": samples,
            }

    return metrics


def select_metric(
    metrics: dict[str, Any],
    name: str,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Select a single metric by name and optional label filter."""
    entry = metrics.get(name)
    if entry is None:
        return None
    if labels is None:
        return entry
    filtered = [
        s for s in entry.get("samples", [])
        if all(s.get("labels", {}).get(k) == v for k, v in labels.items())
    ]
    return {"type": entry.get("type"), "help": entry.get("help"), "samples": filtered}


def gauge_value(entry: dict[str, Any] | None) -> float | None:
    """Extract the scalar value from a Gauge metric entry."""
    if entry is None:
        return None
    samples = entry.get("samples")
    if not samples:
        return None
    return float(samples[0]["value"])


def counter_value(entry: dict[str, Any] | None) -> float | None:
    """Extract the scalar value from a Counter metric entry."""
    return gauge_value(entry)


def histogram_quantiles(
    samples: list[dict[str, Any]],
    quantiles: tuple[float, ...] = (0.50, 0.90, 0.99),
) -> dict[str, float | None]:
    """Estimate quantiles from Prometheus histogram bucket samples.

    Accepts the raw ``samples`` list from a histogram metric entry.
    Returns a dict mapping quantile key (e.g. "p50") to the estimated value
    in the same unit as the histogram buckets, or None if the data is
    insufficient.
    """
    buckets: list[tuple[float, float]] = []
    total_count: float | None = None
    total_sum: float | None = None

    for sample in samples:
        metric_name = str(sample.get("_metric_name", ""))
        le = sample.get("labels", {}).get("le")
        if le is not None:
            buckets.append((float(le), float(sample["value"])))
        elif metric_name.endswith("_sum"):
            total_sum = float(sample["value"])
        elif metric_name.endswith("_count"):
            total_count = float(sample["value"])

    if not buckets:
        return {f"p{int(q*100)}": None for q in quantiles}

    buckets.sort(key=lambda x: x[0])
    if total_count is None:
        total_count = buckets[-1][1]

    if total_count <= 0:
        return {f"p{int(q*100)}": None for q in quantiles}

    result: dict[str, float | None] = {}
    for q in quantiles:
        target = q * total_count
        prev_le = 0.0
        prev_count = 0.0
        found = False
        for le, count in buckets:
            if count >= target:
                ratio = (target - prev_count) / (count - prev_count) if count > prev_count else 0.0
                result[f"p{int(q*100)}"] = prev_le + ratio * (le - prev_le)
                found = True
                break
            prev_le = le
            prev_count = count
        if not found:
            result[f"p{int(q*100)}"] = None
    result["_sum"] = total_sum
    result["_count"] = total_count
    return result