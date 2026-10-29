"""Independent node_exporter sampling with auditable interval rates."""

from __future__ import annotations

import math
import threading
import time

import httpx

from luban_meter.benchmarking.generation_performance.common.prometheus import (
    parse_prometheus_text,
)

CPU_MODES = {"user", "nice", "system", "idle", "iowait", "irq",
             "softirq", "steal"}
DISK_COUNTERS = {
    "read_bytes": "node_disk_read_bytes_total",
    "write_bytes": "node_disk_written_bytes_total",
    "reads": "node_disk_reads_completed_total",
    "writes": "node_disk_writes_completed_total",
    "io_seconds": "node_disk_io_time_seconds_total",
}
NETWORK_COUNTERS = {
    key: "node_network_" + metric + "_total"
    for key, metric in [
        ("receive_bytes", "receive_bytes"),
        ("transmit_bytes", "transmit_bytes"),
        ("receive_errors", "receive_errs"),
        ("transmit_errors", "transmit_errs"),
        ("receive_dropped", "receive_drop"),
        ("transmit_dropped", "transmit_drop"),
    ]
}
FILESYSTEM_GAUGES = {
    "size_bytes": "node_filesystem_size_bytes",
    "free_bytes": "node_filesystem_free_bytes",
    "available_bytes": "node_filesystem_avail_bytes",
    "device_error": "node_filesystem_device_error",
    "readonly": "node_filesystem_readonly",
}


def finite(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value))


def _samples(metrics, name):
    return [s for s in metrics.get(name, {}).get("samples", [])
            if finite(s["value"]) and s["value"] >= 0]


def _value(metrics, name):
    samples = _samples(metrics, name)
    return samples[0]["value"] if len(samples) == 1 else None


def _groups(metrics, fields, labels=("device",)):
    result = {}
    for field, metric in fields.items():
        for sample in _samples(metrics, metric):
            identity = tuple(sample["labels"].get(k) for k in labels)
            if any(v is None for v in identity):
                continue
            group = result.setdefault(identity, dict(zip(labels, identity)))
            group[field] = sample["value"]
    return list(result.values())


def parse_host(text):
    """Keep selected original counters/gauges, not the entire response."""
    metrics = parse_prometheus_text(text)
    cpu, cores = {}, set()
    for sample in _samples(metrics, "node_cpu_seconds_total"):
        labels = sample["labels"]
        mode = labels.get("mode")
        if mode in CPU_MODES and "cpu" in labels:
            cpu[mode] = cpu.get(mode, 0.0) + sample["value"]
            cores.add(labels["cpu"])
    total = _value(metrics, "node_memory_MemTotal_bytes")
    available = _value(metrics, "node_memory_MemAvailable_bytes")
    used = (total - available if finite(total) and finite(available)
            and 0 <= available <= total else None)
    filesystems = _groups(
        metrics, FILESYSTEM_GAUGES, ("device", "mountpoint", "fstype")
    )
    for fs in filesystems:
        size, free = fs.get("size_bytes"), fs.get("free_bytes")
        good = (finite(size) and size > 0 and finite(free)
                and 0 <= free <= size and not fs.get("device_error"))
        fs["used_bytes"] = size - free if good else None
        fs["utilization"] = (100 * (size - free) / size
                             if good else None)
    uname = _samples(metrics, "node_uname_info")
    return {
        "node": uname[0]["labels"] if uname else {},
        "raw": {
            "cpu_seconds": cpu, "cpu_ids": sorted(cores),
            "disks": _groups(metrics, DISK_COUNTERS),
            "networks": _groups(metrics, NETWORK_COUNTERS),
        },
        "memory": {
            "total_bytes": total, "available_bytes": available,
            "used_bytes": used,
            "utilization": 100 * used / total
            if finite(used) and total > 0 else None,
        },
        "filesystems": filesystems,
    }


def _rates(current, previous, seconds, fields, issues):
    old = {p["device"]: p for p in previous}
    result = []
    for group in current:
        prior = old.get(group["device"], {})
        reset = any(finite(prior.get(k)) and v < prior[k]
                    for k, v in group.items() if k in fields)
        if reset:
            issues.append("counter_reset:" + group["device"])
        row = {"device": group["device"]}
        for field in fields:
            now, before = group.get(field), prior.get(field)
            value = ((now - before) / seconds if seconds and seconds > 0
                     and not reset and finite(now) and finite(before)
                     else None)
            key = ("utilization" if field == "io_seconds"
                   else field + "_per_second")
            row[key] = value * 100 if (
                field == "io_seconds" and value is not None
            ) else value
        result.append(row)
    return result


def derive_rates(snapshot, previous, seconds):
    """A reset, failed scrape or new series establishes a new baseline."""
    issues = []
    raw = snapshot["raw"]
    old = previous.get("raw", {}) if previous else {}
    now_cpu, old_cpu = raw["cpu_seconds"], old.get("cpu_seconds", {})
    valid_cpu = (seconds and seconds > 0 and "idle" in now_cpu
                 and now_cpu.keys() == old_cpu.keys()
                 and raw["cpu_ids"] == old.get("cpu_ids")
                 and all(now_cpu[k] >= old_cpu[k] for k in now_cpu))
    delta = {k: now_cpu[k] - old_cpu[k] for k in now_cpu} if valid_cpu else {}
    total = sum(delta.values())
    snapshot["cpu"] = {
        "utilization": (100 * (1 - delta["idle"] / total)
                        if total > 0 else None),
        **{k: 100 * delta[k] / total if total > 0 and k in delta else None
           for k in ("user", "system", "iowait")},
    }
    snapshot["disks"] = _rates(
        raw["disks"], old.get("disks", []), seconds, DISK_COUNTERS, issues
    )
    snapshot["networks"] = _rates(
        raw["networks"], old.get("networks", []), seconds,
        NETWORK_COUNTERS, issues,
    )
    snapshot["rate_issues"] = issues


def _stats(samples, group, field, unit, *, identity=None, rate=False):
    pairs = []
    for sample in samples:
        if rate and not sample.get("rate_in_window"):
            continue
        node = sample.get(group, {})
        if identity is not None:
            node = next((n for n in node if all(n.get(k) == v
                         for k, v in identity.items())), {})
        value = node.get(field)
        if finite(value):
            pairs.append((value, sample["interval_seconds"] if rate else 1))
    weight = sum(w for _, w in pairs)
    return {
        "mean": sum(v * w for v, w in pairs) / weight if weight else None,
        "min": min((v for v, _ in pairs), default=None),
        "max": max((v for v, _ in pairs), default=None),
        "count": len(pairs), "unit": unit,
        "covered_seconds": weight if rate else None,
    }


def summarize(samples, start, end):
    selected = []
    for original in samples:
        if "error" in original or not start <= original["finished_at"] <= end:
            continue
        s = dict(original)
        s["rate_in_window"] = (
            s.get("previous_finished_at") is not None
            and s["previous_finished_at"] >= start
        )
        selected.append(s)
    result = {
        "cpu": {k: _stats(selected, "cpu", k, "%", rate=True)
                for k in ("utilization", "user", "system", "iowait")},
        "memory": {k: _stats(selected, "memory", k,
                            "%" if k == "utilization" else "bytes")
                   for k in ("total_bytes", "available_bytes", "used_bytes",
                             "utilization")},
    }
    for group, labels, fields in [
        ("disks", ("device",), {
            "read_bytes_per_second": "bytes/s",
            "write_bytes_per_second": "bytes/s",
            "reads_per_second": "IOPS", "writes_per_second": "IOPS",
            "utilization": "%"}),
        ("networks", ("device",), {
            key + "_per_second": "bytes/s" if key.endswith("bytes")
            else "events/s" for key in NETWORK_COUNTERS}),
        ("filesystems", ("device", "mountpoint", "fstype"), {
            "size_bytes": "bytes", "used_bytes": "bytes",
            "available_bytes": "bytes", "utilization": "%"}),
    ]:
        identities = {tuple(n.get(k) for k in labels)
                      for s in selected for n in s.get(group, [])}
        result[group] = []
        for values in sorted(identities):
            identity = dict(zip(labels, values))
            result[group].append({**identity, **{
                k: _stats(selected, group, k, unit, identity=identity,
                          rate=group != "filesystems")
                for k, unit in fields.items()
            }})
    return result


class HostMonitorDaemon:
    """One host source; its failures never stop the GPU sampler."""

    def __init__(self, url, interval=1.0):
        self.url = url.rstrip("/").removesuffix("/metrics") + "/metrics"
        self.interval = interval
        self.samples = []
        self.event = threading.Event()
        self.thread = None
        self.client = httpx.Client(timeout=5, trust_env=False)
        self.previous = None
        self.previous_tick = None

    def sample(self):
        snapshot = {"started_at": time.time()}
        try:
            response = self.client.get(self.url)
            response.raise_for_status()
            snapshot.update(parse_host(response.text))
            if not (snapshot["raw"]["cpu_seconds"]
                    or snapshot["raw"]["disks"]
                    or snapshot["raw"]["networks"]
                    or snapshot["filesystems"]
                    or finite(snapshot["memory"]["total_bytes"])):
                raise ValueError("endpoint has no supported node metrics")
            tick = time.monotonic()
            seconds = tick - self.previous_tick if self.previous else None
            snapshot["interval_seconds"] = seconds
            snapshot["previous_finished_at"] = (
                self.previous["finished_at"] if self.previous else None
            )
            derive_rates(snapshot, self.previous, seconds)
            self.previous_tick = tick
            self.previous = snapshot
        except Exception as exc:
            snapshot = {"started_at": snapshot["started_at"],
                        "error": {"type": type(exc).__name__,
                                  "message": str(exc)}}
            self.previous = None
            self.previous_tick = None
        snapshot["finished_at"] = time.time()
        self.samples.append(snapshot)

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.event.wait(self.interval):
            self.sample()

    def stop(self):
        self.event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
            self.sample()
        self.client.close()

    def result(self, start, end):
        valid = [s for s in self.samples if "error" not in s]
        errors = len(self.samples) - len(valid)
        return {
            "status": ("failed" if not valid else
                       "partial_failed" if errors else "success"),
            "tool": "node_exporter", "source": self.url,
            "node": valid[0]["node"] if valid else {},
            "sample_count": len(self.samples), "valid_samples": len(valid),
            "error_count": errors, "interval_seconds": self.interval,
            "window": {"started_at": start, "ended_at": end},
            "summary": summarize(self.samples, start, end),
            "timeseries": self.samples,
        }
