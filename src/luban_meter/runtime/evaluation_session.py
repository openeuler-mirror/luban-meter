"""Common collection lifecycle for both public benchmark base classes."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from luban_meter.core.execution_summary import prepared_summary
from luban_meter.core.metric import compute_metrics
from luban_meter.core.parameters import public_parameters
from luban_meter.core.planning import schema_for
from luban_meter.runtime.device_monitor import DeviceMonitorDaemon
from luban_meter.runtime.load_profiles import profiles
from luban_meter.runtime.prometheus_observer import PrometheusObserver
from luban_meter.runtime.request_scheduler import (
    AdmissionStopped,
    RequestScheduler,
)
from luban_meter.utils.json_io import to_jsonable, write_json_atomic


class RunContext:
    """Injected services for extensions; no process-global active run."""

    def __init__(self, parameters, directory, case_id, stop_file):
        self.parameters = parameters
        self.directory = directory
        self.case_id = case_id
        self.stop_file = stop_file
        self.phase = "prepare"
        self.total_samples = None
        self.local = threading.local()
        self.lock = threading.Lock()
        self.request_records = []
        self.sample_records = []
        self.scheduler = None
        self.events = (directory / "raw" / "events.jsonl").open(
            "a", encoding="utf-8"
        )

    @property
    def dataset_path(self):
        return Path(self.parameters["dataset_path"])

    def stopped(self):
        return self.stop_file.exists()

    def emit(self, kind, value):
        event = {
            "kind": kind,
            "case_id": self.case_id,
            "phase": self.phase,
            "time": time.time(),
            **value,
        }
        with self.lock:
            self.events.write(json.dumps(to_jsonable(event)) + "\n")
            self.events.flush()
            if kind == "sample_finished" and self.phase == "formal":
                self.sample_records.append(dict(value))
            if kind == "request_finished" and self.phase == "formal":
                self.request_records.append(dict(value))

    def start_scheduler(self, formal):
        self.phase = "formal" if formal else "warmup"
        p = dict(self.parameters)
        if not formal:
            p.update(num_requests=None, max_duration=None)
        self.scheduler = RequestScheduler(
            profiles.get(p["profile"])(p),
            p,
            self.stopped,
            self.emit,
            warmup=not formal,
        )

    def request(self, operation, *, cost=1):
        def admitted(ticket):
            self.local.admitted = getattr(self.local, "admitted", 0) + cost
            return operation(ticket)

        try:
            return self.scheduler.run(
                admitted,
                sample_id=getattr(self.local, "sample_id", "setup"),
                cost=cost,
                metadata=getattr(self.local, "scheduling", {}),
            )
        except AdmissionStopped:
            self.local.incomplete = True
            raise

    def collect_sample(self, scenario, index, sample):
        self.local.scheduling = sample.get("scheduling", {})
        self.local.sample_id = str(index)
        self.local.admitted = 0
        self.local.incomplete = False
        row = {
            "sample_id": str(index),
            "weight": sample.get("size", 1),
            "source": dict(sample),
        }
        self.emit("sample_started", row)
        try:
            row["facts"] = dict(scenario.evaluate(sample, self))
            status = row["facts"].get("status", "success")
            row["state"] = (
                "failed"
                if status in ("failed", "service_failed", "sandbox_failed")
                else "complete"
            )
            if self.local.incomplete:
                row["state"] = (
                    "incomplete" if self.local.admitted else "not_started"
                )
        except AdmissionStopped as exc:
            row.update(
                state="incomplete" if self.local.admitted else "not_started",
                facts={},
                error=str(exc),
            )
        except Exception as exc:
            row.update(
                state="failed",
                facts={"status": "failed", "error": str(exc)},
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        self.emit("sample_finished", row)
        return row


def collect_samples(scenario, context):
    """Bound sample futures as well as actual model-request concurrency."""
    samples = enumerate(scenario.samples(context))
    rows = []
    workers = context.scheduler.capacity
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        while pending or not exhausted:
            while len(pending) < workers and not exhausted:
                if context.scheduler.stopped():
                    exhausted = True
                    break
                try:
                    index, sample = next(samples)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(
                    pool.submit(
                        context.collect_sample, scenario, index, sample
                    )
                )
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                rows.extend(future.result() for future in done)
    return sorted(rows, key=lambda row: int(row["sample_id"]))


def coverage(rows, total):
    counts = {
        state: sum(row["weight"] for row in rows if row["state"] == state)
        for state in ("complete", "failed", "incomplete", "not_started")
    }
    observed = sum(counts.values())
    counts["selected"] = total if total is not None else observed
    counts["not_started"] += max(0, counts["selected"] - observed)
    counts["completion_ratio"] = (
        counts["complete"] / counts["selected"] if counts["selected"] else None
    )
    return counts


def run_case(
    scenario_type,
    case,
    directory,
    stop_file,
    observation=None,
    *,
    summary_ack=False,
):
    p = case["parameters"]
    context = RunContext(p, directory, case["id"], stop_file)
    scenario = scenario_type()
    observer = None
    device_monitor = None
    result = {"id": case["id"], "status": "failed", "metrics": {}}
    public = public_parameters(p, schema_for(scenario_type, p))
    result["parameters"] = public
    context.emit("case_started", {"parameters": public})
    try:
        context.start_scheduler(False)
        scenario.prepare(context)
        write_json_atomic(
            directory / "raw" / f"{case['id']}.prepared.json",
            prepared_summary(context, scenario),
        )
        if summary_ack:
            acknowledgement = directory / "raw" / f"{case['id']}.prepared.ack"
            while not acknowledgement.exists():
                if context.stopped():
                    raise InterruptedError("preparation interrupted")
                time.sleep(0.01)
        context.emit(
            "case_ready",
            {
                "metadata": dict(scenario.metadata(context)),
                "total_samples": context.total_samples,
            },
        )
        context.start_scheduler(False)
        warmups = iter(scenario.samples(context))
        for index in range(p["num_warmups"]):
            if context.stopped():
                break
            try:
                sample = next(warmups)
            except StopIteration:
                warmups = iter(scenario.samples(context))
                sample = next(warmups)
            row = context.collect_sample(scenario, index, sample)
            if row["state"] not in ("complete", "not_started"):
                raise RuntimeError(
                    f"warmup failed: {row.get('error', row['facts'])}"
                )
        if p["device_exporter_url"]:
            device_monitor = DeviceMonitorDaemon(
                p["device_exporter_url"], p["device_sample_interval"]
            )
            try:
                device_monitor.start()
            except Exception as exc:
                result["device_observation"] = {"error": str(exc)}
        if observation:
            url = (
                observation.get("url")
                or p["base_url"].rstrip("/") + "/metrics"
            )
            observer = PrometheusObserver(
                url,
                observation["interval"],
                p.get("api_key", ""),
                context.emit,
            )
            observer.start()
        context.start_scheduler(True)
        started_at = time.time()
        rows = collect_samples(scenario, context)
        collection_complete = not context.stopped()
        duration = time.perf_counter() - context.scheduler.started
        ended_at = time.time()
        if observer:
            observer.stop()
            try:
                result["observation"] = observer.result()
            except Exception as exc:
                result["observation"] = {
                    "status": "failed",
                    "error": str(exc),
                    "snapshots": observer.snapshots,
                }
            observer = None
        if device_monitor:
            device_monitor.stop()
            try:
                result["device_observation"] = (
                    to_jsonable(device_monitor.summary())
                    or result.get("device_observation")
                    or {"status": "unavailable"}
                )
            except Exception as exc:
                result["device_observation"] = {"error": str(exc)}
            context.emit("device_observation", result["device_observation"])
            device_monitor = None
        counts = coverage(rows, context.total_samples)
        facts = {
            "parameters": public,
            "samples": rows,
            "requests": context.request_records,
            "metadata": dict(scenario.metadata(context)),
            "coverage": counts,
            "collection_complete": collection_complete,
            "window": {
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_s": duration,
                "request_capacity": context.scheduler.capacity,
                "peak_requests": context.scheduler.peak,
                "admitted_requests": context.scheduler.count,
                "stop_reason": context.scheduler.reason or "data_exhausted",
            },
        }
        raw_path = directory / "raw" / f"{case['id']}.json"
        write_json_atomic(raw_path, facts)
        result.update(
            coverage=counts,
            window=facts["window"],
            raw=str(raw_path),
            status=(
                "interrupted"
                if not collection_complete
                else "partial_failed"
                if counts["failed"]
                else "success"
            ),
        )
        if counts["failed"] and not counts["complete"] and collection_complete:
            result["status"] = "failed"
        if not collection_complete:
            raise InterruptedError("collection interrupted")
        result["collection_complete"] = True
        result["metrics_status"] = "pending"
        # Persist the collection boundary before optional metric work.
        write_json_atomic(
            directory / "raw" / f"{case['id']}.result.json", result
        )
        result["stop_reason"] = scenario.stop_after(facts)
        for name in p["metrics"]:
            try:
                values = compute_metrics([name], scenario.capabilities, facts)[
                    name
                ]
                result["metrics"][name] = values
                if values.get("status") in ("failed", "partial_failed"):
                    if result["status"] == "success":
                        result["status"] = "partial_failed"
            except Exception as exc:
                result["metrics"][name] = {
                    "status": "failed",
                    "error": str(exc),
                }
                if result["status"] == "success":
                    result["status"] = "partial_failed"
            write_json_atomic(
                directory / "raw" / f"{case['id']}.result.json", result
            )
        result["metrics_status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        context.emit("case_failed", {"error": result["error"]})
        if context.stopped():
            result["status"] = "interrupted"
        raw_path = directory / "raw" / f"{case['id']}.json"
        if not raw_path.exists():
            counts = coverage(context.sample_records, context.total_samples)
            write_json_atomic(
                raw_path,
                {
                    "parameters": public,
                    "samples": context.sample_records,
                    "requests": context.request_records,
                    "coverage": counts,
                    "metadata": {},
                    "error": result["error"],
                },
            )
            result.update(raw=str(raw_path), coverage=counts)
    finally:
        if device_monitor is not None:
            device_monitor.stop()
        if observer is not None:
            observer.stop()
        try:
            scenario.close()
        except Exception as exc:
            result["cleanup_error"] = str(exc)
            if result["status"] != "interrupted":
                result["status"] = "failed"
        finally:
            context.events.close()
    if not result.get("collection_complete"):
        from luban_meter.runtime.case_recovery import discard_case

        result = discard_case(
            {"id": case["id"], "parameters": public},
            directory,
            result.get("error", "collection incomplete"),
        )
    write_json_atomic(directory / "raw" / f"{case['id']}.result.json", result)
    return result
