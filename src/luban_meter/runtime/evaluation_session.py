"""Common collection lifecycle for both public benchmark base classes."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path

from luban_meter.core.execution_summary import prepared_summary
from luban_meter.core.metric import compute_metrics
from luban_meter.core.parameters import public_parameters
from luban_meter.core.planning import schema_for
from luban_meter.core.token_totals import request_total
from luban_meter.log import get_logger, log_context
from luban_meter.runtime.device_monitor import DeviceMonitorDaemon
from luban_meter.runtime.host_monitor import HostMonitorDaemon
from luban_meter.runtime.load_profiles import profiles
from luban_meter.runtime.prometheus_observer import PrometheusObserver
from luban_meter.runtime.request_scheduler import (
    AdmissionStopped,
    RequestScheduler,
)
from luban_meter.utils.json_io import to_jsonable, write_json_atomic

logger = get_logger(__name__)


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

    def request(self, operation, *, cost=1, metadata=None):
        def admitted(ticket):
            self.local.admitted = getattr(self.local, "admitted", 0) + cost
            return operation(ticket)

        # Per-call metadata lets one sample describe each of its several
        # requests, which a multi-turn sample needs to record which turn
        # a call belongs to.  Sample-level metadata still applies.
        scheduling = dict(getattr(self.local, "scheduling", {}) or {})
        scheduling.update(metadata or {})
        try:
            return self.scheduler.run(
                admitted,
                sample_id=getattr(self.local, "sample_id", "setup"),
                cost=cost,
                metadata=scheduling,
            )
        except AdmissionStopped:
            logger.debug("Request admission stopped")
            self.local.incomplete = True
            raise

    def collect_sample(self, scenario, index, sample):
        with log_context(sample=str(index), phase=self.phase):
            return self._collect_sample(scenario, index, sample)

    def _collect_sample(self, scenario, index, sample):
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
            if row["state"] == "failed":
                logger.error(
                    "Sample returned failure: status=%s error=%s",
                    status, row["facts"].get("error", "see raw facts"),
                )
        except AdmissionStopped as exc:
            logger.debug("Sample admission stopped")
            row.update(
                state="incomplete" if self.local.admitted else "not_started",
                facts={},
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Sample evaluation failed")
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
                    logger.debug("Sample iterator exhausted")
                    exhausted = True
                    break
                pending.add(
                    pool.submit(
                        copy_context().run,
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
    with log_context(case=case["id"], phase="prepare"):
        logger.info("Case started: scenario=%s", scenario_type.__name__)
        result = _run_case(
            scenario_type, case, directory, stop_file, observation,
            summary_ack=summary_ack,
        )
        logger.info("Case finished: status=%s", result["status"],
                    extra={"phase": "complete"})
        return result


def _run_case(
    scenario_type, case, directory, stop_file, observation, *, summary_ack
):
    p = case["parameters"]
    context = RunContext(p, directory, case["id"], stop_file)
    scenario = scenario_type()
    observer = None
    device_monitor = None
    host_monitor = None
    result = {"id": case["id"], "status": "failed", "metrics": {}}
    public = public_parameters(p, schema_for(scenario_type, p))
    result["parameters"] = public
    context.emit("case_started", {"parameters": public})
    try:
        logger.info("Preparation started")
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
        logger.info("Preparation finished: samples=%s", context.total_samples)
        context.start_scheduler(False)
        logger.info("Warmup started: samples=%s", p["num_warmups"],
                    extra={"phase": "warmup"})
        warmups = iter(scenario.samples(context))
        for index in range(p["num_warmups"]):
            if context.stopped():
                break
            try:
                sample = next(warmups)
            except StopIteration:
                logger.debug("Restarting warmup sample iterator")
                warmups = iter(scenario.samples(context))
                sample = next(warmups)
            row = context.collect_sample(scenario, index, sample)
            if row["state"] not in ("complete", "not_started"):
                raise RuntimeError(
                    f"warmup failed: {row.get('error', row['facts'])}"
                )
        logger.info("Warmup finished", extra={"phase": "warmup"})
        if p["device_exporter_url"]:
            device_monitor = DeviceMonitorDaemon(
                p["device_exporter_url"], p["device_sample_interval"]
            )
            try:
                device_monitor.start()
            except Exception as exc:
                logger.warning("Device monitor start failed", exc_info=True)
                result["device_observation"] = {"error": str(exc)}
        if p.get("host_exporter_url"):
            host_monitor = HostMonitorDaemon(
                p["host_exporter_url"], p["host_sample_interval"]
            )
            host_monitor.start()
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
        logger.info("Collection started", extra={"phase": "formal"})
        context.start_scheduler(True)
        started_at = time.time()
        rows = collect_samples(scenario, context)
        collection_complete = not context.stopped()
        duration = time.perf_counter() - context.scheduler.started
        ended_at = time.time()
        logger.info("Collection finished: samples=%s requests=%s reason=%s",
                    len(rows), context.scheduler.count,
                    context.scheduler.reason or "data_exhausted",
                    extra={"phase": "formal"})
        if observer:
            observer.stop()
            try:
                result["observation"] = observer.result()
            except Exception as exc:
                logger.warning("Prometheus summary failed", exc_info=True)
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
                logger.warning("Device summary failed", exc_info=True)
                result["device_observation"] = {"error": str(exc)}
            context.emit("device_observation", result["device_observation"])
            device_monitor = None
        if host_monitor:
            host_monitor.stop()
            result["host_observation"] = host_monitor.result(
                started_at, ended_at
            )
            context.emit("host_observation", result["host_observation"])
            host_monitor = None
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
        logger.info("Raw facts saved: %s", raw_path,
                    extra={"phase": "formal"})
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
        performance = scenario.category == "generation_performance"
        if performance:
            result["total_tokens"] = request_total(facts["requests"], rows)
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
        for name in dict.fromkeys(p["metrics"]):
            with log_context(phase="metrics"):
                logger.info("Metric started: %s", name)
                try:
                    values = compute_metrics(
                        [name], scenario.capabilities, facts
                    )[name]
                    result["metrics"][name] = values
                    if values.get("status") in ("failed", "partial_failed"):
                        logger.error("Metric failed: %s status=%s",
                                     name, values["status"])
                        if result["status"] == "success":
                            result["status"] = "partial_failed"
                    logger.info("Metric finished: %s", name)
                except Exception as exc:
                    logger.exception("Metric computation failed: %s", name)
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
        logger.exception("Case execution failed",
                         extra={"phase": context.phase})
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
        logger.info("Case cleanup started", extra={"phase": "cleanup"})
        if host_monitor is not None:
            host_monitor.stop()
        if device_monitor is not None:
            device_monitor.stop()
        if observer is not None:
            observer.stop()
        try:
            scenario.close()
        except Exception as exc:
            logger.exception("Case cleanup failed", extra={"phase": "cleanup"})
            result["cleanup_error"] = str(exc)
            if result["status"] != "interrupted":
                result["status"] = "failed"
        finally:
            context.events.close()
        logger.info("Case cleanup finished", extra={"phase": "cleanup"})
    if not result.get("collection_complete"):
        from luban_meter.runtime.case_recovery import discard_case

        result = discard_case(
            {"id": case["id"], "parameters": public},
            directory,
            result.get("error", "collection incomplete"),
        )
        logger.warning("Incomplete Case discarded",
                       extra={"phase": "complete"})
    write_json_atomic(directory / "raw" / f"{case['id']}.result.json", result)
    logger.info("Case result saved", extra={"phase": "complete"})
    return result
