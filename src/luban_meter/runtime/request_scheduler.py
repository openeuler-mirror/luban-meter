"""Bounded model-request admission, independent of sample semantics."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


class AdmissionStopped(Exception):  # noqa: N818 - normal control flow
    """No more requests may start; already admitted work may finish."""


@dataclass(frozen=True)
class Ticket:
    index: int
    scheduled: float
    started: float
    cost: int


def request_capacity(workers, concurrency_limit):
    """Resolve capacity for admission and transport preflight."""
    if (
        not isinstance(workers, int)
        or isinstance(workers, bool)
        or workers < 1
    ):
        raise ValueError("profile.workers must be a positive integer")
    if concurrency_limit is not None:
        if (
            not isinstance(concurrency_limit, int)
            or isinstance(concurrency_limit, bool)
            or concurrency_limit < 1
        ):
            raise ValueError("strategy concurrency_limit must be positive")
        return min(workers, concurrency_limit)
    return workers


class RequestScheduler:
    def __init__(
        self, profile, parameters, stop_requested, emit, *, warmup=False
    ):
        workers = profile.workers
        self.profile = profile
        self.parameters = parameters
        self.stop_requested = stop_requested
        self.emit = emit
        self.started = time.perf_counter()
        self.strategy = (
            profile.create_warmup_strategy()
            if warmup
            else profile.create_strategy()
        )
        self.capacity = request_capacity(
            workers, self.strategy.concurrency_limit
        )
        self.count = 0
        self.active = 0
        self.peak = 0
        self.reason = None
        self.lock = threading.Condition()

    def stopped(self) -> bool:
        if self.stop_requested():
            self.reason = "user_interrupt"
        limit = self.parameters.get("max_duration")
        if limit is not None and time.perf_counter() - self.started >= limit:
            self.reason = self.reason or "max_duration"
        count = self.parameters.get("num_requests")
        if count is not None and self.count >= count:
            self.reason = self.reason or "num_requests"
        return self.reason is not None

    def remaining(self, requested: int) -> int:
        limit = self.parameters.get("num_requests")
        return (
            requested
            if limit is None
            else min(requested, max(0, limit - self.count))
        )

    def run(self, operation, *, sample_id, cost=1, metadata=None):
        with self.lock:
            plan = self.strategy.plan(metadata or {})
            if plan.offset_s is not None and (
                isinstance(plan.offset_s, bool)
                or not isinstance(plan.offset_s, (int, float))
                or not math.isfinite(plan.offset_s)
                or plan.offset_s < 0
            ):
                raise ValueError("planned offset must be finite and >= 0")
            while True:
                if self.stopped() or self.remaining(cost) < cost:
                    raise AdmissionStopped(self.reason or "request budget")
                delay = (
                    self.started + plan.offset_s - time.perf_counter()
                    if plan.offset_s is not None
                    else 0.0
                )
                if self.active < self.capacity and self.strategy.ready(
                    plan, time.perf_counter() - self.started
                ):
                    break
                self.lock.wait(timeout=min(max(delay, 0.01), 0.05))
            now = time.perf_counter()
            scheduled = (
                self.started + plan.offset_s
                if plan.offset_s is not None
                else now
            )
            ticket = Ticket(self.count, scheduled, now, cost)
            self.strategy.admitted(plan)
            self.lock.notify_all()
            self.count += cost
            self.active += 1
            self.peak = max(self.peak, self.active)
        record = {
            "sample_id": sample_id,
            "request_id": f"request-{ticket.index}",
            "logical_requests": cost,
            "scheduled_offset_s": ticket.scheduled - self.started,
            "admission_offset_s": ticket.started - self.started,
            "start_offset_s": ticket.started - self.started,
            "scheduling_delay_s": max(0.0, ticket.started - ticket.scheduled),
            "scheduling": dict(metadata or {}),
        }
        self.emit("request_started", record)
        try:
            result = operation(ticket)
            record["status"] = "success"
            if isinstance(result, dict):
                record["response"] = result
                submitted = result.get("submitted_at_monotonic_s")
                if submitted is not None:
                    record["submission_offset_s"] = submitted - self.started
                    record["submission_delay_s"] = submitted - ticket.scheduled
                if result.get("status") in ("failed", "service_failed"):
                    record["status"] = "failed"
            return result
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            record["end_offset_s"] = time.perf_counter() - self.started
            try:
                with self.lock:
                    self.active -= 1
                    try:
                        self.strategy.completed(record)
                    finally:
                        self.lock.notify_all()
            finally:
                self.emit("request_finished", record)
