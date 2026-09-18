"""Scheduling rules; no sleeping, transport calls or concurrency queues."""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestPlan:
    offset_s: float | None
    order: int | None = None


class SchedulingStrategy(ABC):
    @abstractmethod
    def plan(self, metadata):
        """Bind one request to its immutable relative submission time."""

    def ready(self, plan, elapsed):
        return plan.offset_s is None or elapsed >= plan.offset_s

    def admitted(self, plan):
        """Optional notification after the scheduler accepts a request."""

    def completed(self, record):
        """Optional feedback, serialized by the scheduler."""


class ImmediateStrategy(SchedulingStrategy):
    def plan(self, metadata):
        return RequestPlan(None)


class RateStrategy(SchedulingStrategy):
    def __init__(self, parameters):
        self.parameters = parameters
        self.random = random.Random(parameters.get("seed", 42))
        self.next_offset = 0.0

    def plan(self, metadata):
        offset = self.next_offset
        rate = self.parameters["request_rate"]
        process = self.parameters.get("arrival_process", "constant")
        if math.isinf(rate):
            interval = 0.0
        elif process == "poisson":
            interval = self.random.expovariate(rate)
        elif process == "gamma":
            shape = self.parameters["burstiness"]
            interval = self.random.gammavariate(shape, 1.0 / (rate * shape))
        else:
            interval = 1.0 / rate
        self.next_offset += interval
        return RequestPlan(offset)


class TraceReplayStrategy(SchedulingStrategy):
    def __init__(self, speed):
        self.speed = speed
        self.next_order = 0

    def plan(self, metadata):
        offset = metadata.get("offset_s")
        order = metadata.get("trace_index")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(offset)
            or offset < 0
            or isinstance(order, bool)
            or not isinstance(order, int)
            or order < 0
        ):
            raise ValueError("replay requires valid offset_s and trace_index")
        return RequestPlan(offset / self.speed, order)

    def ready(self, plan, elapsed):
        return plan.order == self.next_order and super().ready(plan, elapsed)

    def admitted(self, plan):
        self.next_order += 1
