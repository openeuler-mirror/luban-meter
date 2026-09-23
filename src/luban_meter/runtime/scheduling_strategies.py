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
    # Optional upper bound, enforced by the shared request scheduler.
    concurrency_limit = None

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


class ConcurrentStrategy(ImmediateStrategy):
    """Refill up to N concurrent slots without an arrival-time schedule.

    The scheduler owns admission and slot release. Sample preparation and
    scoring can leave gaps, so N is an upper bound, not a saturation promise.
    """

    def __init__(self, max_concurrency):
        self.concurrency_limit = max_concurrency


class ThroughputStrategy(ImmediateStrategy):
    """Send as soon as capacity permits, without a target arrival rate.

    Capacity comes from the Profile. This shares immediate admission with
    ConcurrentStrategy; it does not discover an optimal concurrency level.
    """


class SynchronousStrategy(ImmediateStrategy):
    """Allow only one active request, including after a failed request."""

    concurrency_limit = 1


class _IntervalStrategy(SchedulingStrategy):
    """Accumulate planned intervals independently of completion times."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.random = random.Random(parameters.get("seed", 42))
        self.next_offset = 0.0
        self.rate = float(parameters["request_rate"])

    @abstractmethod
    def interval(self):
        """Return the interval after the current planned arrival."""

    def plan(self, metadata):
        offset = self.next_offset
        self.next_offset += self.interval()
        return RequestPlan(offset)


class AsyncConstantStrategy(_IntervalStrategy):
    """Schedule requests at 0, 1/r, 2/r, ... seconds."""

    def interval(self):
        return 1.0 / self.rate


class AsyncPoissonStrategy(_IntervalStrategy):
    """Schedule exponential intervals after an initial request at zero."""

    def interval(self):
        return self.random.expovariate(self.rate)


class AsyncGammaStrategy(_IntervalStrategy):
    """Preserve Gamma arrivals with burstiness as the shape parameter."""

    def interval(self):
        shape = self.parameters["burstiness"]
        return self.random.gammavariate(shape, 1.0 / (self.rate * shape))


def rate_strategy(parameters):
    """Resolve legacy rate options without changing their timing facts."""
    if math.isinf(float(parameters["request_rate"])):
        # Preserve the legacy all-zero targets, including for poisson/gamma.
        return AsyncConstantStrategy(parameters)
    implementations = {
        "constant": AsyncConstantStrategy,
        "poisson": AsyncPoissonStrategy,
        "gamma": AsyncGammaStrategy,
    }
    return implementations[parameters.get("arrival_process", "constant")](
        parameters
    )


class RateStrategy(SchedulingStrategy):
    """Compatibility adapter for extensions using the former rate class."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.strategy = rate_strategy(parameters)

    def plan(self, metadata):
        return self.strategy.plan(metadata)


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
