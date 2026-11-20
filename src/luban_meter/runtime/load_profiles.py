"""Load profiles share one admission scheduler."""

from __future__ import annotations

from abc import ABC, abstractmethod

from luban_meter.core.parameters import Parameter
from luban_meter.core.registry import Registry
from luban_meter.runtime.scheduling_strategies import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    ImmediateStrategy,
    StepStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
    TraceReplayStrategy,
    rate_strategy,
)


class Profile(ABC):
    summary_fields = (
        "request_rate",
        "arrival_process",
        "burstiness",
        "max_concurrency",
        "batch_size",
        "replay_speed",
    )
    execution_mode = "requests"
    parameters: dict[str, Parameter] = {}

    def __init__(self, parameters):
        self.parameters = parameters

    @classmethod
    def parameter_schema(cls):
        schema = {}
        for base in reversed(cls.__mro__):
            schema.update(base.__dict__.get("parameters", {}))
        return schema

    @classmethod
    def validate(cls, parameters):
        """Check profile parameter relationships at planning time."""

    @abstractmethod
    def create_strategy(self):
        """Create isolated scheduling rules for one collection phase."""

    def create_warmup_strategy(self):
        """Warm up without consuming formal arrival or trace state."""
        return ImmediateStrategy()

    @property
    def workers(self) -> int:
        return self.parameters.get("max_concurrency", 1)


profiles = Registry("profile", Profile)


@profiles.register("rate")
class RateProfile(Profile):
    parameters = {
        "request_rate": Parameter(
            float, 1.0, minimum=1e-12, allow_infinity=True, matrix=True
        ),
        "arrival_process": Parameter(
            str, "constant", choices=("constant", "poisson", "gamma")
        ),
        "burstiness": Parameter(float, 1.0, minimum=1e-12),
        "max_concurrency": Parameter(int, 32, minimum=1, matrix=True),
    }

    def create_strategy(self):
        return rate_strategy(self.parameters)


class _ArrivalProfile(Profile):
    """Finite arrival rates with explicit distribution-specific entrypoints."""

    parameters = {
        "request_rate": Parameter(float, 1.0, minimum=1e-12, matrix=True),
        "max_concurrency": Parameter(int, 32, minimum=1, matrix=True),
    }


@profiles.register("constant")
class ConstantProfile(_ArrivalProfile):
    def create_strategy(self):
        return AsyncConstantStrategy(self.parameters)


@profiles.register("poisson")
class PoissonProfile(_ArrivalProfile):
    def create_strategy(self):
        return AsyncPoissonStrategy(self.parameters)


@profiles.register("step")
class StepProfile(Profile):
    """Deterministic staged rate escalation for capacity search."""

    summary_fields = (
        "max_concurrency",
        "step_start_rate",
        "step_rate_increment",
        "step_stage_duration",
        "step_max_rate",
    )

    parameters = {
        "step_start_rate": Parameter(
            float, 1.0, minimum=1e-12, matrix=True
        ),
        "step_rate_increment": Parameter(
            float, 1.0, minimum=0.0, matrix=True
        ),
        "step_stage_duration": Parameter(
            float, 30.0, minimum=0.001, matrix=True
        ),
        "step_max_rate": Parameter(float, None, minimum=1e-12, matrix=True),
        "max_concurrency": Parameter(int, 32, minimum=1, matrix=True),
        "stage_min_requests": Parameter(int, 20, minimum=1),
        "stage_min_slo_rate": Parameter(
            float, 0.99, minimum=0.0, maximum=1.0
        ),
        "stage_min_success_rate": Parameter(
            float, 0.99, minimum=0.0, maximum=1.0
        ),
        "stage_max_timeout_rate": Parameter(
            float, 0.01, minimum=0.0, maximum=1.0
        ),
    }

    @classmethod
    def validate(cls, parameters):
        max_rate = parameters.get("step_max_rate")
        if max_rate is None:
            return
        if float(max_rate) < float(parameters["step_start_rate"]):
            raise ValueError("step_max_rate must be >= step_start_rate")

    def create_strategy(self):
        return StepStrategy(self.parameters)


@profiles.register("concurrent")
class ConcurrentProfile(Profile):
    parameters = {
        "max_concurrency": Parameter(int, 8, minimum=1, matrix=True),
    }

    def create_strategy(self):
        return ConcurrentStrategy(self.workers)


@profiles.register("throughput")
class ThroughputProfile(Profile):
    parameters = {
        "max_concurrency": Parameter(int, 32, minimum=1, matrix=True),
    }

    def create_strategy(self):
        return ThroughputStrategy()


@profiles.register("synchronous")
class SynchronousProfile(Profile):
    parameters = {
        "max_concurrency": Parameter(int, 1, minimum=1, maximum=1),
    }

    def create_strategy(self):
        return SynchronousStrategy()


@profiles.register("batch")
class BatchProfile(Profile):
    execution_mode = "batch"
    parameters = {
        "batch_size": Parameter(int, 1, minimum=1, matrix=True),
    }

    @property
    def workers(self) -> int:
        return 1

    def create_strategy(self):
        return ImmediateStrategy()


@profiles.register("replay")
class ReplayProfile(Profile):
    execution_mode = "trace"
    parameters = {
        "replay_speed": Parameter(float, 1.0, minimum=1e-12),
        "max_concurrency": Parameter(int, 32, minimum=1, matrix=True),
    }

    def create_strategy(self):
        return TraceReplayStrategy(self.parameters["replay_speed"])
