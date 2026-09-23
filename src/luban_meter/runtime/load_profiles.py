"""Load profiles share one admission scheduler."""

from __future__ import annotations

from abc import ABC, abstractmethod

from luban_meter.core.parameters import Parameter
from luban_meter.core.registry import Registry
from luban_meter.runtime.scheduling_strategies import (
    ImmediateStrategy,
    RateStrategy,
    TraceReplayStrategy,
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
        return RateStrategy(self.parameters)


@profiles.register("concurrent")
class ConcurrentProfile(Profile):
    parameters = {
        "max_concurrency": Parameter(int, 8, minimum=1, matrix=True),
    }

    def create_strategy(self):
        return ImmediateStrategy()


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
