"""Small public scenario interfaces, independent of CLI and storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from luban_meter.core.parameters import Parameter
from luban_meter.core.registry import Registry


class Benchmark(ABC):
    """Describe samples and their facts; the runtime owns execution."""

    summary_fields = (
        "model", "backend", "base_url", "endpoint", "data", "dataset_path",
        "profile", "num_requests", "max_duration", "num_warmups",
        "request_timeout", "task_timeout", "interrupt_grace", "metrics",
        "input_tokens", "output_tokens", "input_tokens_std",
        "output_tokens_std", "input_tokens_min", "input_tokens_max",
        "output_tokens_min", "output_tokens_max", "length_policy",
        "input_template", "answer_parser", "template",
    )
    category = ""
    description = ""
    parameters: dict[str, Parameter] = {
        "seed": Parameter(int, 42, minimum=0),
        "request_timeout": Parameter(float, 120.0, minimum=0.001),
        "interrupt_grace": Parameter(float, 60.0, minimum=0.0),
        "task_timeout": Parameter(float, 3600.0, minimum=0.001),
        "num_requests": Parameter(int, None, minimum=1),
        "max_duration": Parameter(float, None, minimum=0.001),
        "num_warmups": Parameter(int, 0, minimum=0),
        "metrics": Parameter(list, [], item_type=str),
        "profile": Parameter(str, "concurrent"),
        "device_exporter_url": Parameter(str, ""),
        "device_sample_interval": Parameter(float, 1.0, minimum=0.01),
        "host_exporter_url": Parameter(str, ""),
        "host_sample_interval": Parameter(float, 1.0, minimum=0.1),
    }
    profile_modes = frozenset({"requests"})
    default_metrics: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    fact_fields: dict = {}

    @classmethod
    def provided_fields(cls, parameters):
        return cls.fact_fields

    @classmethod
    def select_metrics(cls, parameters):
        """Resolve explicit or default metrics for the effective parameters."""
        return list(parameters["metrics"] or cls.default_metrics)

    @classmethod
    def parameter_schema(cls) -> dict[str, Parameter]:
        schema = {}
        for base in reversed(cls.__mro__):
            schema.update(base.__dict__.get("parameters", {}))
        return schema

    @classmethod
    def validate(cls, parameters: Mapping[str, Any]) -> None:
        """Check relationships not expressed by individual parameters."""

    def prepare(self, context) -> None:
        """Initialize selected dependencies in the worker, before timing."""

    @abstractmethod
    def samples(self, context) -> Iterable[Mapping[str, Any]]:
        """Produce samples lazily; sample fields remain available."""

    @abstractmethod
    def evaluate(self, sample: Mapping[str, Any], context) -> Mapping:
        """Collect facts; model calls must pass through context.request."""

    def after_warmup(self, context) -> None:
        """Finish setup before formal scheduling and observation."""

    def metadata(self, context) -> Mapping[str, Any]:
        return {}

    def stop_after(self, facts) -> str | None:
        """Optionally stop later Cases after inspecting collected facts."""
        return None

    def close(self) -> None:
        """Release worker-owned dependencies after collection."""


class GenerationPerformanceBenchmark(Benchmark):
    category = "generation_performance"


class ModelServiceQualityBenchmark(Benchmark):
    category = "model_service_quality"


benchmarks = Registry("scenario", Benchmark)
