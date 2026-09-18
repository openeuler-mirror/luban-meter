"""Import built-in declarations explicitly to populate the registries."""

from . import benchmark_metrics
from .generation_performance import generation_benchmarks
from .model_service_quality import quality_benchmarks
from .model_service_quality.common import template_adapters

__all__ = [
    "generation_benchmarks",
    "quality_benchmarks",
    "benchmark_metrics",
    "template_adapters",
]
