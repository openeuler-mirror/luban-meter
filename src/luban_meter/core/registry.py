"""Discover and resolve Benchmark implementations."""

from __future__ import annotations

import re
from pathlib import Path

from luban_meter.core.config import load_benchmark_config
from luban_meter.core.errors import ConfigurationError, UnknownBenchmarkError
from luban_meter.core.models import BenchmarkSpec, ResolvedRun, RunRequest

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MODULES = {
    "generate": "Large-model generation benchmarks",
    "inference": "Online-service model evaluation benchmarks",
}


class BenchmarkRegistry:
    def __init__(self, benchmark_dir: Path | None = None) -> None:
        self._benchmark_dir = (
            benchmark_dir or Path(__file__).parents[1] / "benchmark"
        )

    def modules(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(_MODULES.items()))

    def list_benchmarks(self, module: str) -> tuple[str, ...]:
        self._validate_name("module", module)
        self._validate_module(module)
        benchmarks = []
        for entry in self._benchmark_dir.glob(f"{module}/*/benchmark.py"):
            benchmark_dir = entry.parent
            if (benchmark_dir / "result.py").is_file():
                benchmarks.append(benchmark_dir.name)
        return tuple(sorted(benchmarks))

    def resolve(self, request: RunRequest) -> ResolvedRun:
        for field in ("module", "benchmark"):
            self._validate_name(field, getattr(request, field))
        self._validate_module(request.module)
        benchmark_dir = self._benchmark_dir / request.module / request.benchmark
        benchmark_entry = benchmark_dir / "benchmark.py"
        result_handler = benchmark_dir / "result.py"
        if not benchmark_entry.is_file() or not result_handler.is_file():
            raise UnknownBenchmarkError(request.module, request.benchmark)
        return ResolvedRun(
            request=request,
            benchmark=BenchmarkSpec(
                module=request.module,
                benchmark=request.benchmark,
                benchmark_entry=benchmark_entry,
                result_handler=result_handler,
            ),
            parameters=load_benchmark_config(request.config),
        )

    @staticmethod
    def _validate_name(field: str, value: str) -> None:
        if not _SAFE_NAME.fullmatch(value):
            raise ConfigurationError(f"{field} must match {_SAFE_NAME.pattern!r}: {value!r}")

    @staticmethod
    def _validate_module(module: str) -> None:
        if module not in _MODULES:
            raise ConfigurationError(f"unknown functional module: {module}")
