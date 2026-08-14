"""Process the raw output of exactly one Benchmark Run."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from luban_meter.core.errors import ResultProcessingError
from luban_meter.core.models import (
    BenchmarkResult,
    RawRunArtifacts,
    ResolvedRun,
    RunRequest,
)
from luban_meter.result.writer import ResultWriter


class ResultManager:
    def __init__(self, writer: ResultWriter | None = None) -> None:
        self._writer = writer or ResultWriter()

    def process(
        self,
        run: ResolvedRun,
        artifacts: RawRunArtifacts,
    ) -> BenchmarkResult:
        raw = self._read_raw_result(artifacts.raw_result)
        status = str(raw.get("status") or "")
        if status != "success":
            return self._result(
                run.request,
                status="failed",
                parameters=run.parameters,
                artifacts=artifacts,
                metadata=raw.get("metadata") or {},
                error=raw.get("error")
                or {
                    "type": "RawBenchmarkFailure",
                    "message": "raw Benchmark result reported failure",
                },
            )

        processor = self._load_processor(run.benchmark.result_handler)
        processed = processor(raw)
        if not isinstance(processed, Mapping):
            raise ResultProcessingError(
                f"result processor must return a mapping: "
                f"{run.benchmark.result_handler}"
            )

        metrics = processed.get("metrics", processed)
        environment = processed.get("environment", {})
        metadata = {
            **dict(raw.get("metadata") or {}),
            **dict(processed.get("metadata") or {}),
        }
        if not isinstance(metrics, Mapping):
            raise ResultProcessingError("processed metrics must be a mapping")
        if not isinstance(environment, Mapping):
            raise ResultProcessingError("processed environment must be a mapping")

        result_status = processed.get("status", "success")
        if not isinstance(result_status, str) or result_status not in {
            "success",
            "partial_failed",
            "failed",
        }:
            raise ResultProcessingError(
                "processed status must be success, partial_failed, or failed"
            )
        error = processed.get("error")
        if error is not None and not isinstance(error, Mapping):
            raise ResultProcessingError("processed error must be a mapping")

        return self._result(
            run.request,
            status=result_status,
            parameters=run.parameters,
            metrics=metrics,
            environment=environment,
            artifacts=artifacts,
            metadata=metadata,
            error=error,
        )

    def failure(
        self,
        request: RunRequest,
        stage: str,
        error: Exception,
        parameters: Mapping[str, Any],
    ) -> BenchmarkResult:
        return self._result(
            request,
            status="failed",
            parameters=parameters,
            metadata={"failure_stage": stage},
            error={
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def write(self, request: RunRequest, result: BenchmarkResult) -> None:
        self._writer.write(
            request.output_dir / request.run_id / "result.json",
            result,
        )

    @staticmethod
    def _read_raw_result(path: Path) -> Mapping[str, Any]:
        if not path.is_file():
            raise ResultProcessingError(f"raw result does not exist: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultProcessingError(f"invalid raw result {path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ResultProcessingError("raw result must contain a JSON object")
        if value.get("schema_version") != "luban-meter.raw/v1":
            raise ResultProcessingError(
                f"unsupported raw result schema: {value.get('schema_version')!r}"
            )
        return value

    @staticmethod
    def _load_processor(
        path: Path,
    ) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
        if not path.is_file():
            raise ResultProcessingError(f"result processor does not exist: {path}")

        module_name = f"_luban_meter_result_{abs(hash(path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ResultProcessingError(f"could not load result processor: {path}")
        module = importlib.util.module_from_spec(spec)
        ResultManager._execute_module(spec.loader, module, path)

        processor = getattr(module, "process", None)
        if not callable(processor):
            raise ResultProcessingError(
                f"result processor must define process(raw_result): {path}"
            )
        return processor

    @staticmethod
    def _execute_module(loader: Any, module: ModuleType, path: Path) -> None:
        try:
            loader.exec_module(module)
        except Exception as exc:
            raise ResultProcessingError(
                f"could not import result processor {path}: {exc}"
            ) from exc

    @staticmethod
    def _result(
        request: RunRequest,
        status: str,
        parameters: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        artifacts: RawRunArtifacts | None = None,
        metadata: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> BenchmarkResult:
        artifact_data: Mapping[str, str] = {}
        if artifacts is not None:
            artifact_data = {
                "raw_result": str(artifacts.raw_result),
                "stdout": str(artifacts.stdout_log),
                "stderr": str(artifacts.stderr_log),
                "directory": str(artifacts.artifact_dir),
            }

        return BenchmarkResult(
            schema_version="luban-meter.result/v1",
            run_id=request.run_id,
            status=status,
            module=request.module,
            vendor=request.vendor,
            benchmark=request.benchmark,
            config=str(request.config),
            model={
                "name": request.model_name,
                "path": str(request.model_path)
                if request.model_path is not None
                else None,
            },
            environment=dict(environment or {}),
            parameters=dict(parameters),
            metrics=dict(metrics or {}),
            artifacts=artifact_data,
            metadata=dict(metadata or {}),
            error=error,
        )
