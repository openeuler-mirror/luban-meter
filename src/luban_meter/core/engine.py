"""Local-only Benchmark orchestration."""

from __future__ import annotations

from luban_meter.core.models import (
    BenchmarkResult,
    ResolvedRun,
    RunRequest,
)
from luban_meter.core.registry import BenchmarkRegistry
from luban_meter.execution.manager import ExecutionManager
from luban_meter.execution.session import ExecutionSession
from luban_meter.result.manager import ResultManager


class CoreEngine:
    def __init__(
        self,
        registry: BenchmarkRegistry,
        execution_manager: ExecutionManager | None = None,
        result_manager: ResultManager | None = None,
    ) -> None:
        self._registry = registry
        self._execution_manager = execution_manager or ExecutionManager()
        self._result_manager = result_manager or ResultManager()

    def run(self, request: RunRequest) -> BenchmarkResult:
        stage = "resolve"
        session: ExecutionSession | None = None
        resolved: ResolvedRun | None = None
        result: BenchmarkResult | None = None

        try:
            resolved = self._registry.resolve(request)

            stage = "open_session"
            session = self._execution_manager.open_session()

            stage = "execute"
            artifacts = session.execute(resolved)

            stage = "process_result"
            result = self._result_manager.process(resolved, artifacts)
        # The Engine is the Run boundary: any tool, execution, or processor
        # failure must be converted into a diagnostic result.json.
        except Exception as exc:  # noqa: BLE001
            result = self._result_manager.failure(
                request=request,
                stage=stage,
                error=exc,
                parameters=resolved.parameters if resolved is not None else {},
            )
        finally:
            if session is not None:
                try:
                    session.close()
                # Cleanup errors are recorded without replacing the Run error.
                except Exception as cleanup_error:  # noqa: BLE001
                    if result is not None:
                        result.metadata = {
                            **dict(result.metadata),
                            "cleanup_error": {
                                "type": type(cleanup_error).__name__,
                                "message": str(cleanup_error),
                            },
                        }

        assert result is not None
        self._result_manager.write(request, result)
        return result
