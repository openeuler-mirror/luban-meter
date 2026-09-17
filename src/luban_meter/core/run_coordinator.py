"""Local-only Benchmark orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping

from luban_meter.core.run_contracts import (
    RawRunArtifacts,
    ResolvedRun,
    RunRequest,
    RunResult,
)
from luban_meter.core.scenario_registry import ScenarioRegistry
from luban_meter.execution.manager import ExecutionManager
from luban_meter.execution.session import ExecutionSession
from luban_meter.result.manager import ResultManager
from luban_meter.result.schema import validate_result
from luban_meter.utils.json_io import to_jsonable


class RunCoordinator:
    def __init__(
        self,
        registry: ScenarioRegistry,
        execution_manager: ExecutionManager | None = None,
        result_manager: ResultManager | None = None,
    ) -> None:
        self._registry = registry
        self._execution_manager = execution_manager or ExecutionManager()
        self._result_manager = result_manager or ResultManager()

    def run(self, request: RunRequest) -> RunResult:
        stage = "resolve"
        session: ExecutionSession | None = None
        resolved_run: ResolvedRun | None = None
        result: RunResult | None = None
        artifacts: RawRunArtifacts | None = None

        try:
            resolved_run = self._registry.resolve(request)

            stage = "open_session"
            session = self._execution_manager.open_session()

            stage = "execute"
            artifacts = session.execute(resolved_run)

            stage = "process_result"
            result = self._result_manager.process(resolved_run, artifacts)

            # Inject device monitoring summary into result.environment
            self._inject_device_monitoring(result, artifacts)

            stage = "validate_result"
            validate_result(to_jsonable(result))
        # The Engine is the Run boundary: any tool, execution, or
        # processor
        # failure must be converted into a diagnostic result.json.
        except Exception as exc:  # noqa: BLE001
            environment = getattr(result, "environment", {})
            result = self._result_manager.failure(
                request=request,
                stage=stage,
                error=exc,
                parameters=resolved_run.parameters
                if resolved_run is not None
                else {},
                artifacts=artifacts,
            )
            if isinstance(environment, Mapping):
                result.environment = dict(environment)
        finally:
            if session is not None:
                try:
                    session.close()
                # Cleanup errors are recorded without replacing the Run
                # error.
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

    @staticmethod
    def _inject_device_monitoring(
        result: RunResult,
        artifacts: RawRunArtifacts,
    ) -> None:
        """Read device monitoring data from raw_result.json and inject
        into result.environment.
        """
        try:
            with artifacts.raw_result.open(
                "r", encoding="utf-8"
            ) as raw_result_stream:
                raw_result = json.load(raw_result_stream)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(raw_result, dict):
            return

        result_environment: dict[str, object] = dict(result.environment)

        # Hardware environment goes first for easy access
        hardware_environment = raw_result.get("hardware_environment")
        if hardware_environment is not None:
            result_environment["hardware_environment"] = hardware_environment

        monitoring = raw_result.get("device_monitoring")
        if monitoring is not None:
            result_environment["device_monitoring"] = monitoring

        result.environment = result_environment
