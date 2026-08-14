"""Run every task in a Suite through the existing single-Run engine."""

from __future__ import annotations

from luban_meter.core.engine import CoreEngine
from luban_meter.core.errors import ConfigurationError
from luban_meter.core.models import RunRequest
from luban_meter.suite.models import (
    SuiteDefinition,
    SuiteRequest,
    SuiteResult,
    SuiteTaskResult,
)
from luban_meter.utils.json_io import write_json_atomic
from luban_meter.utils.run_id import create_run_id


class SuiteRunner:
    def __init__(self, engine: CoreEngine) -> None:
        self._engine = engine

    def run(
        self,
        request: SuiteRequest,
        definition: SuiteDefinition,
    ) -> SuiteResult:
        if request.vendor != definition.vendor:
            raise ConfigurationError(
                f"suite vendor mismatch: {request.vendor} != {definition.vendor}"
            )

        suite_dir = request.output_dir / request.suite_id
        tasks_dir = suite_dir / "tasks"
        write_json_atomic(
            suite_dir / "suite_request.json",
            {"request": request, "definition": definition},
        )

        task_results: list[SuiteTaskResult] = []
        stopped = False
        for task in definition.tasks:
            if stopped:
                task_results.append(
                    SuiteTaskResult(
                        name=task.name,
                        module=task.module,
                        benchmark=task.benchmark,
                        status="skipped",
                    )
                )
                continue

            run_id = create_run_id(task.name)
            run_request = RunRequest(
                run_id=run_id,
                module=task.module,
                vendor=request.vendor,
                benchmark=task.benchmark,
                config=task.config,
                model_path=request.model_path,
                model_name=request.model_name,
                output_dir=tasks_dir,
                timeout=task.timeout or request.timeout,
            )
            result = self._engine.run(run_request)
            result_path = tasks_dir / run_id / "result.json"
            task_results.append(
                SuiteTaskResult(
                    name=task.name,
                    module=task.module,
                    benchmark=task.benchmark,
                    status=result.status,
                    run_id=run_id,
                    result=str(result_path),
                )
            )
            if result.status != "success" and request.fail_fast:
                stopped = True

        status = self._suite_status(task_results)
        suite_result = SuiteResult(
            schema_version="luban-meter.suite-result/v1",
            suite_id=request.suite_id,
            name=definition.name,
            vendor=request.vendor,
            status=status,
            tasks=tuple(task_results),
        )
        write_json_atomic(suite_dir / "suite_result.json", suite_result)
        return suite_result

    @staticmethod
    def _suite_status(tasks: list[SuiteTaskResult]) -> str:
        succeeded = sum(task.status == "success" for task in tasks)
        if succeeded == len(tasks):
            return "success"
        if succeeded == 0:
            return "failed"
        return "partial_failed"
