"""Single command-line entry point for LuBan-Meter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from luban_meter.core.benchmark_registry import BenchmarkRegistry
from luban_meter.core.errors import BenchmarkToolkitError
from luban_meter.core.run_contracts import RunRequest
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.reporting.render import write_report
from luban_meter.reporting.result_reader import (
    build_report_from_result,
    load_report,
)
from luban_meter.suite.loader import SuiteLoader
from luban_meter.suite.runner import SuiteRunner
from luban_meter.suite.suite_contracts import SuiteRequest
from luban_meter.utils.json_io import to_jsonable
from luban_meter.utils.run_id import create_run_id


def _parse_task_config(value: str) -> tuple[str, Path]:
    task_name, separator, config_path = value.partition("=")
    if not separator or not task_name or not config_path:
        raise argparse.ArgumentTypeError(
            "task config must use TASK=PATH with non-empty values"
        )
    return task_name, Path(config_path)


def _add_monitor_args(parser: argparse.ArgumentParser) -> None:
    """Add hardware monitoring CLI arguments to a subparser."""
    monitor = parser.add_argument_group("hardware monitoring")
    monitor.add_argument(
        "--monitor-url",
        help="Prometheus exporter URL for hardware monitoring "
        "(e.g. http://43.138.110.236:9400). "
        "If omitted, hardware monitoring is disabled.",
    )
    monitor.add_argument(
        "--monitor-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )


def _apply_monitor_env(args) -> None:
    """Export monitor args as environment variables for
    ExecutionManager.
    """
    if getattr(args, "monitor_url", None):
        os.environ["LUBAN_MONITOR_URL"] = args.monitor_url
    if getattr(args, "monitor_interval", None):
        os.environ["LUBAN_MONITOR_INTERVAL"] = str(args.monitor_interval)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luban-meter")
    commands = parser.add_subparsers(dest="command", required=True)

    benchmarks = commands.add_parser("benchmarks", help="Inspect Benchmarks")
    benchmark_commands = benchmarks.add_subparsers(
        dest="benchmarks_command", required=True
    )
    benchmark_commands.add_parser("list", help="List available Benchmarks")

    run = commands.add_parser("run", help="Run one local Benchmark")
    run.add_argument("--module", required=True)
    run.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name shown by 'luban-meter benchmarks list'",
    )
    run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="User Benchmark parameter YAML",
    )
    run.add_argument("--model-path", type=Path, help="Model weights host path")
    run.add_argument("--model-name", help="Logical or served model name")
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--timeout", type=int, default=3600)
    run.add_argument("--name", help="Display name saved with the result")
    run.add_argument("--format", choices=("text", "json"), default="text")
    _add_monitor_args(run)

    suite = commands.add_parser("suite", help="Run one Benchmark Suite")
    suite.add_argument(
        "--suite",
        required=True,
        help="Logical Suite name under suite/definitions",
    )
    suite.add_argument(
        "--model-path", type=Path, help="Model weights host path"
    )
    suite.add_argument("--model-name", help="Logical or served model name")
    suite.add_argument("--output", type=Path, default=Path("runs"))
    suite.add_argument("--timeout", type=int, default=3600)
    suite.add_argument("--name", help="Display name saved with the result")
    suite.add_argument("--format", choices=("text", "json"), default="text")
    suite.add_argument(
        "--task-config",
        action="append",
        type=_parse_task_config,
        default=[],
        metavar="TASK=PATH",
        help="Override one Suite task's Benchmark YAML; may be repeated",
    )
    suite.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed task",
    )
    _add_monitor_args(suite)
    report = commands.add_parser("report", help="Report saved v2 results")
    report.add_argument("--input", type=Path, action="append", required=True)
    report.add_argument("--output", type=Path)
    report.add_argument("--name", help="Display name for one input")
    return parser


def _finish(result, path: Path, output_format: str) -> int:
    payload = to_jsonable(result)
    summary = f"{payload.get('status', '')} | {path}"
    try:
        report = build_report_from_result(payload, path.resolve())
        if report.kind == "suite":
            for task in report.tasks:
                if task.result_payload.get("schema_version") and task.source:
                    try:
                        child = build_report_from_result(
                            task.result_payload, task.source
                        )
                        write_report(child, task.source.parent / "report")
                    except Exception as exc:
                        print(
                            f"report error [{task.name}]: {exc}",
                            file=sys.stderr,
                        )
        destination, summary = write_report(report, path.parent / "report")
        print(f"Report: {destination}", file=sys.stderr)
    except Exception as exc:
        print(f"report error: {exc}", file=sys.stderr)
    if output_format == "json":
        print(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        )
    else:
        print(summary)
    return 0 if result.status == "success" else 1


def _report_command(args: argparse.Namespace) -> int:
    failed = False
    for source in args.input:
        try:
            report = load_report(source, args.name)
            output = args.output or report.source.parent / "report"
            if len(args.input) > 1:
                digest = hashlib.sha256(
                    str(report.source).encode()
                ).hexdigest()[:8]
                output = output / f"{report.kind}-{digest}"
            destination, summary = write_report(report, output)
            print(summary)
            print(f"Report: {destination}", file=sys.stderr)
        except Exception as exc:
            failed = True
            print(f"report error [{source}]: {exc}", file=sys.stderr)
    return int(failed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "report":
        if args.name and len(args.input) > 1:
            parser.error("--name applies to a single --input")
        return _report_command(args)

    if args.command == "benchmarks" and args.benchmarks_command == "list":
        registry = BenchmarkRegistry()
        for category_name, description in registry.list_categories():
            benchmark_names = (
                ",".join(registry.list_benchmarks(category_name)) or "-"
            )
            print(f"{category_name}\t{benchmark_names}\t{description}")
        return 0

    if args.command == "run":
        _apply_monitor_env(args)
        request = RunRequest(
            run_id=create_run_id(args.module),
            category_name=args.module,
            benchmark_name=args.benchmark,
            config_path=args.config,
            model_path=args.model_path,
            model_name=args.model_name,
            output_dir=args.output,
            timeout=args.timeout,
            display_name=args.name,
        )

        try:
            with redirect_stdout(sys.stderr):
                result = RunCoordinator(BenchmarkRegistry()).run(request)
        except BenchmarkToolkitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        return _finish(
            result, args.output / request.run_id / "result.json", args.format
        )

    if args.command == "suite":
        task_configs: dict[str, Path] = {}
        for task, config in args.task_config:
            if task in task_configs:
                parser.error(f"duplicate --task-config for task {task!r}")
            task_configs[task] = config
        _apply_monitor_env(args)
        request = SuiteRequest(
            suite_id=create_run_id(args.suite),
            suite=args.suite,
            model_path=args.model_path,
            model_name=args.model_name,
            output_dir=args.output,
            timeout=args.timeout,
            fail_fast=args.fail_fast,
            task_configs=task_configs,
            display_name=args.name,
        )
        try:
            definition = SuiteLoader().load(args.suite)
            with redirect_stdout(sys.stderr):
                result = SuiteRunner(RunCoordinator(BenchmarkRegistry())).run(
                    request,
                    definition,
                )
        except BenchmarkToolkitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        return _finish(
            result,
            args.output / request.suite_id / "suite_result.json",
            args.format,
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
