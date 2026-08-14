"""Single command-line entry point for LuBan-Meter."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from luban_meter.core.engine import CoreEngine
from luban_meter.core.errors import BenchmarkToolkitError
from luban_meter.core.models import RunRequest
from luban_meter.core.registry import BenchmarkRegistry
from luban_meter.suite.loader import SuiteLoader
from luban_meter.suite.models import SuiteRequest
from luban_meter.suite.runner import SuiteRunner
from luban_meter.utils.json_io import to_jsonable
from luban_meter.utils.run_id import create_run_id


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
    run.add_argument("--vendor", required=True)
    run.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark directory name under vendors/<vendor>/benchmark/<module>",
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

    suite = commands.add_parser("suite", help="Run one vendor Suite")
    suite.add_argument("--vendor", required=True)
    suite.add_argument(
        "--suite",
        required=True,
        help="Logical Suite name under vendors/<vendor>/suites",
    )
    suite.add_argument("--model-path", type=Path, help="Model weights host path")
    suite.add_argument("--model-name", help="Logical or served model name")
    suite.add_argument("--output", type=Path, default=Path("runs"))
    suite.add_argument("--timeout", type=int, default=3600)
    suite.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed task",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "benchmarks" and args.benchmarks_command == "list":
        registry = BenchmarkRegistry()
        for module, description in registry.modules():
            benchmarks = ",".join(registry.list_benchmarks(module)) or "-"
            print(f"{module}\t{benchmarks}\t{description}")
        return 0

    if args.command == "run":
        request = RunRequest(
            run_id=create_run_id(args.module),
            module=args.module,
            vendor=args.vendor,
            benchmark=args.benchmark,
            config=args.config,
            model_path=args.model_path,
            model_name=args.model_name,
            output_dir=args.output,
            timeout=args.timeout,
        )

        try:
            result = CoreEngine(BenchmarkRegistry()).run(request)
        except BenchmarkToolkitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            json.dumps(
                to_jsonable(result),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.status == "success" else 1

    if args.command == "suite":
        request = SuiteRequest(
            suite_id=create_run_id(f"{args.vendor}-{args.suite}"),
            vendor=args.vendor,
            suite=args.suite,
            model_path=args.model_path,
            model_name=args.model_name,
            output_dir=args.output,
            timeout=args.timeout,
            fail_fast=args.fail_fast,
        )
        try:
            definition = SuiteLoader().load(args.vendor, args.suite)
            result = SuiteRunner(CoreEngine(BenchmarkRegistry())).run(
                request,
                definition,
            )
        except BenchmarkToolkitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            json.dumps(
                to_jsonable(result),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.status == "success" else 1

    return 2
