"""Flat command-line options generated from registered parameter schemas."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.execution_summary import case_summary
from luban_meter.core.metric import metrics
from luban_meter.core.planning import build_plan, cli_schema, public_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.reporting.html import load_result, write_html


def _build_parser():
    schema = cli_schema()
    parser = argparse.ArgumentParser(prog="luban-meter", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("benchmarks")
    listing.add_argument("action", choices=("list",))
    metric_list = commands.add_parser("metrics")
    metric_list.add_argument("action", choices=("list",))
    for command in ("run", "plan"):
        run = commands.add_parser(command, allow_abbrev=False)
        run.add_argument("--scenario", default=argparse.SUPPRESS)
        run.add_argument("--config", type=Path)
        run.add_argument("--task", action="append", default=[])
        run.add_argument("--result-dir", type=Path, default=Path("runs"))
        run.add_argument("--fail-fast", action="store_true")
        run.add_argument("--observe", choices=("prometheus",))
        run.add_argument("--observe-url")
        run.add_argument("--observe-interval", type=float, default=1.0)
        for name, spec in schema.items():
            kwargs = {"default": argparse.SUPPRESS, "help": spec.help or name}
            if spec.type is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            elif spec.type is list:
                kwargs.update(nargs="+", type=spec.item_type or str)
            else:
                kwargs["type"] = spec.type
            run.add_argument("--" + name.replace("_", "-"), **kwargs)
    report = commands.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path)
    return parser, schema


def main(argv=None):
    parser, schema = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmarks":
            for name, implementation in benchmarks.items():
                print(f"{implementation.category}\t{name}")
            return 0
        if args.command == "metrics":
            for name, implementation in metrics.items():
                print(f"{name}\t{','.join(sorted(implementation.requires))}")
            return 0
        if args.command == "report":
            path = args.output or args.input.parent / "report.html"
            print(write_html(load_result(args.input), path))
            return 0
        values = vars(args)
        base = {
            key: values[key] for key in (*schema, "scenario") if key in values
        }
        if (
            not math.isfinite(args.observe_interval)
            or args.observe_interval <= 0
        ):
            raise ValueError("observe-interval must be positive")
        if args.observe_url and not args.observe:
            raise ValueError("observe-url requires --observe prometheus")
        plan = build_plan(
            base, args.config, args.task, observe=bool(args.observe)
        )
        if args.command == "plan":
            print(json.dumps(public_plan(plan), ensure_ascii=False, indent=2))
            return 0
        print(f"计划：{len(plan['tasks'])}个任务，{plan['case_count']}个Case")
        for task in plan["tasks"]:
            print(
                f"  {task['name']}: {task['scenario']} "
                f"({len(task['cases'])} Cases)"
            )
            for case in task["cases"]:
                summary = case_summary(
                    benchmarks.get(task["scenario"]), case["parameters"]
                )
                print(f"    {case['id']}: " + json.dumps(
                    summary, ensure_ascii=False
                ), flush=True)
        observation = (
            {"url": args.observe_url, "interval": args.observe_interval}
            if args.observe
            else None
        )
        result, directory = RunCoordinator().run(
            plan,
            args.result_dir,
            fail_fast=args.fail_fast,
            observation=observation,
        )
        report_path = write_html(result, directory / "report.html")
        print(f"{result['status']} | {directory / 'result.json'}")
        print(f"HTML: {report_path}")
        return (
            0
            if result["status"] == "success"
            else (130 if result["status"] == "interrupted" else 1)
        )
    except (ValueError, TypeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
