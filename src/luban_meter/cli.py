"""Flat command-line options generated from registered parameter schemas."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.metric import metrics
from luban_meter.core.planning import build_plan, cli_schema, public_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.reporting.data import load_result
from luban_meter.reporting.html import write_html
from luban_meter.reporting.progress import TerminalProgress
from luban_meter.reporting.terminal import render_text
from luban_meter.reporting.terminal_details import (
    DEFAULT_LIMIT,
    SORT_FIELDS,
    render_details,
)


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
        if command == "run":
            run.add_argument(
                "--no-summary", action="store_true",
                help="关闭运行结束后的终端结果摘要",
            )
            run.add_argument(
                "--no-progress", action="store_true",
                help="关闭运行中的终端进度显示（不影响结束后的摘要）",
            )
        run.add_argument("--observe", choices=("prometheus",))
        run.add_argument("--observe-url")
        run.add_argument("--observe-interval", type=float, default=1.0)
        for name, spec in schema.items():
            kwargs = {"default": argparse.SUPPRESS, "help": spec.help or name}
            if spec.type is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            elif spec.type is dict:
                kwargs["type"] = json.loads
            elif spec.type is list:
                kwargs.update(nargs="+", type=spec.item_type or str)
            else:
                kwargs["type"] = spec.type
            run.add_argument("--" + name.replace("_", "-"), **kwargs)
    report = commands.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output", type=Path)
    report.add_argument("--format", choices=("html", "text"), default="html")
    report.add_argument("--verbose", action="store_true",
                        help="文本报告显示统计口径、时间戳和诊断信息")
    report.add_argument("--task", action="append", default=[])
    report.add_argument("--case", action="append", default=[])
    report.add_argument("--details", choices=("errors", "requests"),
                        help="按需读取原始文件，显示错误或请求列表（文本）")
    report.add_argument("--sort", choices=tuple(SORT_FIELDS),
                        help="请求详情按已保存指标降序；缺失值放在末尾")
    report.add_argument("--limit", type=int,
                        help="请求详情最多显示的记录数，默认 10，必须为正数")
    report.add_argument("--include-body", action="store_true",
                        help="请求详情显式显示已保存正文预览，可能含敏感信息")
    report.add_argument(
        "--all", dest="all_cases", action="store_true",
        help="显示全部匹配 Case；文本报告默认最多显示 5 个",
    )
    report.add_argument(
        "--include-requests", action="store_true",
        help="内嵌原始请求及已保存的输入输出，生成完整详情 HTML",
    )
    report.add_argument(
        "--raw-root", type=Path,
        help="原始运行目录；默认使用输入结果文件所在目录",
    )
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
            if not args.details and (
                args.sort or args.limit is not None or args.include_body
            ):
                raise ValueError(
                    "--sort, --limit and --include-body require --details"
                )
            if args.details and args.format != "text":
                raise ValueError("--details requires --format text")
            if args.format == "text":
                if args.include_requests or (
                    args.raw_root and not args.details
                ):
                    raise ValueError(
                        "text reports do not support --include-requests; "
                        "--raw-root requires --details"
                    )
                result = load_result(args.input)
                if args.details:
                    text = render_details(
                        result, raw_root=args.raw_root or args.input.parent,
                        details=args.details, tasks=args.task, cases=args.case,
                        sort=args.sort,
                        limit=(DEFAULT_LIMIT if args.limit is None
                               else args.limit),
                        verbose=args.verbose, include_body=args.include_body,
                        result_path=args.input, output_path=args.output,
                    )
                else:
                    text = render_text(
                        result, tasks=args.task, cases=args.case,
                        all_cases=args.all_cases, verbose=args.verbose,
                        result_path=args.input,
                    )
                if args.output:
                    if args.output.resolve() == args.input.resolve():
                        raise ValueError(
                            "output must not overwrite input JSON"
                        )
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + "\n", encoding="utf-8")
                else:
                    print(text)
                return 0
            if args.task or args.case or args.all_cases or args.verbose:
                raise ValueError(
                    "--task, --case, --all and --verbose require --format text"
                )
            if args.raw_root and not args.include_requests:
                raise ValueError("--raw-root requires --include-requests")
            filename = "report-full.html" if args.include_requests else (
                "report.html"
            )
            path = args.output or args.input.parent / filename
            print(write_html(
                load_result(args.input), path,
                include_requests=args.include_requests,
                raw_root=args.raw_root or args.input.parent,
            ))
            if args.include_requests:
                print("完整报告含已保存的原始输入输出；未采集字段不补造。")
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
            progress=None if args.no_progress else TerminalProgress(plan),
        )
        report_path = write_html(result, directory / "report.html")
        print(f"{result['status']} | {directory / 'result.json'}")
        print(f"HTML: {report_path}")
        if not args.no_summary:
            try:
                print(render_text(
                    result, result_path=directory / "result.json",
                ))
            except Exception as exc:
                # Display is optional: facts, HTML and exit status come first.
                try:
                    print(f"终端摘要未能显示: {exc}", file=sys.stderr)
                except (OSError, UnicodeError):
                    pass
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
