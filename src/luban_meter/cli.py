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
from luban_meter.core.schedule_commands import add_parser, execute
from luban_meter.log import (
    LOG_LEVEL_NAMES,
    add_log_secrets,
    configure_logging,
    get_logger,
    instance_filename,
    log_command,
    log_context,
    log_parameters,
    redact_text,
    resolve_log_dir,
    task_secrets,
)
from luban_meter.reporting.data import load_result
from luban_meter.reporting.html import write_html
from luban_meter.reporting.progress import TerminalProgress
from luban_meter.reporting.terminal import render_text
from luban_meter.reporting.terminal_details import (
    DEFAULT_LIMIT,
    SORT_FIELDS,
    render_details,
)
from luban_meter.utils.json_io import write_json_atomic
from luban_meter.utils.run_id import create_run_id

logger = get_logger("luban_meter.cli")


class CommandParser(argparse.ArgumentParser):
    def error(self, message):
        logger.error("Argument parsing failed: %s", message)
        super().error(redact_text(message))


def _build_parser():
    schema = cli_schema()
    parser = CommandParser(prog="luban-meter", allow_abbrev=False)
    parser.add_argument("--log-level", choices=LOG_LEVEL_NAMES,
                        default="info", help="终端和实例文件的日志级别")
    parser.add_argument("--log-dir", type=Path,
                        help="覆盖本次调用的日志目录")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("benchmarks")
    listing.add_argument("action", choices=("list",))
    metric_list = commands.add_parser("metrics")
    metric_list.add_argument("action", choices=("list",))
    schedule = add_parser(commands)
    for command in ("run", "plan", "search-capacity"):
        run = commands.add_parser(command, allow_abbrev=False)
        if command == "search-capacity":
            run.add_argument("--scenario", default="online_serving")
            run.add_argument(
                "--verify", action=argparse.BooleanOptionalAction,
                default=False,
                help="搜索出稳定档后，按该速率追加恒定负载复验运行",
            )
        else:
            run.add_argument("--scenario", default=argparse.SUPPRESS)
            run.add_argument("--config", type=Path)
            run.add_argument("--task", action="append", default=[])
        run.add_argument("--result-dir", type=Path, default=Path("runs"))
        if command != "search-capacity":
            run.add_argument("--fail-fast", action="store_true")
        run.add_argument("--log-level", choices=LOG_LEVEL_NAMES,
                         default=argparse.SUPPRESS,
                         help="终端和文件日志级别")
        run.add_argument("--log-dir", type=Path,
                         default=argparse.SUPPRESS,
                         help="日志目录，默认 <运行目录>/log")
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
    for child in (listing, metric_list, report, schedule):
        child.add_argument("--log-level", choices=LOG_LEVEL_NAMES,
                           default=argparse.SUPPRESS)
        child.add_argument("--log-dir", type=Path, default=argparse.SUPPRESS)
    return parser, schema


def main(argv=None):
    parser, schema = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    # Locate the logging destination before argparse can terminate on errors.
    options = {}
    secret_flags = {"--" + k.replace("_", "-") for k, v in schema.items()
                    if v.secret}
    secrets = []
    positional = []
    tokens = iter(argv)
    for token in tokens:
        flag, equal, value = token.partition("=")
        if flag in secret_flags | {"--log-dir", "--result-dir", "--log-level",
                                   "--request-params"}:
            value = value if equal else next(tokens, "")
            options[flag] = value
            if flag in secret_flags or flag == "--request-params":
                secrets.append(value)
        else:
            positional.append(token)
    command = positional[0] if positional else None
    prefix = {"run": "run", "search-capacity": "search"}.get(command, "cli")
    run_id = create_run_id(prefix)
    fallback = (
        Path(options.get("--result-dir", "runs")) / run_id / "log"
        if command in ("run", "search-capacity") else None
    )
    level = options.get("--log-level", "info")
    with configure_logging(
        resolve_log_dir(options.get("--log-dir"), fallback=fallback),
        filename=instance_filename(run_id),
        level=level if level in LOG_LEVEL_NAMES else "info", secrets=secrets,
    ) as log_file:
        logger.info("Parsing command: luban-meter %s",
                    log_command(argv, hidden_flags=secret_flags))
        args = parser.parse_args(argv)
        parsed = log_parameters({
            key: "[redacted]" if key in schema and schema[key].secret
            else value for key, value in vars(args).items()
        }, secrets=secrets)
        add_log_secrets(secrets)
        logger.info("Parsed command and options: %s", parsed)
        return _execute(args, schema, run_id, log_file)


def _capacity_metric(result):
    """Find the first successful capacity_search metric in a run."""
    for task in result.get("tasks", []):
        for case in task.get("cases", []):
            metric = (case.get("metrics") or {}).get("capacity_search")
            if metric and metric.get("status") == "success":
                return task["name"], case, metric
    return None, None, None


def _render_conclusion(conclusion):
    lines = ["容量搜索（capacity search）结论"]
    status = conclusion.get("status", "unknown")
    lines.append(f"  状态: {status}")
    rate = conclusion.get("max_stable_request_rate") or {}
    if rate.get("value") is not None:
        lines.append(
            f"  最大稳定 Request Rate: {rate['value']} {rate['unit']}"
        )
    stage = conclusion.get("max_stable_stage")
    if stage and stage.get("value") is not None:
        lines.append(f"  最大稳定阶段: {stage['value']}")
    reason = (conclusion.get("search") or {}).get("reason") or (
        conclusion.get("reason")
    )
    if reason:
        lines.append(f"  原因: {reason}")
    verification = conclusion.get("verification")
    if verification:
        lines.append(
            f"  复验: {verification['status']} @ {verification['rate']} "
            "req/s"
        )
    stages = conclusion.get("stages") or []
    if stages:
        lines.append("  阶段判定:")
        lines.append(
            "    阶段  速率(req/s)  准入  成功  失败  超时  "
            "SLO达标率  判定"
        )
        for entry in stages:
            satisfied = (entry.get("slo") or {}).get("satisfied_rate") or {}
            slo_text = (
                str(satisfied["value"])
                if satisfied.get("value") is not None else "N/A"
            )
            lines.append(
                "    {:>4}  {:>10}  {:>4}  {:>4}  {:>4}  {:>4}  "
                "{:>8}  {}".format(
                    entry["stage_index"],
                    entry["offered_request_rate"]["value"],
                    entry["admitted_requests"]["value"],
                    entry["successful_requests"]["value"],
                    entry["failed_requests"]["value"],
                    entry["timeout_requests"]["value"],
                    slo_text,
                    entry["verdict"],
                )
            )
    judgment = conclusion.get("judgment")
    if judgment:
        lines.append(
            "  判据: SLO达标率>={} 成功率>={} 超时率<={} "
            "阶段最小样本>={}".format(
                judgment["stage_min_slo_rate"]["value"],
                judgment["stage_min_success_rate"]["value"],
                judgment["stage_max_timeout_rate"]["value"],
                judgment["stage_min_requests"]["value"],
            )
        )
    if conclusion.get("result"):
        lines.append(f"  结果: {conclusion['result']}")
    return "\n".join(lines)


def _run_capacity_plan(plan, args, run_id):
    add_log_secrets(task_secrets(plan["tasks"]))
    observation = (
        {"url": args.observe_url, "interval": args.observe_interval}
        if args.observe
        else None
    )
    return RunCoordinator().run(
        plan, args.result_dir, fail_fast=False, observation=observation,
        progress=TerminalProgress(plan), log_level=args.log_level,
        log_dir=args.log_dir, run_id=run_id,
    )


def _verify_capacity(base, plan, conclusion, args, run_id):
    rate = conclusion["search"]["max_stable_request_rate"]["value"]
    if rate is None:
        return conclusion
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    duration = 2.0 * float(parameters["step_stage_duration"])
    min_requests = int(parameters["stage_min_requests"])
    verify_base = dict(base)
    verify_base.update(
        profile="step",
        step_start_rate=rate,
        step_rate_increment=0.0,
        step_stage_duration=duration,
        step_max_rate=rate,
        num_requests=max(min_requests, math.ceil(rate * duration)),
        # The search run just warmed the service; sustained-load
        # confirmation needs no separate warmup window.
        num_warmups=0,
    )
    logger.info(
        "Capacity verification started: rate=%s req/s duration=%s s",
        rate, duration,
    )
    verify_plan = build_plan(verify_base, None)
    result, directory = _run_capacity_plan(
        verify_plan, args, create_run_id("verify"),
    )
    task_name, case, metric = _capacity_metric(result)
    if metric is None:
        verification = {
            "status": "failed",
            "rate": rate,
            "run_id": result["id"],
            "result": str(directory / "result.json"),
            "reason": (
                "capacity_search metric unavailable; inspect the "
                "verification run result and log"
            ),
        }
    else:
        stage = metric["values"]["stages"][0]
        verification = {
            "status": (
                "confirmed" if stage["verdict"] == "stable" else "unstable"
            ),
            "rate": rate,
            "duration_s": duration,
            "run_id": result["id"],
            "result": str(directory / "result.json"),
            "stage": stage,
        }
    logger.info(
        "Capacity verification finished: status=%s rate=%s",
        verification["status"], rate,
    )
    return {**conclusion, "verification": verification}


def _search_capacity(args, schema, run_id, log_file):
    values = vars(args)
    base = {
        key: values[key] for key in (*schema, "scenario") if key in values
    }
    if base.get("profile") is not None and base["profile"] != "step":
        raise ValueError("search-capacity uses the step profile")
    base["profile"] = "step"
    if (
        not math.isfinite(args.observe_interval)
        or args.observe_interval <= 0
    ):
        raise ValueError("observe-interval must be positive")
    if args.observe_url and not args.observe:
        raise ValueError("observe-url requires --observe prometheus")
    plan = build_plan(base, None, observe=bool(args.observe))
    logger.info("Capacity search started: cases=%s", plan["case_count"])
    result, directory = _run_capacity_plan(plan, args, run_id)
    task_name, case, metric = _capacity_metric(result)
    if metric is None:
        conclusion = {
            "kind": "capacity_search",
            "status": "failed",
            "run_id": result["id"],
            "reason": (
                "capacity_search metric unavailable; inspect the case "
                "status and log for collection or service failures"
            ),
            "result": str(directory / "result.json"),
        }
    else:
        search = metric["values"]["search"]
        conclusion = {
            "kind": "capacity_search",
            "status": search["status"],
            "run_id": result["id"],
            "task": task_name,
            "case": case["id"],
            "max_stable_request_rate": search["max_stable_request_rate"],
            "max_stable_stage": search.get("max_stable_stage"),
            "search": search,
            "judgment": metric["values"]["judgment"],
            "stages": metric["values"]["stages"],
            "result": str(directory / "result.json"),
        }
    if args.verify and conclusion["status"] == "found":
        conclusion = _verify_capacity(base, plan, conclusion, args, run_id)
    write_json_atomic(directory / "capacity-search.json", conclusion)
    logger.info("Capacity search conclusion saved: %s",
                directory / "capacity-search.json")
    print(_render_conclusion(conclusion))
    logger.info(
        "Search-capacity completed: status=%s result=%s log=%s",
        conclusion["status"], directory / "result.json", log_file,
    )
    verification_status = (conclusion.get("verification") or {}).get(
        "status"
    )
    if conclusion["status"] in ("found", "bounded") and (
        verification_status not in ("unstable", "failed")
    ):
        return 0
    return 1


def _execute(args, schema, run_id, log_file):
    try:
        if args.command == "schedule":
            return execute(args)
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
                    logger.info("Text report saved: %s", args.output)
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
            logger.info("HTML report started")
            report_path = write_html(
                load_result(args.input), path,
                include_requests=args.include_requests,
                raw_root=args.raw_root or args.input.parent,
            )
            logger.info("HTML report saved: %s", report_path)
            if args.include_requests:
                logger.info(
                    "完整报告含已保存的原始输入输出；未采集字段不补造。"
                )
            return 0
        if args.command == "search-capacity":
            return _search_capacity(args, schema, run_id, log_file)
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
        add_log_secrets(task_secrets(plan["tasks"]))
        if args.command == "plan":
            logger.info("Resolved plan: %s", log_parameters(public_plan(plan)),
                        extra={"file_only": True})
            print(json.dumps(public_plan(plan), ensure_ascii=False, indent=2))
            return 0
        logger.info("Run plan: tasks=%s cases=%s",
                    len(plan["tasks"]), plan["case_count"])
        for task in plan["tasks"]:
            with log_context(task=task["name"]):
                logger.info("Task plan: scenario=%s cases=%s",
                            task["scenario"], len(task["cases"]))
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
            log_level=args.log_level,
            log_dir=args.log_dir, run_id=run_id,
        )
        logger.info("HTML report started")
        report_path = write_html(result, directory / "report.html")
        logger.info("HTML report saved: %s", report_path)
        if not args.no_summary:
            try:
                print(render_text(
                    result, result_path=directory / "result.json",
                ))
            except Exception:
                # Display failures do not change the result or exit code.
                logger.warning("Terminal summary failed", exc_info=True)
        logger.info("CLI run completed: status=%s result=%s log=%s",
                    result["status"], directory / "result.json", log_file)
        return (
            0
            if result["status"] == "success"
            else (130 if result["status"] == "interrupted" else 1)
        )
    except (ValueError, TypeError, OSError) as exc:
        logger.exception("CLI command failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
