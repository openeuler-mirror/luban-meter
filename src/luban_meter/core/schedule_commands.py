"""Small CLI surface for file-backed benchmark schedules."""

import time
from pathlib import Path
from uuid import uuid4

from luban_meter.core.planning import build_plan, public_plan
from luban_meter.core.schedule_store import OPEN_STATES, ScheduleStore
from luban_meter.core.schedule_time import (
    display_time,
    next_cron,
    stamp,
    timezone_for,
    timing,
)
from luban_meter.core.scheduled_runs import ScheduleService
from luban_meter.log import (
    add_log_secrets,
    get_logger,
    log_parameters,
    task_secrets,
)

logger = get_logger(__name__)


def add_parser(commands):
    parser = commands.add_parser("schedule", allow_abbrev=False)
    parser.add_argument("target", help="任务JSON，或 list/update/cancel/serve")
    parser.add_argument("value", nargs="?", help="修改或取消的计划ID")
    parser.add_argument("config", nargs="?", type=Path,
                        help="update可选的新任务JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--at", help="一次性日期时间，如 2026-11-11 02:30")
    group.add_argument("--cron", help="五字段cron，如 30 2 * * *")
    parser.add_argument("--timezone", help="IANA时区，默认保存当前系统时区")
    return parser


def snapshot(path):
    resolved = build_plan({}, path)
    add_log_secrets(task_secrets(resolved["tasks"]))
    logger.info("Schedule parameters: %s",
                log_parameters(public_plan(resolved)),
                extra={"file_only": True})
    return resolved


def register(store, config, at=None, cron=None, zone=None, now=None):
    now = time.time() if now is None else now
    calendar = timing(at, cron, zone, now)
    config = Path(config).resolve()
    resolved = snapshot(config)
    with store.transaction() as records:
        plan = records.save_plan({
            "id": uuid4().hex[:12], "calendar_id": uuid4().hex,
            "created_at": stamp(now), "source": str(config),
            "enabled": True, "resolved": resolved, **calendar,
        }, now)
    logger.info("Schedule created: plan=%s next=%s timezone=%s directory=%s",
                plan["id"], plan["next_at"], plan["timezone"], store.root)
    return plan


def update(
    store, plan_id, config=None, at=None, cron=None, zone=None, now=None,
):
    now = time.time() if now is None else now
    if config is None and at is None and cron is None and zone is None:
        raise ValueError("update requires a task JSON or time options")
    resolved = snapshot(Path(config).resolve()) if config else None
    with store.transaction() as records:
        plan = records.get_plan(plan_id)
        rows = [r for r in records.executions.values()
                if r["plan_id"] == plan_id]
        if any(r["status"] in OPEN_STATES - {"queued"} for r in rows):
            raise ValueError("cannot modify a starting or running plan")
        queued = [r for r in rows if r["status"] == "queued"]
        if not plan["enabled"] or (not plan["cron"]
                                   and not plan["next_at"] and not queued):
            raise ValueError("cannot modify a cancelled or completed plan")
        changed_time = at is not None or cron is not None
        if changed_time:
            plan.update(timing(at, cron, zone or plan["timezone"], now))
        elif zone is not None:
            timezone_for(zone)
            plan["timezone"] = zone
            if plan["cron"]:
                changed_time = True
                plan["next_at"] = next_cron(plan["cron"], zone, now)
        if resolved is not None:
            plan.update(source=str(Path(config).resolve()), resolved=resolved)
        if changed_time:
            plan["calendar_id"] = uuid4().hex
        plan = records.save_plan(plan, now)
        for row in queued:
            if changed_time:
                row = dict(row, status="cancelled", finished_at=stamp(now),
                           reason="plan modified before start")
            else:
                row = dict(row, plan_revision=plan["revision"])
            records.save_execution(row, now)
    logger.info("Schedule updated: plan=%s next=%s", plan_id, plan["next_at"])
    return plan


def cancel(store, plan_id, now=None):
    now = time.time() if now is None else now
    with store.transaction() as records:
        plan = records.get_plan(plan_id)
        if not plan["enabled"]:
            logger.info("Schedule already cancelled: plan=%s", plan_id)
            return
        plan.update(enabled=False, next_at="")
        records.save_plan(plan, now)
        for row in list(records.executions.values()):
            if row["plan_id"] != plan_id or row["status"] not in OPEN_STATES:
                continue
            queued = row["status"] == "queued"
            records.save_execution(
                dict(row, status="cancelled" if queued else "stopping",
                     finished_at=stamp(now) if queued else "",
                     reason="plan cancelled"), now,
            )
            if not queued:
                store.request_stop(row["execution_id"])
    logger.info("Schedule cancelled: plan=%s", plan_id)


def render_list(records):
    lines = ["\t".join((
        "PLAN", "PLAN_STATE", "EXECUTION", "STATUS", "TIME/CRON", "TIMEZONE",
        "SCHEDULED", "STARTED", "FINISHED", "QUEUE", "CONFIG", "RESULT",
        "LOG", "SUMMARY",
    ))]
    queue = {row["execution_id"]: str(i + 1)
             for i, row in enumerate(records.queue())}
    for plan in sorted(records.plans.values(), key=lambda p: p["created_at"]):
        rows = sorted((r for r in records.executions.values()
                       if r["plan_id"] == plan["id"]),
                      key=lambda r: r["sequence"])
        state = ("cancelled" if not plan["enabled"] else
                 "active" if plan["cron"] or plan["next_at"]
                 or any(r["status"] in OPEN_STATES for r in rows)
                 else "completed")
        if plan["enabled"] and plan["next_at"]:
            rows.append({"status": "pending", "scheduled_at": plan["next_at"]})
        if not rows:
            rows.append({"status": state, "scheduled_at": plan["at"]})
        for row in rows:
            used = (records.revisions[row["plan_revision"]]
                    if "plan_revision" in row else plan)
            columns = [
                plan["id"], state, row.get("execution_id", "-"), row["status"],
                used["cron"] or display_time(used["at"], used["timezone"]),
                used["timezone"],
                display_time(row.get("scheduled_at"), used["timezone"]),
                display_time(row.get("started_at"), used["timezone"]),
                display_time(row.get("finished_at"), used["timezone"]),
                queue.get(row.get("execution_id")), used["source"],
                row.get("result_path"), row.get("log_path"),
                row.get("summary") or row.get("reason"),
            ]
            lines.append("\t".join(str(v or "-") for v in columns))
    return "\n".join(lines) if records.plans else "暂无计划"


def execute(args):
    store = ScheduleStore(Path.cwd())
    target = args.target
    calendar = (args.at, args.cron, args.timezone)
    if target in {"list", "serve", "cancel"} and any(calendar):
        raise ValueError(f"{target} does not accept time options")
    if target in {"list", "serve"}:
        if args.value or args.config:
            raise ValueError(f"{target} does not accept extra arguments")
        if target == "serve":
            ScheduleService(store, log_level=args.log_level,
                            log_dir=args.log_dir).serve()
        else:
            with store.transaction() as records:
                print(render_list(records))
            logger.info("Schedule list read: directory=%s", store.root)
        return 0
    if target in {"update", "cancel"}:
        if not args.value:
            raise ValueError(f"{target} requires a plan ID")
        if target == "cancel":
            if args.config:
                raise ValueError("cancel does not accept a task JSON")
            cancel(store, args.value)
            print(f"已请求取消计划 {args.value}")
            return 0
        plan = update(store, args.value, args.config, *calendar)
    else:
        if args.value or args.config:
            raise ValueError("schedule accepts one task JSON")
        plan = register(store, target, *calendar)
    due = display_time(plan["next_at"], plan["timezone"])
    print(f"计划 {plan['id']} 已保存；下次时间 {due} 时区 {plan['timezone']}")
    return 0
