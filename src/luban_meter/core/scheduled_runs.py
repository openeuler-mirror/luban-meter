"""One resident dispatcher, a durable FIFO, and isolated scheduled runs."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from filelock import Timeout

from luban_meter.core.schedule_store import OPEN_STATES
from luban_meter.core.schedule_time import next_cron, stamp, timestamp
from luban_meter.log import get_logger, log_command
from luban_meter.utils.run_id import create_run_id

logger = get_logger(__name__)


def matches_calendar(records, row, plan):
    snapshot = records.revisions[row["plan_revision"]]
    return bool(row["retry_of"]) or (
        snapshot["calendar_id"] == plan["calendar_id"]
    )


def finish_execution(records, row, project, now, returncode=None):
    row = dict(row)
    plan = records.plans[row["plan_id"]]
    target = Path(project) / "runs" / row["run_id"] / "result.json"
    result = None
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
    complete = result is not None and "finished_at" in result
    row["status"] = (
        "cancelled" if not plan["enabled"] else
        result["status"] if complete else "interrupted"
    )
    if returncode and row["status"] != "cancelled":
        if returncode == 130:
            row["status"] = "interrupted"
        elif not complete or result["status"] == "success":
            row["status"] = "failed"
        row["reason"] = f"scheduled runner exited with code {returncode}"
    row["finished_at"] = stamp(result["finished_at"] if complete else now)
    if result is not None:
        row["result_path"] = str(target)
        tasks = result.get("tasks", [])
        cases = [case for task in tasks for case in task.get("cases", [])]
        states = {state: sum(c.get("status") == state for c in cases)
                  for state in sorted({c["status"] for c in cases})}
        row["summary"] = (
            f"tasks={len(tasks)} cases={len(cases)} "
            f"case_states={json.dumps(states)} "
            f"duration_s={result.get('duration_s', '')}"
        )
    return records.save_execution(row, now)


class ScheduleService:
    def __init__(self, store, *, log_level="info", log_dir=None):
        self.store = store
        self.log_level = log_level
        self.log_dir = (
            str(Path(log_dir).expanduser().resolve()) if log_dir else ""
        )
        self.stopping = False
        self.process = None
        self.execution_id = None

    def stop(self, signum=None, frame=None):
        self.stopping = True

    def recover(self, now=None):
        """Fence old runners before restoring interrupted attempts."""
        live_clock = now is None
        now = time.time() if live_clock else now
        cutoff = now
        with self.store.transaction() as records:
            old = [dict(row) for row in records.executions.values()
                   if row["status"] in {"starting", "running", "stopping"}
                   or (row["status"] == "interrupted"
                       and not row["finished_at"])]
            for row in old:
                self.store.request_stop(row["execution_id"])
                # A late child must not start this retired attempt.
                row["status"] = "interrupted"
                records.save_execution(row, now)
        logger.info("Schedule recovery: waiting for previous runner to exit")
        with self.store.lock("active"):
            if live_clock:
                now = time.time()
            with self.store.transaction() as records:
                for row in old:
                    finish_execution(records, row, self.store.project, now)
                retries = {row["retry_of"]
                           for row in records.executions.values()}
                for row in list(records.executions.values()):
                    plan = records.plans[row["plan_id"]]
                    if (row["status"] == "interrupted" and plan["enabled"]
                            and row["execution_id"] not in retries):
                        snapshot = records.revisions[row["plan_revision"]]
                        records.new_execution(
                            snapshot, row["scheduled_at"], "queued", now,
                            queued_at=stamp(now),
                            retry_of=row["execution_id"],
                        )
                self.admit(records, now, missed_before=cutoff)
        logger.info("Schedule recovery completed")

    def admit(self, records, now, missed_before=None):
        plans = sorted(records.plans.values(),
                       key=lambda p: (p["next_at"] or "~", p["created_at"]))
        for plan in plans:
            due = plan["next_at"]
            if not plan["enabled"] or not due or timestamp(due) > now:
                continue
            existing = [r for r in records.executions.values()
                        if r["plan_id"] == plan["id"]]
            if not any(row["scheduled_at"] == due for row in existing):
                busy = any(
                    r["status"] in OPEN_STATES
                    and (r["status"] != "queued"
                         or matches_calendar(records, r, plan))
                    for r in existing
                )
                missed = (missed_before is not None
                          and timestamp(due) < missed_before)
                status = ("missed" if missed else
                          "skipped" if busy else "queued")
                reason = ("service was not running" if missed else
                          "previous occurrence unfinished" if busy else "")
                records.new_execution(
                    plan, due, status, now, reason=reason,
                    queued_at=stamp(now) if status == "queued" else "",
                    finished_at=stamp(now) if status != "queued" else "",
                )
            plan = dict(plan)
            plan["next_at"] = (
                next_cron(plan["cron"], plan["timezone"], now)
                if plan["cron"] else ""
            )
            records.save_plan(plan, now)

    def tick(self, now=None):
        now = time.time() if now is None else now
        with self.store.transaction() as records:
            if self.process is not None:
                row = records.executions[self.execution_id]
                if not records.plans[row["plan_id"]]["enabled"]:
                    self.store.request_stop(self.execution_id)
                code = self.process.poll()
                if code is not None:
                    finish_execution(records, row, self.store.project,
                                     now, code)
                    self.process = None
                    self.execution_id = None
            self.admit(records, now)
            if self.process is not None or self.stopping:
                return
            for row in records.queue():
                plan = records.plans[row["plan_id"]]
                if (not plan["enabled"]
                        or not matches_calendar(records, row, plan)):
                    row = dict(row, status="cancelled", finished_at=stamp(now))
                    records.save_execution(row, now)
                    continue
                row = dict(
                    row, status="starting", run_id=create_run_id("run"),
                    plan_revision=(row["plan_revision"] if row["retry_of"]
                                   else plan["revision"]),
                )
                records.save_execution(row, now)
                try:
                    argv = [sys.executable, "-m",
                            "luban_meter.runtime.schedule_worker",
                            str(self.store.project), row["execution_id"],
                            self.log_level, self.log_dir]
                    logger.info("Starting scheduled command: %s",
                                log_command(argv))
                    environment = dict(os.environ)
                    package = str(Path(__file__).resolve().parents[2])
                    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
                        package, environment.get("PYTHONPATH"),
                    )))
                    self.process = subprocess.Popen(
                        argv, cwd=self.store.project, env=environment,
                        start_new_session=os.name == "posix",
                    )
                    self.execution_id = row["execution_id"]
                except OSError:
                    logger.exception("Scheduled runner launch failed")
                    records.save_execution(
                        dict(row, status="failed", finished_at=stamp(now),
                             reason="scheduled runner launch failed"), now,
                    )
                break

    def serve(self):
        try:
            with self.store.lock("service", timeout=0):
                handlers = {sig: signal.signal(sig, self.stop)
                            for sig in (signal.SIGINT, signal.SIGTERM)}
                try:
                    self.recover()
                    logger.info("Schedule service ready: %s", self.store.root)
                    while not self.stopping:
                        self.tick()
                        time.sleep(0.2)
                finally:
                    if self.process is not None:
                        self.store.request_stop(self.execution_id)
                        self.process.wait()
                        with self.store.transaction() as records:
                            finish_execution(
                                records, records.executions[self.execution_id],
                                self.store.project, time.time(),
                                self.process.returncode,
                            )
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
                    logger.info("Schedule service stopped; queue retained")
        except Timeout as exc:
            logger.exception("Schedule service already running")
            raise ValueError("schedule service is already running") from exc
