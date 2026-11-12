"""Daily JSON plan snapshots and CSV execution history, without a database."""

import copy
import csv
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from luban_meter.core.schedule_time import stamp
from luban_meter.log import get_logger, redact_text
from luban_meter.utils.json_io import write_json_atomic

logger = get_logger(__name__)

OPEN_STATES = {"queued", "starting", "running", "stopping"}
FIELDS = (
    "sequence", "recorded_at", "execution_id", "plan_id", "plan_revision",
    "scheduled_at", "status", "queued_at", "started_at", "finished_at",
    "run_id", "retry_of", "result_path", "log_path", "summary", "reason",
)


class ScheduleStore:
    def __init__(self, project):
        self.project = Path(project).resolve()
        self.root = self.project / "schedule"

    def lock(self, name, timeout=-1):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return FileLock(self.root / f".{name}.lock", timeout=timeout)

    @contextmanager
    def transaction(self):
        with self.lock("store"):
            yield Records(self.root)

    def request_stop(self, execution_id):
        (self.root / f".stop-{execution_id}").touch()

    def stop_requested(self, execution_id):
        return (self.root / f".stop-{execution_id}").exists()


class Records:
    def __init__(self, root):
        self.root = root
        self.plans = {}
        self.revisions = {}
        self.executions = {}
        self.sequence = 0
        for day in sorted(root.glob("????-??-??")):
            if not day.is_dir():
                continue
            source = day / "plans.json"
            if source.exists():
                for plan in json.loads(source.read_text(encoding="utf-8")):
                    revision = int(plan["revision"])
                    self.sequence = max(self.sequence, revision)
                    self.revisions[revision] = plan
                    previous = self.plans.get(plan["id"], {})
                    if revision > previous.get("revision", 0):
                        self.plans[plan["id"]] = plan
            source = day / "executions.csv"
            if source.exists():
                with source.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames != list(FIELDS):
                        raise ValueError(f"invalid execution CSV: {source}")
                    for row in reader:
                        if None in row or any(v is None for v in row.values()):
                            raise ValueError(f"incomplete CSV row: {source}")
                        sequence = int(row["sequence"])
                        self.sequence = max(self.sequence, sequence)
                        row["plan_revision"] = int(row["plan_revision"])
                        row["sequence"] = sequence
                        previous = self.executions.get(row["execution_id"], {})
                        if sequence > previous.get("sequence", 0):
                            self.executions[row["execution_id"]] = row

    def day(self, now):
        path = self.root / datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        path.mkdir(mode=0o700, exist_ok=True)
        if not (path / "plans.json").exists():
            write_json_atomic(path / "plans.json", [])
        if not (path / "executions.csv").exists():
            with (path / "executions.csv").open(
                "w", encoding="utf-8", newline="",
            ) as stream:
                os.chmod(stream.name, 0o600)
                csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
        return path

    def save_snapshot(self, plan, day):
        path = day / "plans.json"
        history = json.loads(path.read_text(encoding="utf-8"))
        if not any(p["revision"] == plan["revision"] for p in history):
            history.append(plan)
            write_json_atomic(path, history)

    def save_plan(self, plan, now):
        self.sequence += 1
        plan = copy.deepcopy(plan)
        plan["revision"] = self.sequence
        plan["recorded_at"] = stamp(now)
        self.save_snapshot(plan, self.day(now))
        self.plans[plan["id"]] = plan
        self.revisions[self.sequence] = plan
        return plan

    def save_execution(self, execution, now):
        self.sequence += 1
        row = {name: execution.get(name, "") for name in FIELDS}
        row.update(sequence=self.sequence, recorded_at=stamp(now))
        row["reason"] = redact_text(str(row["reason"]))
        day = self.day(now)
        self.save_snapshot(self.revisions[int(row["plan_revision"])], day)
        path = day / "executions.csv"
        with path.open("a", encoding="utf-8", newline="") as stream:
            os.chmod(path, 0o600)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        self.executions[row["execution_id"]] = row
        logger.info("Schedule execution: plan=%s execution=%s status=%s "
                    "scheduled=%s started=%s finished=%s reason=%s",
                    row["plan_id"], row["execution_id"], row["status"],
                    row["scheduled_at"], row["started_at"], row["finished_at"],
                    row["reason"])
        return row

    def new_execution(self, plan, due, status, now, **fields):
        return self.save_execution({
            "execution_id": uuid4().hex, "plan_id": plan["id"],
            "plan_revision": plan["revision"], "scheduled_at": due,
            "status": status, **fields,
        }, now)

    def queue(self):
        return sorted(
            (row for row in self.executions.values()
             if row["status"] == "queued"),
            key=lambda row: (not bool(row["retry_of"]), row["scheduled_at"],
                             self.plans[row["plan_id"]]["created_at"]),
        )

    def get_plan(self, plan_id):
        if plan_id not in self.plans:
            raise ValueError(f"unknown schedule plan: {plan_id}")
        return copy.deepcopy(self.plans[plan_id])
