"""Delete one indexed Task and update the Run's managed results."""

import json
import os
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from luban_meter.core.planning import TASK_IDENTIFIER
from luban_meter.core.task_index import task_index
from luban_meter.core.token_totals import (
    run_throughput,
    run_throughput_selected,
)
from luban_meter.log import get_logger, log_context
from luban_meter.reporting.data import load_result
from luban_meter.reporting.html import write_html
from luban_meter.runtime.task_worker import aggregate_status
from luban_meter.utils.json_io import write_json_atomic

logger = get_logger(__name__)


def _artifact_paths(root, task):
    run_id, name = task["run_id"], task["task_name"]
    if not (TASK_IDENTIFIER.fullmatch(run_id)
            and TASK_IDENTIFIER.fullmatch(name)):
        raise ValueError("invalid run or task name in task index")
    run_dir = root / run_id
    task_dir = run_dir / "tasks" / name
    if (Path(task["run_dir"]) != run_dir
            or Path(task["result_path"]) != task_dir / "result.json"):
        raise ValueError("task index paths do not match the result directory")
    plan = run_dir / "plan.json"
    result = run_dir / "result.json"
    reports = (run_dir / "report.html", run_dir / "report-full.html")
    for path in (run_dir, task_dir.parent, task_dir, plan, result, *reports):
        if path.is_symlink():
            raise ValueError(f"cannot delete through symbolic link: {path}")
    return task_dir, plan, result, reports


def _tail(path):
    if not path.is_file():
        return b""
    with path.open("rb") as stream:
        stream.seek(max(0, stream.seek(0, os.SEEK_END) - 65536))
        return stream.read()


def _legacy_run_finished(result, report, log_file):
    if result is None or result.get("finished_at") is None:
        return False
    if _tail(report).rstrip().endswith(b"</html>"):
        return True
    return any(
        b" | luban_meter.cli | _execute:" in line
        and (b" | CLI run completed:" in line
             or b" | CLI command failed:" in line)
        for line in _tail(log_file).splitlines()
    )


def _remaining_tasks(tasks, target):
    if not isinstance(tasks, list):
        raise ValueError("saved tasks must be a list")
    remaining = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("saved task must be an object")
        matches_id = task.get("id") == target["id"]
        if matches_id != (task.get("name") == target["task_name"]):
            raise ValueError("saved task identity does not match task index")
        if not matches_id:
            remaining.append(task)
    return remaining


def _remove_artifacts(paths, target, result):
    task_dir, plan_path, result_path, reports = paths
    existing_reports = [path for path in reports if path.exists()]
    if existing_reports and result is None:
        raise ValueError("run result is required to update existing reports")
    run_dir = result_path.parent
    with TemporaryDirectory(prefix=".delete-task-", dir=run_dir) as temp:
        staging = Path(temp)
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(plan, dict):
                raise ValueError("saved plan must be an object")
            plan["tasks"] = _remaining_tasks(plan.get("tasks"), target)
            plan["case_count"] = sum(len(t["cases"]) for t in plan["tasks"])
            write_json_atomic(staging / plan_path.name, plan)
        if result is not None:
            tasks = _remaining_tasks(result["tasks"], target)
            result["tasks"] = tasks
            result["status"] = aggregate_status(tasks)
            metrics = result.setdefault("metrics", {})
            if run_throughput_selected(tasks):
                metrics["token_throughput"] = run_throughput(tasks)
            else:
                metrics.pop("token_throughput", None)
            write_json_atomic(staging / result_path.name, result)
            for path in existing_reports:
                write_html(
                    result, staging / path.name,
                    include_requests=path.name == "report-full.html",
                    raw_root=run_dir,
                )
        if task_dir.exists():
            shutil.rmtree(task_dir)
        for path in staging.iterdir():
            path.replace(run_dir / path.name)


def _delete_run_task(root, task_id):
    with task_index(root) as database:
        task = database.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if task is None:
            raise ValueError(f"task id not found: {task_id}")
        if task["deleted_at"] is not None:
            logger.info("Task already deleted: id=%s", task_id)
            return
        run = database.execute(
            "SELECT finished_at FROM runs WHERE id = ?", (task["run_id"],)
        ).fetchone()
        if run is not None and run["finished_at"] is None:
            raise ValueError(f"task run has not finished: {task_id}")
        siblings = database.execute("""
            SELECT task_name FROM tasks
            WHERE run_id = ? AND id != ? AND deleted_at IS NULL
        """, (task["run_id"], task_id)).fetchall()

    paths = _artifact_paths(root, task)
    task_dir, plan_path, result_path, reports = paths
    if task_dir.exists():
        for sibling in siblings:
            directory = task_dir.parent / sibling["task_name"]
            if directory.exists() and task_dir.samefile(directory):
                raise ValueError("task directory is shared by another ID")
    result = load_result(result_path) if result_path.exists() else None
    if result is not None and (result["kind"] != "run"
                               or result.get("id") != task["run_id"]):
        raise ValueError("run result identity does not match task index")
    if any(path.exists() for path in (task_dir, plan_path, result_path,
                                      *reports)):
        if run is None and not _legacy_run_finished(
            result, reports[0], Path(task["log_path"])
        ):
            raise ValueError(f"cannot confirm run has finished: {task_id}")
        logger.info("Deleting task result: id=%s directory=%s",
                    task_id, task_dir)
        _remove_artifacts(paths, task, result)

    with task_index(root) as database:
        database.execute(
            "UPDATE tasks SET deleted_at = ? WHERE id = ?",
            (time.time(), task_id),
        )
    logger.info("Task deleted: id=%s", task_id)


def delete_task(result_dir, task_id):
    """Serialize deletions for the same Run without blocking index writers."""
    if not TASK_IDENTIFIER.fullmatch(task_id):
        raise ValueError(f"invalid task id: {task_id!r}")
    root = result_dir.resolve()
    with log_context(task=task_id, phase="delete"):
        with task_index(root) as database:
            task = database.execute(
                "SELECT run_id FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if task is None:
            raise ValueError(f"task id not found: {task_id}")
        run_id = task["run_id"]
        if not TASK_IDENTIFIER.fullmatch(run_id):
            raise ValueError("invalid run name in task index")
        lock_dir = root / ".task-delete-locks"
        lock_dir.mkdir(exist_ok=True)
        with FileLock(lock_dir / f"{run_id}.lock"):
            _delete_run_task(root, task_id)
