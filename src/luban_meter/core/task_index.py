"""Persist task IDs, artifact locations, and manual run completion."""

import sqlite3
import time
from contextlib import closing, contextmanager
from uuid import uuid4

from luban_meter.log import get_logger, log_context

logger = get_logger(__name__)


@contextmanager
def task_index(result_dir, *, create=False):
    """Serialize index mutations and migrate older task registrations."""
    index = result_dir.resolve() / "task-index.sqlite3"
    if index.is_symlink():
        raise ValueError(f"task index is a symbolic link: {index}")
    if create:
        index.parent.mkdir(parents=True, exist_ok=True)
    elif not index.is_file():
        raise ValueError(f"task index not found: {index}")
    mode = "rwc" if create else "rw"
    try:
        database = sqlite3.connect(
            f"{index.as_uri()}?mode={mode}", timeout=30, uri=True,
        )
        with closing(database), database:
            database.row_factory = sqlite3.Row
            database.execute("BEGIN IMMEDIATE")
            database.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY NOT NULL,
                    run_id TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    result_path TEXT NOT NULL,
                    events_path TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    deleted_at REAL
                )
            """)
            columns = {r["name"] for r in database.execute(
                "PRAGMA table_info(tasks)"
            )}
            if "deleted_at" not in columns:
                database.execute("ALTER TABLE tasks ADD deleted_at REAL")
            database.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY NOT NULL,
                    finished_at REAL
                )
            """)
            yield database
    except sqlite3.Error as exc:
        raise OSError(f"task index update failed: {index}: {exc}") from exc


@contextmanager
def registered_run(tasks, result_dir, run_id, log_file):
    """Reserve task IDs before execution and record when writes end."""
    run_dir = result_dir.resolve() / run_id
    task_ids = [task.get("id") or str(uuid4()) for task in tasks]
    with task_index(result_dir, create=True) as database:
        for task, task_id in zip(tasks, task_ids, strict=True):
            directory = run_dir / "tasks" / task["name"]
            inserted = database.execute("""
                INSERT INTO tasks (
                    id, run_id, task_name, run_dir, result_path,
                    events_path, log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
            """, (
                task_id, run_id, task["name"], str(run_dir),
                str(directory / "result.json"),
                str(directory / "raw" / "events.jsonl"), str(log_file),
            ))
            if inserted.rowcount == 0:
                raise ValueError(
                    f"task {task['name']}: duplicate task id: {task_id}"
                )
        database.execute("INSERT INTO runs (id) VALUES (?)", (run_id,))

    for task, task_id in zip(tasks, task_ids, strict=True):
        task["id"] = task_id
        with log_context(task=task["name"]):
            logger.info("Task ID registered: id=%s index=%s", task_id,
                        run_dir.parent / "task-index.sqlite3")

    try:
        yield
    finally:
        with task_index(result_dir) as database:
            database.execute(
                "UPDATE runs SET finished_at = ? WHERE id = ?",
                (time.time(), run_id),
            )
