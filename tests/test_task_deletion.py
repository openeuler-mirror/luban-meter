"""Delete indexed artifacts through the CLI and preserve sibling Tasks."""

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from luban_meter.cli import main
from luban_meter.core import task_deletion
from luban_meter.core.task_index import registered_run
from luban_meter.core.token_totals import run_throughput
from tests import test_unified_runtime

mock_service = test_unified_runtime.service


def _report_tasks(path):
    content = path.read_text()
    match = re.search(r'id="report-data"[^>]*>(.*?)</script>', content, re.S)
    return json.loads(match[1])["result"]["tasks"]


@pytest.mark.parametrize("external_log", [False, True])
def test_delete_task_and_preserve_sibling(
    mock_service, tmp_path, capsys, monkeypatch, external_log,
):
    monkeypatch.delenv("LUBAN_LOG_DIR", raising=False)
    config = tmp_path / "task.json"
    parameters = test_unified_runtime.serving(mock_service)
    config.write_text(json.dumps({"tasks": [
        {"name": "first", "id": "kfa-first", **parameters,
         "cases": {"matrix": {"input_tokens": [8, 16]}}},
        {"name": "second", "id": "kfa-second", **parameters},
    ]}))
    root = tmp_path / "runs"
    run = [
        "run", "--config", str(config), "--result-dir", str(root),
        "--no-summary", "--no-progress",
    ]
    if external_log:
        run += ["--log-dir", str(tmp_path / "external-log")]
    assert main(run) == 0

    run_dir, = root.glob("run-*")
    first = run_dir / "tasks" / "first"
    second = run_dir / "tasks" / "second"
    assert (first / "raw" / "case_2.json").is_file()
    (first / "task-local.log").write_text("Task artifact to delete\n")
    artifact = (second / "result.json").read_bytes()
    command_logs = ["--log-dir", str(tmp_path / "command-logs")]
    assert main([
        "report", "--input", str(run_dir / "result.json"),
        "--include-requests", *command_logs,
    ]) == 0
    with closing(sqlite3.connect(root / "task-index.sqlite3")) as db:
        log_path, = db.execute(
            "SELECT log_path FROM tasks WHERE id = 'kfa-first'"
        ).fetchone()
        log = Path(log_path)
        instance_log = log.read_bytes()

        # A pre-upgrade index has neither a completion table nor tombstones.
        if external_log:
            with db:
                db.execute("DROP TABLE runs")
                db.execute("ALTER TABLE tasks DROP COLUMN deleted_at")

        delete = ["delete", "kfa-first", "--result-dir", str(root),
                  *command_logs]
        capsys.readouterr()
        with monkeypatch.context() as patch:
            def report_error(*args, **kwargs):
                raise OSError("cannot stage report")

            patch.setattr(task_deletion, "write_html", report_error)
            assert main(delete) == 2
            assert first.is_dir()
            assert "cannot stage report" in capsys.readouterr().err

        assert main(delete) == 0
        assert json.loads(capsys.readouterr().out) == {
            "id": "kfa-first", "status": "deleted",
        }
        assert not first.exists()
        assert (second / "result.json").read_bytes() == artifact
        plan = json.loads((run_dir / "plan.json").read_text())
        result = json.loads((run_dir / "result.json").read_text())
        assert plan["case_count"] == 1
        assert [task["id"] for task in plan["tasks"]] == ["kfa-second"]
        assert result["tasks"] == [json.loads(artifact)]
        assert result["metrics"]["token_throughput"] == run_throughput(
            result["tasks"]
        )
        for filename in ("report.html", "report-full.html"):
            tasks = _report_tasks(run_dir / filename)
            assert [task["id"] for task in tasks] == ["kfa-second"]
        deleted_at, = db.execute(
            "SELECT deleted_at FROM tasks WHERE id = 'kfa-first'"
        ).fetchone()
        assert deleted_at is not None
        assert main(delete) == 0
        assert db.execute(
            "SELECT deleted_at FROM tasks WHERE id = 'kfa-first'"
        ).fetchone() == (deleted_at,)

    assert main(["delete", "kfa-second", "--result-dir", str(root),
                 *command_logs]) == 0
    assert not second.exists()
    assert json.loads((run_dir / "plan.json").read_text())["case_count"] == 0
    assert json.loads((run_dir / "result.json").read_text())["tasks"] == []
    assert log.read_bytes() == instance_log
    assert main([*run, "--task", "first"]) == 2
    assert "duplicate task id: kfa-first" in capsys.readouterr().err


def test_delete_existing_index_with_missing_artifacts(tmp_path, capsys):
    root = tmp_path.resolve()
    run_dir = root / "run-legacy"
    with closing(sqlite3.connect(root / "task-index.sqlite3")) as db, db:
        db.execute("""
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL,
                task_name TEXT NOT NULL, run_dir TEXT NOT NULL,
                result_path TEXT NOT NULL, events_path TEXT NOT NULL,
                log_path TEXT NOT NULL
            )
        """)
        db.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)", (
            "legacy", run_dir.name, "task_1", str(run_dir),
            str(run_dir / "tasks/task_1/result.json"),
            str(run_dir / "tasks/task_1/raw/events.jsonl"),
            str(run_dir / "log/instance.log"),
        ))
    args = ["--result-dir", str(root), "--log-dir", str(root / "audit")]
    assert main(["delete", "unknown", *args]) == 2
    assert "task id not found" in capsys.readouterr().err
    assert main(["delete", "a/b", *args]) == 2
    assert "invalid task id" in capsys.readouterr().err
    assert main(["delete", "legacy", *args]) == 0
    assert not run_dir.exists()
    with closing(sqlite3.connect(root / "task-index.sqlite3")) as db:
        assert db.execute("SELECT deleted_at FROM tasks").fetchone()[0]


def test_delete_report_does_not_block_another_run_finishing(
    tmp_path, monkeypatch,
):
    root = tmp_path / "runs"
    first = {"name": "first", "id": "a", "cases": [], "status": "success",
             "category": "model_service_quality"}
    with registered_run([first], root, "run-a", tmp_path / "a.log"):
        run = root / "run-a"
        run.mkdir()
        (run / "result.json").write_text(json.dumps({
            "kind": "run", "id": "run-a", "tasks": [first],
            "status": "success", "finished_at": 1, "metrics": {},
        }))
        (run / "report.html").touch()

    report_started = Event()
    allow_report = Event()
    second_registered = Event()
    allow_second_finish = Event()
    write_html = task_deletion.write_html

    def slow_report(*args, **kwargs):
        report_started.set()
        assert allow_report.wait(10)
        return write_html(*args, **kwargs)

    def finish_other_run():
        second = {"name": "second", "id": "b"}
        with registered_run([second], root, "run-b", tmp_path / "b.log"):
            second_registered.set()
            assert allow_second_finish.wait(10)

    monkeypatch.setattr(task_deletion, "write_html", slow_report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        other_run = pool.submit(finish_other_run)
        try:
            assert second_registered.wait(5)
            deletion = pool.submit(task_deletion.delete_task, root, "a")
            assert report_started.wait(5)
            allow_second_finish.set()
            other_run.result(timeout=2)
            assert not deletion.done()
        finally:
            allow_second_finish.set()
            allow_report.set()
        deletion.result(timeout=5)

    with closing(sqlite3.connect(root / "task-index.sqlite3")) as db:
        assert db.execute(
            "SELECT finished_at FROM runs WHERE id = 'run-b'"
        ).fetchone()[0] is not None
