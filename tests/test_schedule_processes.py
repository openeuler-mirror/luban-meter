"""Real CLI, dispatcher, coordinator and benchmark subprocess acceptance."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from luban_meter.core.schedule_store import ScheduleStore
from luban_meter.core.schedule_time import display_time, stamp, timestamp
from tests import test_extension_worker

extension = test_extension_worker.extension


def cli(project, environment, *args, ok=True):
    result = subprocess.run(
        [sys.executable, "-m", "luban_meter.cli", "schedule", *args],
        cwd=project, env=environment, capture_output=True, text=True,
        timeout=15,
    )
    if ok:
        assert result.returncode == 0, result.stderr
    return result


def config(project, name, delay=0.02, requests=1):
    path = project / f"{name}.json"
    path.write_text(json.dumps({"tasks": [{
        "name": name, "scenario": "fixture", "delay": delay,
        "num_requests": requests, "max_concurrency": 1,
        "fixture_secret": "schedule-test-secret",
    }]}))
    return path


def register(project, environment, path, due):
    result = cli(project, environment, "--at", stamp(due), "--timezone", "UTC",
                 str(path))
    return result.stdout.split()[1]


def rows(project):
    with ScheduleStore(project).transaction() as records:
        return list(records.executions.values())


def wait_until(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError("scheduled subprocess condition timed out")


def serve(project, environment, *, level="info", log_dir=None):
    # Capture test stderr; product logs remain in the normal instance log.
    capture = project / "service-console.txt"
    previous = capture.read_text().count("Schedule service ready") if (
        capture.exists()
    ) else 0
    stream = (project / "service-console.txt").open("a")
    argv = [sys.executable, "-m", "luban_meter.cli", "schedule", "serve",
            "--log-level", level]
    if log_dir:
        argv.extend(["--log-dir", str(log_dir)])
    process = subprocess.Popen(
        argv,
        cwd=project, env=environment, stdout=subprocess.DEVNULL, stderr=stream,
    )
    stream.close()
    if level in {"debug", "info"}:
        wait_until(lambda: capture.read_text().count("Schedule service ready")
                   > previous)
    else:
        # At higher levels the service's INFO readiness log is suppressed.
        lock = ScheduleStore(project).lock("service", timeout=0)

        def ready():
            from filelock import Timeout

            try:
                with lock:
                    return False
            except Timeout:
                return True

        wait_until(ready)
    return process


def stop(process):
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=15)


def test_daily_storage_serial_queue_modify_cancel_and_list(
    extension, tmp_path,
):
    first_config = config(tmp_path, "first", delay=0.3, requests=10)
    second_config = config(tmp_path, "second")
    third_config = config(tmp_path, "third")
    process = serve(tmp_path, extension)
    try:
        due = time.time() + 3
        first = register(tmp_path, extension, first_config, due)
        # Running uses the snapshot, even if the source JSON disappears.
        first_config.unlink()
        second = register(tmp_path, extension, second_config, due)
        third = register(tmp_path, extension, third_config, due)
        wait_until(lambda: len(rows(tmp_path)) == 3
                   and any(r["status"] == "running" for r in rows(tmp_path)))
        listing = cli(tmp_path, extension, "list").stdout
        assert "running" in listing and "queued" in listing
        invalid = cli(tmp_path, extension, "update", first,
                      "--at", stamp(due + 100), ok=False)
        assert invalid.returncode == 2
        assert "cannot modify" in invalid.stderr
        updated = config(tmp_path, "updated", requests=2)
        cli(tmp_path, extension, "update", second, str(updated))
        cli(tmp_path, extension, "cancel", third)
        cli(tmp_path, extension, "cancel", third)
        wait_until(lambda: sum(r["status"] == "success"
                               for r in rows(tmp_path)) == 2)
        completed = {r["plan_id"]: r for r in rows(tmp_path)
                     if r["status"] == "success"}
        a, b = completed[first], completed[second]
        assert timestamp(b["started_at"]) >= timestamp(a["finished_at"])
        assert a["scheduled_at"] == b["scheduled_at"]
        assert timestamp(b["started_at"]) > due
        result = json.loads(Path(b["result_path"]).read_text())
        assert result["schedule"]["plan"]["tasks"][0]["name"] == "updated"
        parameters = result["tasks"][0]["cases"][0]["parameters"]
        assert parameters["num_requests"] == 2
        first_result = json.loads(Path(a["result_path"]).read_text())
        assert first_result["duration_s"] < (
            timestamp(b["finished_at"]) - due
        )
        log = Path(a["log_path"]).read_text()
        assert "Scheduled run started" in log and "Worker initializing" in log
        assert "Resolved Case" in log and "Worker finished" in log
        assert "schedule-test-secret" not in log
        assert "schedule-test-secret" not in Path(a["result_path"]).read_text()
        (only_log,) = Path(a["log_path"]).parent.iterdir()
        assert str(only_log) == a["log_path"]
        listing = cli(tmp_path, extension, "list").stdout
        assert first in listing and second in listing and third in listing
        assert "success" in listing and "cancelled" in listing
        assert display_time(a["started_at"], "UTC") in listing
        assert display_time(b["finished_at"], "UTC") in listing
        (day,) = (tmp_path / "schedule").glob("????-??-??")
        assert (day / "plans.json").is_file()
        assert (day / "executions.csv").is_file()
        assert '"num_requests": 10' in (day / "plans.json").read_text()
        if os.name == "posix":
            assert (day / "plans.json").stat().st_mode & 0o077 == 0
            assert (day / "executions.csv").stat().st_mode & 0o077 == 0
        # A second dispatcher fails rather than starting another runner.
        duplicate = cli(tmp_path, extension, "serve", ok=False)
        assert duplicate.returncode == 2
        assert "already running" in duplicate.stderr
    finally:
        stop(process)


def test_cancel_running_releases_queue_and_never_revives(extension, tmp_path):
    process = serve(tmp_path, extension)
    try:
        due = time.time() + 2
        first = register(tmp_path, extension,
                         config(tmp_path, "first", 0.4, 10), due)
        second = register(tmp_path, extension, config(tmp_path, "second"), due)
        wait_until(lambda: any(
            r["plan_id"] == first and r["status"] == "running"
            for r in rows(tmp_path)))
        cli(tmp_path, extension, "cancel", first)
        wait_until(lambda: any(
            r["plan_id"] == second and r["status"] == "success"
            for r in rows(tmp_path)))
        cancelled = next(r for r in rows(tmp_path) if r["plan_id"] == first)
        assert cancelled["status"] == "cancelled"
        assert cancelled["finished_at"]
        assert Path(cancelled["result_path"]).is_file()
        stop(process)
        old_ids = {r["execution_id"] for r in rows(tmp_path)}
        process = serve(tmp_path, extension)
        time.sleep(0.4)
        assert {r["execution_id"] for r in rows(tmp_path)} == old_ids
    finally:
        stop(process)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL restart scenario")
def test_restart_retests_from_start_and_preserves_queue(extension, tmp_path):
    process = serve(tmp_path, extension)
    try:
        due = time.time() + 2
        first = register(tmp_path, extension,
                         config(tmp_path, "first", 0.2, 10), due)
        second = register(tmp_path, extension, config(tmp_path, "second"), due)
        original = wait_until(lambda: next(
            (r for r in rows(tmp_path)
             if r["plan_id"] == first and r["status"] == "running"), None,
        ))
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        process = serve(tmp_path, extension)
        wait_until(lambda: sum(r["status"] == "success"
                               for r in rows(tmp_path)) == 2)
        history = rows(tmp_path)
        old = next(r for r in history
                   if r["execution_id"] == original["execution_id"])
        retry = next(r for r in history
                     if r["retry_of"] == original["execution_id"])
        following = next(r for r in history if r["plan_id"] == second)
        assert old["status"] == "interrupted"
        assert Path(old["result_path"]).is_file()
        assert retry["run_id"] != old["run_id"]
        assert retry["status"] == "success"
        assert timestamp(retry["started_at"]) >= timestamp(old["finished_at"])
        assert timestamp(following["started_at"]) >= timestamp(
            retry["finished_at"])
        result = json.loads(Path(retry["result_path"]).read_text())
        assert result["tasks"][0]["cases"][0]["metrics"][
            "fixture_score"]["score"]["value"] == 10
        assert original["execution_id"] in cli(
            tmp_path, extension, "list").stdout
    finally:
        stop(process)


def test_failure_keeps_traceback_and_continues_queue(extension, tmp_path):
    failing = config(tmp_path, "failing")
    definition = json.loads(failing.read_text())
    definition["tasks"][0]["failure"] = "request"
    failing.write_text(json.dumps(definition))
    log_dir = tmp_path / "chosen-logs"
    process = serve(tmp_path, extension, level="error", log_dir=log_dir)
    try:
        due = time.time() + 2
        first = register(tmp_path, extension, failing, due)
        second = register(tmp_path, extension, config(tmp_path, "following"),
                          due)
        wait_until(lambda: any(
            r["plan_id"] == second and r["status"] == "success"
            for r in rows(tmp_path)))
        failed = next(r for r in rows(tmp_path) if r["plan_id"] == first)
        assert failed["status"] == "failed"
        log = Path(failed["log_path"]).read_text()
        assert "Traceback" in log and "fixture request failed" in log
        assert "schedule-test-secret" not in log
        assert "Run started" not in log
        assert Path(failed["log_path"]).parent == log_dir
        assert failed["summary"] and failed["finished_at"]
    finally:
        stop(process)


def test_cron_cli_and_empty_list(extension, tmp_path):
    assert "暂无计划" in cli(tmp_path, extension, "list").stdout
    source = config(tmp_path, "periodic")
    created = cli(tmp_path, extension, "--cron", "30 2 * * *",
                  "--timezone", "Asia/Shanghai", str(source))
    plan_id = created.stdout.split()[1]
    with ScheduleStore(tmp_path).transaction() as records:
        plan = records.plans[plan_id]
        assert plan["cron"] == "30 2 * * *"
        assert plan["timezone"] == "Asia/Shanghai"
        assert timestamp(plan["next_at"]) > time.time()
    cli(tmp_path, extension, "update", plan_id, "--cron", "*/5 * * * *")
    listing = cli(tmp_path, extension, "list").stdout
    assert "*/5 * * * *" in listing and "pending" in listing
    assert "Asia/Shanghai" in listing
    invalid = cli(tmp_path, extension, "--at", "invalid", str(source),
                  ok=False)
    assert invalid.returncode == 2
    cli(tmp_path, extension, "cancel", plan_id)
    assert "cancelled" in cli(tmp_path, extension, "list").stdout
