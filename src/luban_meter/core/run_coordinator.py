"""Parent-owned execution, interruption, evidence and resource cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import public_plan
from luban_meter.runtime.case_recovery import recover_cases
from luban_meter.runtime.resource_manager import resources
from luban_meter.runtime.task_worker import aggregate_status
from luban_meter.utils.json_io import write_json_atomic
from luban_meter.utils.run_id import create_run_id


def kill_worker(process):
    if process is None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


class RunCoordinator:
    def __init__(self):
        self.process = None
        self.interruptions = 0
        self.stop_file = None
        self.interrupted_at = None
        self.progress = None

    def _notify(self, kind, **facts):
        if self.progress is not None:
            try:
                self.progress(kind, **facts)
            except Exception:
                # Display failures never change execution or result status.
                self.progress = None

    def interrupt(self, signum, frame):
        self.interruptions += 1
        if self.interrupted_at is None:
            self.interrupted_at = time.monotonic()
        self.stop_file.touch()
        if self.interruptions >= 2:
            kill_worker(self.process)

    def run(
        self, plan, output, *, fail_fast=False, observation=None, progress=None
    ):
        self.interruptions = 0
        self.interrupted_at = None
        self.progress = progress
        run_id = create_run_id("run")
        directory = Path(output).resolve() / run_id
        directory.mkdir(parents=True, exist_ok=False)
        self.stop_file = directory / ".stop"
        write_json_atomic(directory / "plan.json", public_plan(plan))
        result = {
            "kind": "run",
            "id": run_id,
            "status": "running",
            "started_at": time.time(),
            "tasks": [],
        }
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, self.interrupt)
        failed = False
        self._notify("run_started")
        try:
            for task in plan["tasks"]:
                if self.interruptions or (failed and fail_fast):
                    task_result = {
                        "kind": "task",
                        "name": task["name"],
                        "scenario": task["scenario"],
                        "category": task["category"],
                        "status": "skipped",
                        "cases": [],
                    }
                else:
                    task_result = self._run_task(task, directory, observation)
                self._notify("task_finished", task=task, result=task_result)
                result["tasks"].append(task_result)
                failed |= task_result["status"] != "success"
                result["status"] = aggregate_status(result["tasks"])
                write_json_atomic(directory / "result.json", result)
        except BaseException:
            self._notify("aborted")
            raise
        finally:
            kill_worker(self.process)
            self.process = None
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        result["status"] = (
            "interrupted"
            if self.interruptions
            else aggregate_status(result["tasks"])
        )
        result["finished_at"] = time.time()
        write_json_atomic(directory / "result.json", result)
        self._notify("run_finished", result=result)
        return result, directory

    def _run_task(self, task, run_directory, observation):
        directory = run_directory / "tasks" / task["name"]
        for name in ("raw", "logs", "artifacts"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        self._notify("task_started", task=task, directory=directory)
        implementation = benchmarks.get(task["scenario"])
        handles = []
        source = directory / ".worker-input.json"
        errors = []
        outcome = None
        try:
            environment = dict(os.environ)
            package = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (
                        package,
                        environment.get("PYTHONPATH"),
                    ),
                )
            )
            p = task["cases"][0]["parameters"]
            for name in implementation.resources:
                handle = resources.get(name)().open(directory / "artifacts", p)
                handles.append(handle)
                environment.update(handle.environment())
            with source.open("x", encoding="utf-8") as stream:
                os.chmod(source, 0o600)
                json.dump(
                    {
                        "task": task,
                        "directory": str(directory),
                        "stop_file": str(self.stop_file),
                        "observation": observation,
                    },
                    stream,
                )
            with (directory / "logs" / "stdout.log").open("w") as stdout:
                with (directory / "logs" / "stderr.log").open("w") as stderr:
                    self.process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "luban_meter.runtime.task_worker",
                            str(source),
                        ],
                        stdout=stdout,
                        stderr=stderr,
                        env=environment,
                        start_new_session=os.name == "posix",
                    )
                    deadline = time.monotonic() + p["task_timeout"]
                    shown = set()
                    while self.process.poll() is None:
                        for prepared in sorted(
                            (directory / "raw").glob("*.prepared.json")
                        ):
                            if prepared not in shown:
                                self._notify("log")
                                print(
                                    f"准备完成 {task['name']}/"
                                    f"{prepared.stem}: "
                                    + prepared.read_text(),
                                    flush=True,
                                )
                                prepared.with_suffix(".ack").touch()
                                shown.add(prepared)
                        if (
                            self.interrupted_at is not None
                            and time.monotonic() - self.interrupted_at
                            >= p["interrupt_grace"]
                        ):
                            errors.append("interrupt grace expired")
                            kill_worker(self.process)
                            break
                        if time.monotonic() >= deadline:
                            errors.append("task timeout")
                            kill_worker(self.process)
                            break
                        self._notify("poll")
                        time.sleep(0.05)
                    if self.process.returncode:
                        errors.append(
                            "worker exited with code "
                            f"{self.process.returncode}"
                        )
            target = directory / "result.json"
            if target.is_file():
                outcome = json.loads(target.read_text())
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            kill_worker(self.process)
            self.process = None
            source.unlink(missing_ok=True)
            for handle in reversed(handles):
                try:
                    handle.cleanup()
                except Exception as exc:
                    errors.append(f"resource cleanup failed: {exc}")
        if outcome is None:
            outcome = {
                "kind": "task",
                "name": task["name"],
                "scenario": task["scenario"],
                "category": task["category"],
                "status": "failed",
                "cases": [],
            }
        if errors:
            outcome["status"] = "failed"
            outcome["errors"] = errors
        if self.interruptions:
            outcome["status"] = "interrupted"
        if errors or self.interruptions:
            recover_cases(task, directory, outcome, outcome["status"])
        outcome["artifacts"] = str(directory)
        write_json_atomic(
            directory / "raw" / "manifest.json",
            {
                "kind": "raw_manifest",
                "scenario": task["scenario"],
                "events": "events.jsonl",
                "cases": [
                    {
                        "id": case["id"],
                        "status": case["status"],
                        "path": case.get("raw"),
                    }
                    for case in outcome["cases"]
                ],
            },
        )
        write_json_atomic(directory / "result.json", outcome)
        return outcome
