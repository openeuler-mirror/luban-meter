"""Parent-owned execution, interruption, evidence and resource cleanup."""

from __future__ import annotations

import codecs
import json
import locale
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import public_plan
from luban_meter.core.token_totals import run_throughput
from luban_meter.log import (
    configure_logging,
    console_progress,
    get_log_file_path,
    get_logger,
    instance_filename,
    log_context,
    log_parameters,
    relay_console,
    resolve_log_dir,
    resolve_log_level,
    task_secrets,
)
from luban_meter.runtime.case_recovery import recover_cases
from luban_meter.runtime.resource_manager import resources
from luban_meter.runtime.task_worker import aggregate_status
from luban_meter.utils.json_io import write_json_atomic
from luban_meter.utils.run_id import create_run_id

logger = get_logger(__name__)


def kill_worker(process):
    if process is None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        logger.debug("Worker already exited")
    process.wait()


class RunCoordinator:
    def __init__(self):
        self.process = None
        self.interruptions = 0
        self.stop_file = None
        self.interrupted_at = None
        self.progress = None
        self.worker_stderr = None
        self.stderr_decoder = None
        self.stderr_pending = ""

    def _drain_worker_stderr(self, *, final=False):
        if self.worker_stderr is None:
            return
        # Nonblocking reads keep timeout, stop and progress polling responsive.
        read_bytes = 0
        try:
            while final or read_bytes < 1024 * 1024:
                try:
                    chunk = os.read(self.worker_stderr.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                read_bytes += len(chunk)
                self.stderr_pending += self.stderr_decoder.decode(chunk)
                complete, newline, self.stderr_pending = (
                    self.stderr_pending.rpartition("\n")
                )
                if newline:
                    relay_console(complete + newline)
            if final:
                self.stderr_pending += self.stderr_decoder.decode(
                    b"", final=True,
                )
                if self.stderr_pending:
                    relay_console(self.stderr_pending + "\n")
                    self.stderr_pending = ""
        except OSError:
            logger.warning("Worker stderr forwarding failed", exc_info=True)
        finally:
            if final:
                try:
                    self.worker_stderr.close()
                except OSError:
                    logger.warning("Worker stderr close failed", exc_info=True)
                self.worker_stderr = None

    def _notify(self, kind, **facts):
        if self.progress is not None:
            try:
                self.progress(kind, **facts)
            except Exception:
                # Display failures never change execution or result status.
                self.progress = None
                logger.warning("Progress display disabled", exc_info=True)

    def interrupt(self, signum, frame):
        self.interruptions += 1
        if self.interrupted_at is None:
            self.interrupted_at = time.monotonic()
        self.stop_file.touch()
        if self.interruptions >= 2:
            kill_worker(self.process)

    def run(
        self, plan, output, *, fail_fast=False, observation=None,
        progress=None, log_dir=None, log_level="info", run_id=None,
    ):
        self.interruptions = 0
        self.interrupted_at = None
        self.progress = progress
        self.log_level = resolve_log_level(log_level)
        from_cli = run_id is not None and get_log_file_path() is not None
        run_id = run_id or create_run_id("run")
        directory = Path(output).resolve() / run_id
        directory.mkdir(parents=True, exist_ok=from_cli)
        self.log_directory = resolve_log_dir(
            log_dir, fallback=directory / "log"
        ).resolve()
        self.log_filename = instance_filename(run_id)
        scope = nullcontext() if from_cli else configure_logging(
            self.log_directory, filename=self.log_filename,
            level=self.log_level,
            secrets=task_secrets(plan["tasks"]),
            before_emit=lambda: self._notify("log"),
        )
        with scope, console_progress(lambda: self._notify("log")):
            logger.info("Run started: id=%s tasks=%s",
                        run_id, len(plan["tasks"]))
            for task in public_plan(plan)["tasks"]:
                for case in task["cases"]:
                    with log_context(task=task["name"], case=case["id"]):
                        logger.info("Resolved Case: scenario=%s parameters=%s",
                                    task["scenario"],
                                    log_parameters(case["parameters"]),
                                    extra={"file_only": True})
            return self._run(plan, directory, fail_fast, observation)

    def _run(self, plan, directory, fail_fast, observation):
        self.stop_file = directory / ".stop"
        write_json_atomic(directory / "plan.json", public_plan(plan))
        started_monotonic = time.perf_counter()
        logger.info("Plan saved: %s", directory / "plan.json")
        result = {
            "kind": "run",
            "id": directory.name,
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
                    logger.info("Task skipped: %s", task["name"],
                                extra={"task": task["name"]})
                    task_result = {
                        "kind": "task",
                        "name": task["name"],
                        "scenario": task["scenario"],
                        "category": task["category"],
                        "status": "skipped",
                        "cases": [],
                    }
                else:
                    with log_context(task=task["name"]):
                        task_result = self._run_task(
                            task, directory, observation
                        )
                self._notify("task_finished", task=task, result=task_result)
                result["tasks"].append(task_result)
                failed |= task_result["status"] != "success"
                result["status"] = aggregate_status(result["tasks"])
                write_json_atomic(directory / "result.json", result)
        except BaseException:
            logger.exception("Run aborted")
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
        result["duration_s"] = time.perf_counter() - started_monotonic
        result["metrics"] = {}
        if any(task.get("category") == "generation_performance"
               for task in result["tasks"]):
            result["metrics"]["token_throughput"] = run_throughput(
                result["tasks"]
            )
        write_json_atomic(directory / "result.json", result)
        logger.info("Run finished: status=%s result=%s",
                    result["status"], directory / "result.json")
        self._notify("run_finished", result=result)
        return result, directory

    def _run_task(self, task, run_directory, observation):
        directory = run_directory / "tasks" / task["name"]
        for name in ("raw", "artifacts"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        self._notify("task_started", task=task, directory=directory)
        logger.info("Task started: scenario=%s cases=%s directory=%s",
                    task["scenario"], len(task["cases"]), directory)
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
                logger.info("Resource opening: %s", name)
                handle = resources.get(name)().open(directory / "artifacts", p)
                handles.append(handle)
                environment.update(handle.environment())
                logger.info("Resource opened: %s", name)
            with source.open("x", encoding="utf-8") as stream:
                os.chmod(source, 0o600)
                json.dump(
                    {
                        "task": task,
                        "directory": str(directory),
                        "stop_file": str(self.stop_file),
                        "observation": observation,
                        "log_secrets": task_secrets([task]),
                        "log_directory": str(self.log_directory),
                        "log_filename": self.log_filename,
                        "log_level": self.log_level,
                    },
                    stream,
                )
            logger.debug("Launching worker: timeout=%s grace=%s",
                         p["task_timeout"], p["interrupt_grace"])
            argv = [sys.executable, "-m", "luban_meter.runtime.task_worker",
                    str(source)]
            logger.info("Starting command: %s", argv)
            self.process = subprocess.Popen(
                argv,
                env=environment,
                start_new_session=os.name == "posix",
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self.worker_stderr = self.process.stderr
            encoding = environment.get("PYTHONIOENCODING", "")
            encoding = (encoding.partition(":")[0]
                        or locale.getpreferredencoding(False))
            self.stderr_decoder = codecs.getincrementaldecoder(encoding)(
                errors="replace",
            )
            self.stderr_pending = ""
            # https://docs.python.org/3.12/library/os.html#os.set_blocking
            os.set_blocking(self.worker_stderr.fileno(), False)
            logger.info("Worker started: pid=%s", self.process.pid)
            deadline = time.monotonic() + p["task_timeout"]
            shown = set()
            logged_interruptions = 0
            while self.process.poll() is None:
                self._drain_worker_stderr()
                if self.interruptions != logged_interruptions:
                    logger.warning("Interrupt requested: count=%s",
                                   self.interruptions)
                    logged_interruptions = self.interruptions
                for prepared in sorted(
                    (directory / "raw").glob("*.prepared.json")
                ):
                    if prepared not in shown:
                        with log_context(
                            case=prepared.name.removesuffix(".prepared.json"),
                            phase="prepare",
                        ):
                            logger.info(
                                "Case prepared: %s",
                                log_parameters(json.loads(prepared.read_text())),
                                extra={"file_only": True},
                            )
                        prepared.with_suffix(".ack").touch()
                        shown.add(prepared)
                if (
                    self.interrupted_at is not None
                    and time.monotonic() - self.interrupted_at
                    >= p["interrupt_grace"]
                ):
                    errors.append("interrupt grace expired")
                    logger.error(
                        "Interrupt grace expired; killing worker"
                    )
                    kill_worker(self.process)
                    break
                if time.monotonic() >= deadline:
                    errors.append("task timeout")
                    logger.error("Task timeout; killing worker")
                    kill_worker(self.process)
                    break
                self._notify("poll")
                time.sleep(0.05)
            self._drain_worker_stderr()
            if self.interruptions != logged_interruptions:
                logger.warning("Interrupt requested: count=%s",
                               self.interruptions)
            logger.info("Worker exited: pid=%s returncode=%s",
                        self.process.pid, self.process.returncode)
            if self.process.returncode:
                errors.append(
                    "worker exited with code "
                    f"{self.process.returncode}"
                )
                logger.error("Worker failed: returncode=%s",
                             self.process.returncode)
            target = directory / "result.json"
            if target.is_file():
                outcome = json.loads(target.read_text())
        except Exception as exc:
            logger.exception("Task execution failed")
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            logger.info("Task cleanup started")
            kill_worker(self.process)
            self._drain_worker_stderr(final=True)
            self.process = None
            source.unlink(missing_ok=True)
            for handle in reversed(handles):
                try:
                    handle.cleanup()
                except Exception as exc:
                    logger.exception("Resource cleanup failed")
                    errors.append(f"resource cleanup failed: {exc}")
            logger.info("Task cleanup finished")
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
            logger.warning("Recovering Case results after worker failure/stop")
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
        logger.info("Task finished: status=%s result=%s",
                    outcome["status"], directory / "result.json")
        return outcome
