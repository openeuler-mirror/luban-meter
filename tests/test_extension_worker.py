"""Explicit imports must register extensions in parent and child processes."""

import errno
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

EXTENSION = """
import time
from luban_meter.api import (
    ModelServiceQualityBenchmark, Metric, Parameter, benchmarks, metrics,
)
from luban_meter.log import get_logger

@metrics.register("fixture_score")
class Score(Metric):
    requires = frozenset({"score"})
    def compute(self, facts):
        if facts["parameters"]["failure"] == "metric":
            raise ValueError("fixture metric failed")
        time.sleep(facts["parameters"]["metric_delay"])
        rows = [r for r in facts["samples"] if r["state"] == "complete"]
        return {"score": {"value": sum(r["facts"]["score"] for r in rows),
                          "unit": "point", "sample_count": len(rows)}}

@benchmarks.register("fixture")
class Fixture(ModelServiceQualityBenchmark):
    parameters = {"metric_delay": Parameter(float, 0.0, minimum=0),
                  "delay": Parameter(float, 0.0, minimum=0),
                  "failure": Parameter(str, "none"),
                  "fixture_secret": Parameter(str, "", secret=True),
                  "calls_per_sample": Parameter(int, 1, minimum=1)}
    capabilities = frozenset({"score"})
    default_metrics = ("fixture_score",)
    def prepare(self, context):
        self.failure = context.parameters["failure"]
        if self.failure == "stderr":
            get_logger(__name__).info("WORKER_ENCODING_MARKER 中文错误")
        if self.failure == "prepare":
            raise ValueError("fixture preparation failed")
        context.total_samples = 10
    def samples(self, context):
        return ({"n": n} for n in range(10))
    def evaluate(self, sample, context):
        if self.failure == "sample":
            raise ValueError("fixture sample failed")
        if self.failure == "request":
            def fail(ticket):
                raise ValueError("fixture request failed "
                                 + context.parameters["fixture_secret"])
            context.request(fail)
        for _ in range(context.parameters["calls_per_sample"]):
            context.request(lambda ticket: time.sleep(
                context.parameters["delay"]))
        return {"score": 1}
    def close(self):
        if self.failure == "stderr":
            import sys
            sys.stderr.write("worker-stream-" + "中" * 100000 + "\\n")
        if self.failure == "cleanup":
            raise ValueError("fixture cleanup failed")
"""


@pytest.fixture(scope="module")
def extension(tmp_path_factory):
    root = tmp_path_factory.mktemp("extension")
    package = root / "luban_meter"
    shutil.copytree(
        Path(__file__).parents[1] / "src/luban_meter",
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (package / "benchmarking/fixture.py").write_text(EXTENSION)
    with (package / "benchmarking/__init__.py").open("a") as stream:
        stream.write("\nfrom . import fixture\n")
    return {**os.environ, "PYTHONPATH": str(root)}


def command(tmp_path, *options):
    return [
        sys.executable,
        "-m",
        "luban_meter.cli",
        "run",
        "--scenario",
        "fixture",
        "--max-concurrency",
        "1",
        "--result-dir",
        str(tmp_path),
        *options,
    ]


def instance_log(path):
    (log_file,) = (path.parent / "log").iterdir()
    return log_file.read_text()


def read_result(tmp_path):
    (path,) = tmp_path.glob("*/result.json")
    return json.loads(path.read_text()), path


def test_custom_worker_and_partial_multi_request(extension, tmp_path):
    completed = subprocess.run(
        command(tmp_path, "--num-requests", "5", "--calls-per-sample", "2"),
        env=extension,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    result, path = read_result(tmp_path)
    case = result["tasks"][0]["cases"][0]
    assert case["window"]["admitted_requests"] == 5
    assert case["coverage"]["complete"] == 2
    assert case["coverage"]["incomplete"] == 1
    assert case["coverage"]["not_started"] == 7
    assert case["metrics"]["fixture_score"]["score"]["value"] == 2
    assert path.with_name("report.html").is_file()
    run_log = instance_log(path)
    task_log = run_log
    assert "Run started" in run_log
    assert "Worker started: pid=" in run_log
    assert "Task finished: status=success" in run_log
    assert "HTML report saved" in run_log
    assert "CLI run completed" in run_log
    for step in ("Preparation finished", "Warmup finished",
                 "Collection finished", "Raw facts saved", "Metric finished",
                 "Case cleanup finished", "Worker finished"):
        assert step in task_log
    assert "pid=" in task_log and "task=task_1 case=case_1" in task_log
    assert "request=request-" not in task_log  # No per-success-request logs.
    assert "Worker initializing" in completed.stderr
    assert "Case finished: status=success" in completed.stderr
    (only_file,) = (path.parent / "log").iterdir()
    assert only_file.name.startswith(f"luban-meter-{path.parent.name}-")
    assert "Resolved Case" in run_log and "Starting command" in run_log
    assert "Case prepared:" in run_log
    assert "Resolved Case:" not in completed.stderr
    assert "Case prepared:" not in completed.stderr
    assert "Parsed command and options" in run_log


def run_in_terminal(argv, environment):
    import pty

    master, slave = pty.openpty()
    process = subprocess.Popen(
        argv, env=environment, stdout=subprocess.PIPE, stderr=slave,
    )
    os.close(slave)
    chunks = []
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not select.select([master], [], [], 0.1)[0]:
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
            if not chunk:
                break
            chunks.append(chunk)
        else:
            pytest.fail("CLI did not close terminal output")
        stdout, _ = process.communicate(timeout=5)
        return process.returncode, stdout.decode(), b"".join(chunks).decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


@pytest.mark.skipif(os.name != "posix", reason="PTY requires POSIX")
@pytest.mark.parametrize("mode", ["tty", "pipe", "no-progress"])
@pytest.mark.parametrize("no_color", [False, True])
@pytest.mark.parametrize("failure", ["none", "metric"])
def test_cli_worker_logs_share_progress_output(
    extension, tmp_path, mode, no_color, failure,
):
    environment = {**extension, "TERM": "xterm", "COLUMNS": "180"}
    environment.pop("NO_COLOR", None)
    environment.pop("LUBAN_LOG_DIR", None)
    if no_color:
        environment["NO_COLOR"] = "1"
    argv = command(
        tmp_path, "--num-requests", "6", "--delay", "0.08",
        "--no-summary", "--failure", failure,
        *(["--no-progress"] if mode == "no-progress" else []),
    )
    if mode == "pipe":
        completed = subprocess.run(
            argv, env=environment, capture_output=True, text=True, timeout=15,
        )
        code, stdout, stderr = (
            completed.returncode, completed.stdout, completed.stderr,
        )
    else:
        code, stdout, stderr = run_in_terminal(argv, environment)
    assert code == (0 if failure == "none" else 1), stderr
    assert stdout == ""
    result, path = read_result(tmp_path)
    log = instance_log(path)
    assert log.count("Run started:") == 1
    assert log.count("Collection finished:") == 1
    assert log.count("Worker finished:") == 1
    assert "\x1b" not in log
    assert "Resolved Case:" in log and "Case prepared:" in log
    assert "Resolved Case:" not in stderr and "Case prepared:" not in stderr
    assert "request=request-" not in log
    assert stderr.count("Collection finished:") == 1
    mixed = [line for line in stderr.splitlines()
             if "case1/1" in line and re.search(r"20\d\d-\d\d-\d\d ", line)]
    assert mixed == []
    worker = next(line for line in stderr.splitlines()
                  if "Collection finished:" in line)
    assert "luban_meter.runtime.evaluation_session" in worker
    assert "task=task_1 case=case_1" in worker
    plain_worker = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", worker)
    assert plain_worker in log
    parent = next(line for line in log.splitlines() if "Run started:" in line)
    assert re.search(r"pid=(\d+)", parent).group(1) != re.search(
        r"pid=(\d+)", worker,
    ).group(1)
    assert ("\x1b[32mINFO\x1b[0m" in worker) is (
        mode != "pipe" and not no_color
    )
    if failure == "metric":
        assert "ValueError: fixture metric failed" in stderr
        assert "Traceback (most recent call last)" in stderr
    if mode == "no-progress":
        assert "当前任务" not in stderr and "case1/1" not in stderr
    else:
        assert "case1/1" in stderr and "状态 " in stderr
    assert result["status"] == (
        "success" if failure == "none" else "partial_failed"
    )


def test_worker_stderr_pipe_drains_long_utf8_output(extension, tmp_path):
    completed = subprocess.run(
        command(tmp_path, "--num-requests", "1", "--failure", "stderr",
                "--no-progress", "--no-summary"),
        env=extension, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0
    assert "worker-stream-" + "中" * 100000 + "\n" in completed.stderr
    assert completed.stderr.count("Worker finished:") == 1
    _, path = read_result(tmp_path)
    assert "worker-stream-" not in instance_log(path)


@pytest.mark.parametrize("encoding", [
    "gb18030", "gb18030:strict", ":strict", "",
])
def test_worker_stderr_pipe_preserves_python_io_encoding(
    extension, tmp_path, encoding,
):
    environment = {**extension, "PYTHONIOENCODING": encoding}
    actual_stream = subprocess.run(
        [sys.executable, "-c", "import json, sys; print(json.dumps({"
         "'encoding': sys.stderr.encoding, 'errors': sys.stderr.errors}))"],
        env=environment, capture_output=True, check=True,
    )
    stream = json.loads(actual_stream.stdout)
    assert stream["errors"] == "backslashreplace"
    completed = subprocess.run(
        command(tmp_path, "--num-requests", "1", "--failure", "stderr",
                "--no-progress", "--no-summary"),
        env=environment, capture_output=True, timeout=15,
    )
    stderr = completed.stderr.decode(stream["encoding"])
    assert completed.returncode == 0, stderr
    assert "WORKER_ENCODING_MARKER 中文错误" in stderr
    assert "worker-stream-" + "中" * 100000 + "\n" in stderr
    _, path = read_result(tmp_path)
    assert "WORKER_ENCODING_MARKER 中文错误" in instance_log(path)


@pytest.mark.parametrize("failure", [
    "prepare", "request", "sample", "metric", "cleanup",
])
def test_caught_failures_keep_tracebacks(extension, tmp_path, failure):
    completed = subprocess.run(
        command(tmp_path, "--failure", failure, "--num-requests", "1",
                "--fixture-secret", "fixture-key-that-must-be-redacted"),
        env=extension, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 1, completed.stderr
    _, path = read_result(tmp_path)
    task_log = instance_log(path)
    assert "Traceback (most recent call last)" in task_log
    assert "ValueError: fixture" in task_log
    assert "fixture-key-that-must-be-redacted" not in task_log
    assert "fixture-key-that-must-be-redacted" not in completed.stderr
    assert "task=task_1 case=case_1" in task_log
    if failure == "request":
        assert "sample=0 request=request-0" in task_log
        assert "fixture request failed [redacted]" in task_log
    if failure == "prepare":
        # Recovery deletes raw facts/events, but keeps the diagnostic log.
        assert "Incomplete Case discarded" in task_log


@pytest.mark.parametrize("level", ["debug", "info", "warn", "error"])
def test_cli_log_level_reaches_worker(extension, tmp_path, level):
    completed = subprocess.run(
        command(tmp_path, "--failure", "request", "--num-requests", "1",
                "--log-level", level),
        env=extension, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 1, completed.stderr
    _, path = read_result(tmp_path)
    run_log = instance_log(path)
    task_log = run_log
    assert ("Run started" in run_log) is (level in ("debug", "info"))
    assert ("Worker initializing" in task_log) is (level in ("debug", "info"))
    assert ("Launching worker" in run_log) is (level == "debug")
    assert ("Worker logging configured" in task_log) is (level == "debug")
    assert "Request failed" in task_log
    assert "Traceback (most recent call last)" in task_log
    assert "Request failed" in completed.stderr


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_log_directory_overrides_default(extension, tmp_path, explicit):
    env_dir, cli_dir = tmp_path / "env-log", tmp_path / "cli-log"
    options = ["--log-dir", str(cli_dir)] if explicit else []
    completed = subprocess.run(
        command(tmp_path / "runs", "--num-requests", "1", *options),
        env={**extension, "LUBAN_LOG_DIR": str(env_dir)},
        capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    result, path = read_result(tmp_path / "runs")
    target = cli_dir if explicit else env_dir
    (log_file,) = target.iterdir()
    assert "Worker initializing" in log_file.read_text()
    assert not (path.parent / "log").exists()
    assert str(log_file) in completed.stderr
    assert str(log_file) not in completed.stdout
    parameters = result["tasks"][0]["cases"][0]["parameters"]
    assert "log_dir" not in parameters and "log_level" not in parameters


def test_invalid_cli_log_level_fails_before_run(extension, tmp_path):
    completed = subprocess.run(
        command(tmp_path / "runs", "--log-level", "invalid"),
        env=extension, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
    (log_file,) = (tmp_path / "runs").glob("*/log/*.log")
    assert "Argument parsing failed" in log_file.read_text()


def test_global_log_options_are_inherited_by_run(extension, tmp_path):
    log_dir = tmp_path / "custom-log"
    invocation = command(tmp_path / "runs", "--num-requests", "1")
    invocation[3:3] = ["--log-level", "debug", "--log-dir", str(log_dir)]
    completed = subprocess.run(
        invocation, env=extension, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    (log_file,) = log_dir.iterdir()
    assert "Launching worker" in log_file.read_text()
    assert "Worker logging configured" in log_file.read_text()


def test_report_failure_is_logged_after_worker_exits(extension, tmp_path):
    script = """
import sys
from luban_meter import cli
def fail_report(*args, **kwargs):
    raise OSError("fixture report failed")
cli.write_html = fail_report
raise SystemExit(cli.main(sys.argv[1:]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script,
         *command(tmp_path, "--num-requests", "1")[3:]],
        env=extension, capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 2
    result, path = read_result(tmp_path)
    assert result["status"] == "success"
    run_log = instance_log(path)
    assert "Run finished: status=success" in run_log
    assert "HTML report started" in run_log
    assert "Traceback (most recent call last)" in run_log
    assert "OSError: fixture report failed" in run_log


def wait_for_request(process, tmp_path):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        for path in tmp_path.rglob("events.jsonl"):
            if '"kind": "request_started"' in path.read_text():
                return
        if process.poll() is not None:
            raise AssertionError(process.communicate())
        time.sleep(0.02)
    raise AssertionError("worker did not start")


@pytest.mark.parametrize("force", [False, True])
def test_interrupt_discards_incomplete_evidence(extension, tmp_path, force):
    process = subprocess.Popen(
        command(tmp_path, "--delay", "0.5"),
        env=extension,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_request(process, tmp_path)
        process.send_signal(signal.SIGINT)
        if force:
            time.sleep(0.05)
            process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result, path = read_result(tmp_path)
    assert result["status"] == "interrupted"
    run_log = instance_log(path)
    assert "Interrupt requested" in run_log
    assert "Run finished: status=interrupted" in run_log
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "discarded"
    assert "parameters" in case and "reason" in case
    assert "raw" not in case and "metrics" not in case
    raw = next(tmp_path.rglob("manifest.json")).parent
    assert not (raw / "case_1.json").exists()
    for events in tmp_path.rglob("events.jsonl"):
        assert '"case_id": "case_1"' not in events.read_text()


@pytest.mark.parametrize("fail_fast", [False, True])
def test_timeout_recovers_and_continues(extension, tmp_path, fail_fast):
    config = tmp_path / "tasks.json"
    config.write_text(
        json.dumps(
            {
                "tasks": [
                    {"name": "slow", "delay": 2, "task_timeout": 0.3},
                    {"name": "next", "num_requests": 1},
                ]
            }
        )
    )
    result = subprocess.run(
        command(
            tmp_path / "out",
            "--config",
            str(config),
            *(["--fail-fast"] if fail_fast else []),
        ),
        env=extension,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1, result.stderr
    report, path = read_result(tmp_path / "out")
    assert [t["status"] for t in report["tasks"]] == [
        "failed",
        "skipped" if fail_fast else "success",
    ]
    assert "task timeout" in report["tasks"][0]["errors"]
    run_log = instance_log(path)
    assert "task=slow" in run_log and "Task timeout; killing worker" in run_log
    assert "Recovering Case results" in run_log


def test_first_interrupt_grace_expiry_kills_worker(extension, tmp_path):
    process = subprocess.Popen(
        command(tmp_path, "--delay", "5", "--interrupt-grace", "0.1"),
        env=extension,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_request(process, tmp_path)
        started = time.monotonic()
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 130, stderr
        assert time.monotonic() - started < 3
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result, path = read_result(tmp_path)
    assert result["tasks"][0]["cases"][0]["status"] == "discarded"
    assert "interrupt grace expired" in result["tasks"][0]["errors"]
    assert "Interrupt grace expired; killing worker" in instance_log(path)


def test_interrupt_during_metrics_retains_committed_collection(
    extension, tmp_path
):
    process = subprocess.Popen(
        command(tmp_path, "--num-requests", "1", "--metric-delay", "5"),
        env=extension,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        original = None
        target = None
        while time.monotonic() < deadline:
            for path in tmp_path.rglob("case_1.json"):
                facts = json.loads(path.read_text())
                if facts.get("collection_complete"):
                    target, original = path, path.read_bytes()
                    break
            if target:
                break
            time.sleep(0.02)
        assert target, "collection did not commit"
        process.send_signal(signal.SIGINT)
        time.sleep(0.05)
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 130, stderr
        assert target.read_bytes() == original
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result, _ = read_result(tmp_path)
    case = result["tasks"][0]["cases"][0]
    assert case["collection_complete"]
    assert case["status"] != "discarded"
    assert case["metrics_status"] == "not_computed"
