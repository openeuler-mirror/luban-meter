"""Explicit imports must register extensions in parent and child processes."""

import json
import os
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

@metrics.register("fixture_score")
class Score(Metric):
    requires = frozenset({"score"})
    def compute(self, facts):
        time.sleep(facts["parameters"]["metric_delay"])
        rows = [r for r in facts["samples"] if r["state"] == "complete"]
        return {"score": {"value": sum(r["facts"]["score"] for r in rows),
                          "unit": "point", "sample_count": len(rows)}}

@benchmarks.register("fixture")
class Fixture(ModelServiceQualityBenchmark):
    parameters = {"metric_delay": Parameter(float, 0.0, minimum=0),
                  "delay": Parameter(float, 0.0, minimum=0),
                  "calls_per_sample": Parameter(int, 1, minimum=1)}
    capabilities = frozenset({"score"})
    default_metrics = ("fixture_score",)
    def prepare(self, context):
        context.total_samples = 10
    def samples(self, context):
        return ({"n": n} for n in range(10))
    def evaluate(self, sample, context):
        for _ in range(context.parameters["calls_per_sample"]):
            context.request(lambda ticket: time.sleep(
                context.parameters["delay"]))
        return {"score": 1}
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
    report, _ = read_result(tmp_path / "out")
    assert [t["status"] for t in report["tasks"]] == [
        "failed",
        "skipped" if fail_fast else "success",
    ]
    assert "task timeout" in report["tasks"][0]["errors"]


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
    result, _ = read_result(tmp_path)
    assert result["tasks"][0]["cases"][0]["status"] == "discarded"
    assert "interrupt grace expired" in result["tasks"][0]["errors"]


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
