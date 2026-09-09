"""Tests for the HumanEval pass@1 benchmark and result contract."""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from luban_meter.benchmark.inference.humaneval import benchmark as humaneval
from luban_meter.benchmark.inference.humaneval import result as humaneval_result
from luban_meter.benchmark.inference.humaneval.executor import ExecutionResult
from luban_meter.core.models import RunRequest
from luban_meter.core.registry import BenchmarkRegistry

ROOT = Path(__file__).resolve().parents[1]


def write_dataset(path: Path) -> None:
    records = [
        {
            "task_id": "HumanEvalTest/0",
            "prompt": ('def add(a: int, b: int) -> int:\n    """Return a plus b."""\n'),
            "canonical_solution": "    return a + b\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3",
            "entry_point": "add",
        },
        {
            "task_id": "HumanEvalTest/1",
            "prompt": (
                "def is_even(value: int) -> bool:\n"
                '    """Return whether value is even."""\n'
            ),
            "canonical_solution": "    return value % 2 == 0\n",
            "test": "def check(candidate):\n    assert candidate(2) is True",
            "entry_point": "is_even",
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def base_parameters(tmp_path: Path) -> dict[str, Any]:
    dataset = tmp_path / "HumanEval.jsonl"
    write_dataset(dataset)
    return {
        "dataset_path": str(dataset),
        "max_samples": 2,
        "docker_host": "unix:///test/docker.sock",
        "sandbox_image": "test-humaneval:latest",
    }


def test_validate_parameters(tmp_path: Path) -> None:
    parameters = base_parameters(tmp_path)
    config = humaneval.validate_parameters(parameters)
    assert config["samples_per_task"] == 1
    assert config["pass_k"] == [1]
    assert config["prompt_version"] == "humaneval-completion-v1"
    with pytest.raises(ValueError):
        humaneval.validate_parameters({**parameters, "samples_per_task": 2})
    with pytest.raises(ValueError):
        humaneval.validate_parameters({**parameters, "pass_k": [1, 2]})
    with pytest.raises(ValueError):
        humaneval.validate_parameters({**parameters, "prompt_format": "chat"})
    with pytest.raises(ValueError):
        humaneval.validate_parameters({**parameters, "allow_host_fallback": True})
    with pytest.raises(ValueError):
        humaneval.validate_parameters({**parameters, "docker_host": "tcp://host:2375"})


def test_bundled_official_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = humaneval.resolve_data_path("data/humaneval/HumanEval.jsonl")
    records = humaneval.prepare_samples(humaneval.load_records(dataset))
    assert len(records) == 164
    assert records[0]["task_id"] == "HumanEval/0"
    assert records[-1]["task_id"] == "HumanEval/163"


def test_prepare_samples_and_program_validation() -> None:
    sample = {
        "task_id": "x",
        "prompt": "def f():\n",
        "test": "def check(candidate):\n    assert candidate() == 1",
        "entry_point": "f",
    }
    prepared = humaneval.prepare_samples([sample])
    program = humaneval.build_program(prepared[0], "    return 1\n")
    assert "def f():\n    return 1" in program
    assert program.endswith("check(f)\n")
    with pytest.raises(ValueError):
        humaneval.prepare_samples([sample, sample])
    with pytest.raises(ValueError):
        humaneval.prepare_samples([{**sample, "test": ""}])


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        prompt = str(payload.get("prompt") or "")
        if "def add" in prompt:
            text = "    return a + b\n"
        else:
            text = "```python\n    return False\n```"
        response = {
            "choices": [{"text": text}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return None


@pytest.fixture()
def fake_service():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


class _FakeSandbox:
    def __init__(self, _config: object) -> None:
        pass

    def preflight(self) -> dict[str, Any]:
        return {
            "runtime": "fake-docker",
            "image": "test-humaneval:latest",
            "image_id": "sha256:test",
            "security_options": ["name=seccomp,profile=builtin"],
        }

    def execute(self, program: str) -> ExecutionResult:
        passed = "return a + b" in program
        return ExecutionResult(
            status="passed" if passed else "failed_test",
            passed=passed,
            execution_ms=1.0,
            error_type=None if passed else "AssertionError",
            error_message=None if passed else "",
        )


def test_end_to_end_pass_at_1(
    fake_service: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = base_parameters(tmp_path)
    parameters["service_url"] = fake_service
    monkeypatch.setattr(humaneval, "DockerSandboxExecutor", _FakeSandbox)
    raw = humaneval.run_benchmark({"model_name": "test-model"}, parameters)
    assert raw["status"] == "success"
    assert raw["metrics"]["counts"] == {
        "total": 2,
        "passed": 1,
        "parse_failed": 0,
        "service_failed": 0,
        "sandbox_failed": 0,
    }
    first, second = raw["metrics"]["samples"]
    assert first["sample_id"] == 0
    assert first["passed"] is True
    assert len(first["program_sha256"]) == 64
    assert second["execution_status"] == "failed_test"

    final = humaneval_result.process(raw)
    metrics = final["metrics"]["task_view"]["humaneval"]
    assert final["status"] == "success"
    assert metrics["pass_at_1"]["value"] == pytest.approx(0.5)
    assert metrics["pass_at_1"]["count"] == 2


def test_result_counts_infrastructure_failure_in_denominator() -> None:
    raw = {
        "status": "success",
        "metrics": {
            "samples": [
                {
                    "task_id": "a",
                    "sample_id": 0,
                    "status": "success",
                    "execution_status": "passed",
                    "passed": True,
                },
                {
                    "task_id": "b",
                    "sample_id": 0,
                    "status": "service_failed",
                    "execution_status": None,
                    "passed": False,
                },
            ]
        },
        "metadata": {},
    }
    final = humaneval_result.process(raw)
    assert final["status"] == "partial_failed"
    assert final["metrics"]["task_view"]["humaneval"]["pass_at_1"][
        "value"
    ] == pytest.approx(0.5)


def test_registry_discovers_humaneval() -> None:
    registry = BenchmarkRegistry()
    assert "humaneval" in registry.list_benchmarks("inference")
    config = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "humaneval"
        / "humaneval.yaml"
    )
    request = RunRequest(
        run_id="inference-humaneval-test",
        module="inference",
        benchmark="humaneval",
        config=config,
        model_path=None,
        model_name=None,
        output_dir=ROOT / "runs",
    )
    run = registry.resolve(request)
    assert run.parameters["samples_per_task"] == 1
    assert run.parameters["pass_k"] == [1]
