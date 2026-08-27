"""Tests for the ceval inference benchmark: parameters, scoring, results."""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from luban_meter.benchmark.inference.ceval import benchmark as ceval
from luban_meter.core.models import RunRequest
from luban_meter.core.registry import BenchmarkRegistry

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT / "src" / "luban_meter" / "benchmark" / "inference" / "ceval" / "result.py"
)

MODULE_COUNT = 0


def load_result_module():
    global MODULE_COUNT
    MODULE_COUNT += 1
    spec = importlib.util.spec_from_file_location(
        f"ceval_result_test_{MODULE_COUNT}", RESULT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def base_parameters(tmp_path: Path) -> dict[str, Any]:
    dataset = tmp_path / "val.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "一",
                        "choices": ["a", "b", "c", "d"],
                        "answer": "B",
                        "subject": "计算机网络",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "question": "二",
                        "choices": ["e", "f", "g", "h"],
                        "answer": "C",
                        "subject": "计算机网络",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    return {
        "dataset_path": str(dataset),
        "max_samples": 2,
        "few_shot": 0,
        "prompt_format": "base",
    }


def test_validate_parameters(tmp_path: Path) -> None:
    config = ceval.validate_parameters(base_parameters(tmp_path))
    assert config["eval_mode"] == "ppl"
    assert config["prompt_format"] == "base"
    with pytest.raises(ValueError):
        ceval.validate_parameters(
            {**base_parameters(tmp_path), "temperature": 1.0}
        )
    with pytest.raises(ValueError):
        ceval.validate_parameters({**base_parameters(tmp_path), "eval_mode": "x"})
    with pytest.raises(ValueError):
        ceval.validate_parameters({**base_parameters(tmp_path), "few_shot": 3})
    with pytest.raises(ValueError):
        ceval.validate_parameters(
            {**base_parameters(tmp_path), "prompt_format": "chat"}
        )


def _build_echo_logprobs(full_prompt: str) -> list[Any]:
    """Realistic ``echo=true, max_tokens=1`` logprobs payload.

    The fake tokenizer yields one token per character. Continuation tokens carry
    the per-letter score and the trailing generated token is set to a distinct
    sentinel (``-99.0``) so tests can confirm it is excluded from the score.
    """
    marker = "答案："
    idx = full_prompt.rfind(marker)
    prompt_count = idx + len(marker) if idx >= 0 else 0
    continuation = full_prompt[prompt_count:].strip()
    letter = continuation[0] if continuation else "A"
    score = -0.1 if letter == "B" else -2.0
    full_count = len(full_prompt)
    token_logprobs: list[Any] = [None] + [-1.0] * (full_count - 1)
    for position in range(prompt_count, full_count):
        token_logprobs[position] = score
    token_logprobs.append(-99.0)  # generated token; must be excluded
    return token_logprobs


class FakeClient:
    """Realistic echo+logprobs fake so choice B always wins."""

    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text)))

    def completion_logprobs(self, full_prompt: str) -> dict[str, Any]:
        token_logprobs = _build_echo_logprobs(full_prompt)
        return {
            "tokens": ["t"] * len(token_logprobs),
            "token_logprobs": token_logprobs,
        }


def test_score_choices_prefers_best_letter() -> None:
    scores, latency_ms = ceval.score_choices(
        FakeClient(), "问题……\n答案：", ["a", "b", "c", "d"]
    )
    assert set(scores) == {"A", "B", "C", "D"}
    assert max(scores, key=scores.get) == "B"
    # The generated token (-99.0) must be excluded; otherwise B would be ~-17.
    assert scores["B"] == pytest.approx(-0.1)
    assert latency_ms == 0.0


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/v1/chat/completions":
            response = {
                "choices": [{"message": {"content": "答案是 B"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }
        elif self.path == "/v1/completions":
            prompt = body.get("prompt", "")
            token_logprobs = _build_echo_logprobs(prompt)
            response = {
                "choices": [
                    {
                        "text": "",
                        "logprobs": {
                            "tokens": ["t"] * len(token_logprobs),
                            "token_logprobs": token_logprobs,
                        },
                    }
                ],
                "usage": {"prompt_tokens": len(token_logprobs)},
            }
        elif self.path == "/tokenize":
            text = body.get("prompt", "")
            response = {"tokens": list(range(len(text))), "count": len(text)}
        else:
            response = {}
        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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


def test_end_to_end_gen_mode(fake_service: str, tmp_path: Path) -> None:
    parameters = base_parameters(tmp_path)
    parameters.update(
        {"service_url": fake_service, "eval_mode": "gen", "prompt_format": "chat"}
    )
    raw = ceval.run_benchmark({"model_name": "test-model"}, parameters)
    assert raw["status"] == "success"
    assert raw["metrics"]["counts"] == {
        "total": 2,
        "correct": 1,
        "parse_failed": 0,
        "service_failed": 0,
    }
    assert raw["metadata"]["eval_mode"] == "gen"
    assert raw["metrics"]["samples"][0]["prediction"] == "B"

    result_module = load_result_module()
    final = result_module.process(raw)
    assert final["status"] == "success"
    ceval_metrics = final["metrics"]["task_view"]["ceval"]
    assert ceval_metrics["accuracy"]["value"] == pytest.approx(0.5)
    assert ceval_metrics["accuracy"]["unit"] == "ratio"
    by_subject = ceval_metrics["accuracy_by_subject"]["计算机网络"]
    assert by_subject["value"] == pytest.approx(0.5)
    assert by_subject["count"] == 2


def test_end_to_end_ppl_mode(fake_service: str, tmp_path: Path) -> None:
    parameters = base_parameters(tmp_path)
    parameters["service_url"] = fake_service
    raw = ceval.run_benchmark({"model_name": "test-model"}, parameters)
    assert raw["status"] == "success"
    samples = raw["metrics"]["samples"]
    assert all(sample["prediction"] == "B" for sample in samples)
    assert all(sample["choice_scores"] is not None for sample in samples)
    assert raw["metrics"]["counts"]["correct"] == 1


def test_process_partial_and_failed() -> None:
    result_module = load_result_module()

    def sample(status: str, correct: bool | None) -> dict[str, Any]:
        return {
            "id": "x",
            "subject": "s",
            "status": status,
            "prediction": "B" if status == "success" else None,
            "reference": "B",
            "correct": correct,
            "input_tokens": 1,
            "output_tokens": 1,
        }

    raw = {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metrics": {"samples": [sample("success", True), sample("service_failed", None)]},
        "metadata": {"model": "m"},
    }
    final = result_module.process(raw)
    assert final["status"] == "partial_failed"
    ceval_metrics = final["metrics"]["task_view"]["ceval"]
    assert ceval_metrics["accuracy"]["value"] == pytest.approx(1.0)
    assert ceval_metrics["service_failed"]["value"] == 1

    failed = result_module.process(
        {
            "status": "failed",
            "metrics": {},
            "metadata": {"measurement": "m"},
            "error": {"type": "RuntimeError", "message": "boom"},
        }
    )
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == "boom"

    with pytest.raises(ValueError):
        result_module.process(
            {
                "status": "success",
                "metrics": {"samples": [sample("service_failed", None)]},
                "metadata": {},
            }
        )


def test_registry_discovers_ceval() -> None:
    registry = BenchmarkRegistry()
    assert "ceval" in registry.list_benchmarks("inference")
    config = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "ceval"
        / "ceval.yaml"
    )
    request = RunRequest(
        run_id="inference-ceval-test",
        module="inference",
        benchmark="ceval",
        config=config,
        model_path=None,
        model_name=None,
        output_dir=ROOT / "runs",
    )
    run = registry.resolve(request)
    assert run.benchmark.benchmark_entry.name == "benchmark.py"
    assert run.benchmark.result_handler.name == "result.py"
    assert run.parameters["eval_mode"] == "ppl"
    assert run.parameters["stop"] == ["\n"]
