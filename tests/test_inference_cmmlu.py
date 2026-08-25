"""Tests for the cmmlu inference benchmark: parameters, scoring, results."""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from luban_meter.benchmark.inference.cmmlu import benchmark as cmmlu
from luban_meter.core.models import RunRequest
from luban_meter.core.registry import BenchmarkRegistry

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT / "src" / "luban_meter" / "benchmark" / "inference" / "cmmlu" / "result.py"
)

MODULE_COUNT = 0


def load_result_module():
    global MODULE_COUNT
    MODULE_COUNT += 1
    spec = importlib.util.spec_from_file_location(
        f"cmmlu_result_test_{MODULE_COUNT}", RESULT_PATH
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
                        "subject": "中国历史",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "question": "二",
                        "choices": ["e", "f", "g", "h"],
                        "answer": "C",
                        "subject": "伦理学",
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
    }


def test_validate_parameters(tmp_path: Path) -> None:
    config = cmmlu.validate_parameters(base_parameters(tmp_path))
    assert config["eval_mode"] == "ppl"
    assert config["prompt_version"] == "cmmlu-v1"
    with pytest.raises(ValueError):
        cmmlu.validate_parameters(
            {**base_parameters(tmp_path), "prompt_version": "ceval-v1"}
        )
    with pytest.raises(ValueError):
        cmmlu.validate_parameters({**base_parameters(tmp_path), "temperature": 1.0})


class FakeClient:
    """Returns deterministic logprobs so choice B always wins."""

    def tokenize(self, text: str) -> list[int]:
        return [0, 1, 2]

    def completion_logprobs(self, prompt: str) -> dict[str, Any]:
        continuation = prompt.rsplit("答案：", 1)[-1].strip()
        letter = continuation[0]
        score = -0.1 if letter == "B" else -2.0
        token_logprobs = [None, -1.0, -1.0, -1.0, score, score, score]
        return {
            "tokens": ["t"] * len(token_logprobs),
            "token_logprobs": token_logprobs,
        }


def test_score_choices_prefers_best_letter() -> None:
    scores, latency_ms = cmmlu.score_choices(
        FakeClient(), "问题……\n答案：", ["a", "b", "c", "d"]
    )
    assert max(scores, key=scores.get) == "B"
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
            continuation = prompt.rsplit("答案：", 1)[-1].strip()
            letter = continuation[0] if continuation else "A"
            score = [-0.1] * 3 if letter == "B" else [-2.0] * 3
            token_logprobs = [None, -1.0, -1.0, -1.0, *score]
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
            response = {"tokens": [1, 2, 3], "count": 3}
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


def test_end_to_end_ppl_mode(fake_service: str, tmp_path: Path) -> None:
    parameters = base_parameters(tmp_path)
    parameters["service_url"] = fake_service
    raw = cmmlu.run_benchmark({"model_name": "test-model"}, parameters)
    assert raw["status"] == "success"
    assert raw["metadata"]["measurement"] == "cmmlu_choice_accuracy_online_service"
    assert raw["metadata"]["dataset"] == "CMMLU"
    assert raw["metrics"]["counts"]["correct"] == 1

    result_module = load_result_module()
    final = result_module.process(raw)
    assert final["status"] == "success"
    cmmlu_metrics = final["metrics"]["task_view"]["cmmlu"]
    assert cmmlu_metrics["accuracy"]["value"] == pytest.approx(0.5)
    assert "中国历史" in cmmlu_metrics["accuracy_by_subject"]


def test_end_to_end_gen_mode(fake_service: str, tmp_path: Path) -> None:
    parameters = base_parameters(tmp_path)
    parameters.update({"service_url": fake_service, "eval_mode": "gen"})
    raw = cmmlu.run_benchmark({"model_name": "test-model"}, parameters)
    assert raw["status"] == "success"
    assert raw["metrics"]["counts"] == {
        "total": 2,
        "correct": 1,
        "parse_failed": 0,
        "service_failed": 0,
    }


def test_registry_discovers_cmmlu() -> None:
    registry = BenchmarkRegistry()
    assert "cmmlu" in registry.list_benchmarks("inference")
    config = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "cmmlu"
        / "cmmlu.yaml"
    )
    request = RunRequest(
        run_id="inference-cmmlu-test",
        module="inference",
        benchmark="cmmlu",
        config=config,
        model_path=None,
        model_name=None,
        output_dir=ROOT / "runs",
    )
    run = registry.resolve(request)
    assert run.benchmark.result_handler.name == "result.py"
    assert run.parameters["prompt_version"] == "cmmlu-v1"
    assert run.parameters["dataset_path"] == "data/cmmlu/val.jsonl"

