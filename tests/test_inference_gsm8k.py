"""Tests for the gsm8k inference benchmark: parameters, prompts, results."""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from luban_meter.benchmark.inference.common.prompts import render_math_prompt
from luban_meter.benchmark.inference.gsm8k import benchmark as gsm8k
from luban_meter.core.models import RunRequest
from luban_meter.core.registry import BenchmarkRegistry

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT / "src" / "luban_meter" / "benchmark" / "inference" / "gsm8k" / "result.py"
)

MODULE_COUNT = 0


def load_result_module():
    global MODULE_COUNT
    MODULE_COUNT += 1
    spec = importlib.util.spec_from_file_location(
        f"gsm8k_result_test_{MODULE_COUNT}", RESULT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def base_parameters(tmp_path: Path) -> dict[str, Any]:
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "小明有6箱苹果，每箱12个，一共多少个？",
                        "answer": "6 x 12 = 72.\n#### 72",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "q2",
                        "question": "一个数加上3等于8，这个数是多少？",
                        "answer": "8 - 3 = 5.\n#### 5",
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
    config = gsm8k.validate_parameters(base_parameters(tmp_path))
    assert config["eval_mode"] == "gen"
    assert config["max_tokens"] == 512
    with pytest.raises(ValueError):
        gsm8k.validate_parameters({**base_parameters(tmp_path), "eval_mode": "ppl"})
    with pytest.raises(ValueError):
        gsm8k.validate_parameters(
            {**base_parameters(tmp_path), "prompt_version": "ceval-v1"}
        )
    with pytest.raises(ValueError):
        gsm8k.validate_parameters({**base_parameters(tmp_path), "few_shot": 8})


def test_render_math_prompt() -> None:
    sample = {"question": "1+1=?", "answer": "2.\n#### 2"}
    few_shot = [{"question": "示例题 2+2=?", "answer": "4.\n#### 4"}]
    prompt = render_math_prompt(sample, few_shot_samples=few_shot)
    assert prompt.startswith("以下是数学应用题")
    assert "示例题 2+2=?" in prompt
    assert "#### 4" in prompt
    assert prompt.endswith("问题：1+1=?\n答案：")


def test_prepare_samples_rejects_bad_reference() -> None:
    with pytest.raises(ValueError):
        gsm8k.prepare_samples([{"question": "q", "answer": "no number here"}])
    prepared = gsm8k.prepare_samples(
        [{"question": "q", "answer": "so\n#### 1,234"}]
    )
    assert prepared[0]["reference_number"] == 1234.0


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        messages = body.get("messages") or [{}]
        content = messages[0].get("content", "") if isinstance(messages, list) else ""
        if "小明" in content:
            text = "6 乘以 12 等于 72。\n#### 72"
        else:
            text = "我不知道这个答案。"
        response = {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }
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
    few_shot = tmp_path / "few_shot.jsonl"
    few_shot.write_text(
        json.dumps(
            {"id": "fs1", "question": "示例题", "answer": "1.\n#### 1"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    parameters = base_parameters(tmp_path)
    parameters.update(
        {"service_url": fake_service, "few_shot": 1, "few_shot_path": str(few_shot)}
    )
    raw = gsm8k.run_benchmark({"model_name": "test-model"}, parameters)
    assert raw["status"] == "success"
    assert raw["metadata"]["measurement"] == "gsm8k_exact_match_online_service"
    assert raw["metrics"]["counts"] == {
        "total": 2,
        "correct": 1,
        "parse_failed": 1,
        "service_failed": 0,
    }
    good, bad = raw["metrics"]["samples"]
    assert good["prediction"] == 72.0
    assert good["reference"] == 72.0
    assert good["correct"] is True
    assert "示例题" in good["prompt"]
    assert bad["status"] == "parse_failed"

    result_module = load_result_module()
    final = result_module.process(raw)
    assert final["status"] == "success"
    gsm8k_metrics = final["metrics"]["task_view"]["gsm8k"]
    assert gsm8k_metrics["exact_match"]["value"] == pytest.approx(0.5)
    assert gsm8k_metrics["parse_failed"]["value"] == 1


def test_process_partial_and_failed() -> None:
    result_module = load_result_module()

    def sample(status: str, correct: bool | None) -> dict[str, Any]:
        return {
            "id": "x",
            "status": status,
            "prediction": 7.0 if status == "success" else None,
            "reference": 7.0,
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
    assert final["metrics"]["task_view"]["gsm8k"]["exact_match"]["value"] == pytest.approx(1.0)

    failed = result_module.process(
        {
            "status": "failed",
            "metrics": {},
            "metadata": {"measurement": "m"},
            "error": {"type": "RuntimeError", "message": "boom"},
        }
    )
    assert failed["status"] == "failed"

    with pytest.raises(ValueError):
        result_module.process(
            {
                "status": "success",
                "metrics": {"samples": [sample("service_failed", None)]},
                "metadata": {},
            }
        )


def test_registry_discovers_gsm8k() -> None:
    registry = BenchmarkRegistry()
    assert "gsm8k" in registry.list_benchmarks("inference")
    config = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "gsm8k"
        / "gsm8k.yaml"
    )
    request = RunRequest(
        run_id="inference-gsm8k-test",
        module="inference",
        benchmark="gsm8k",
        config=config,
        model_path=None,
        model_name=None,
        output_dir=ROOT / "runs",
    )
    run = registry.resolve(request)
    assert run.benchmark.result_handler.name == "result.py"
    assert run.parameters["eval_mode"] == "gen"
    assert run.parameters["max_tokens"] == 512
    assert run.parameters["dataset_path"] == "data/gsm8k/test.jsonl"



def test_dataset_loader_survives_unicode_line_separators(tmp_path: Path) -> None:
    """Official GSM8K text contains U+2028/U+2029; loaders must not split on them."""
    from luban_meter.benchmark.inference.common.dataset import load_records

    record = {
        "id": "sep-1",
        "question": "part one\u2028part two",
        "answer": "reasoning\u2029more\n#### 3",
    }
    dataset = tmp_path / "sep.jsonl"
    dataset.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = load_records(dataset)
    assert loaded == [record]

