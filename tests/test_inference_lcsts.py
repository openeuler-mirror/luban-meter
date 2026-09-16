"""Tests for the lcsts inference benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from luban_meter.benchmark.inference.common.parsers import (
    general_postprocess,
    lcsts_postprocess,
)
from luban_meter.benchmark.inference.common.prompts import (
    render_lcsts_prompt,
)
from luban_meter.benchmark.inference.lcsts import (
    benchmark as lcsts,
)
from luban_meter.core.models import RunRequest
from luban_meter.core.registry import BenchmarkRegistry

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "src"
    / "luban_meter"
    / "benchmark"
    / "inference"
    / "lcsts"
    / "result.py"
)

MODULE_COUNT = 0


def load_result_module():
    global MODULE_COUNT
    MODULE_COUNT += 1
    spec = importlib.util.spec_from_file_location(
        f"lcsts_result_test_{MODULE_COUNT}",
        RESULT_PATH,
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
                        "id": "s1",
                        "content": "\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u5c0f\u660e\u53bb\u516c\u56ed\u6563\u6b65\u3002",
                        "abst": "\u5c0f\u660e\u53bb\u516c\u56ed",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "s2",
                        "content": "\u4e2d\u56fd\u7ecf\u6d4e\u6301\u7eed\u589e\u957f\uff0cGDP\u8d85\u9884\u671f\u3002",
                        "abst": "\u4e2d\u56fd\u7ecf\u6d4e\u589e\u957f",
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


# ---- Parameter validation ----


def test_validate_parameters(tmp_path: Path) -> None:
    config = lcsts.validate_parameters(
        base_parameters(tmp_path)
    )
    assert config["eval_mode"] == "gen"
    assert config["prompt_format"] == "chat"
    assert config["max_tokens"] == 512
    assert config["few_shot"] == 0
    with pytest.raises(ValueError):
        lcsts.validate_parameters(
            {**base_parameters(tmp_path), "eval_mode": "ppl"}
        )
    with pytest.raises(ValueError):
        lcsts.validate_parameters(
            {
                **base_parameters(tmp_path),
                "prompt_version": "gsm8k-v1",
            }
        )
    with pytest.raises(ValueError):
        lcsts.validate_parameters(
            {**base_parameters(tmp_path), "few_shot": 4}
        )


# ---- Prompt rendering ----


def test_render_lcsts_prompt() -> None:
    sample = {
        "content": "\u4eca\u5929\u5929\u6c14\u5f88\u597d",
        "abst": "\u5929\u6c14\u597d",
    }
    prompt = render_lcsts_prompt(sample)
    assert "\u9605\u8bfb\u4ee5\u4e0b\u6587\u7ae0" in prompt
    assert "\u4eca\u5929\u5929\u6c14\u5f88\u597d" in prompt
    assert prompt.endswith("\u6458\u8981\u5982\u4e0b\uff1a")


def test_render_lcsts_prompt_few_shot() -> None:
    sample = {
        "content": "\u6d4b\u8bd5\u6587\u7ae0",
        "abst": "\u6d4b\u8bd5",
    }
    few_shot = [
        {
            "content": "\u793a\u4f8b\u6587\u7ae0",
            "abst": "\u793a\u4f8b\u6458\u8981",
        }
    ]
    prompt = render_lcsts_prompt(
        sample, few_shot_samples=few_shot
    )
    assert "\u793a\u4f8b\u6587\u7ae0" in prompt
    assert "\u793a\u4f8b\u6458\u8981" in prompt
    assert "\u6d4b\u8bd5\u6587\u7ae0" in prompt


def test_render_lcsts_prompt_empty_content() -> None:
    with pytest.raises(ValueError):
        render_lcsts_prompt({"content": "", "abst": "x"})


# ---- Post-processing ----


def test_lcsts_postprocess() -> None:
    assert lcsts_postprocess("1. \u6458\u8981\u5185\u5bb9") == "\u6458\u8981\u5185\u5bb9"
    assert lcsts_postprocess("- \u6458\u8981") == "\u6458\u8981"
    assert lcsts_postprocess(
        "\u7b2c\u4e00\u884c\n\u7b2c\u4e8c\u884c"
    ) == "\u7b2c\u4e00\u884c"
    assert lcsts_postprocess(
        "\u201c\u6458\u8981\u201d"
    ) == "\u6458\u8981"


def test_general_postprocess() -> None:
    # Chinese punct is removed by [^\w\s] but does NOT
    # trigger truncation (only ASCII \n . , do).
    assert general_postprocess(
        "\u6d4b\u8bd5\u3002\u622a\u65ad"
    ) == "\u6d4b\u8bd5\u622a\u65ad"
    # ASCII period triggers truncation.
    assert general_postprocess(
        "\u6d4b\u8bd5.\u622a\u65ad"
    ) == "\u6d4b\u8bd5"
    # Newline triggers truncation.
    assert general_postprocess(
        "\u6d4b\u8bd5\n\u622a\u65ad"
    ) == "\u6d4b\u8bd5"


# ---- prepare_samples ----


def test_prepare_samples_rejects_bad_content() -> None:
    with pytest.raises(ValueError):
        lcsts.prepare_samples(
            [{"content": "", "abst": "x"}]
        )
    with pytest.raises(ValueError):
        lcsts.prepare_samples(
            [{"content": "x", "abst": ""}]
        )
    prepared = lcsts.prepare_samples(
        [{"content": " \u6587\u7ae0 ", "abst": " \u6458\u8981 "}]
    )
    assert prepared[0]["content"] == "\u6587\u7ae0"
    assert prepared[0]["abst"] == "\u6458\u8981"


# ---- Fake service and end-to-end ----


def test_run_benchmark_end_to_end(tmp_path: Path) -> None:
    """Test end-to-end with a mock client."""
    from unittest.mock import MagicMock, patch
    from luban_meter.benchmark.inference.common.client import OpenAIClient

    mock_client = MagicMock(spec=OpenAIClient)
    mock_client.chat.return_value = {
        "text": "\u6d4b\u8bd5\u6458\u8981",
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_ms": 100.0,
    }

    params = base_parameters(tmp_path)
    with patch(
        "luban_meter.benchmark.inference.lcsts.benchmark.OpenAIClient",
        return_value=mock_client,
    ):
        raw = lcsts.run_benchmark(
            {"model_name": "test-model"}, params
        )

    assert raw["status"] == "success"
    assert raw["schema_version"] == "luban-meter.raw/v1"
    samples = raw["metrics"]["samples"]
    assert len(samples) == 2
    for s in samples:
        assert s["status"] == "success"
        assert s["prediction"] == "\u6d4b\u8bd5\u6458\u8981"
        assert s["input_tokens"] == 10
        assert s["output_tokens"] == 5
    assert raw["metrics"]["counts"]["total"] == 2
    assert raw["metadata"]["dataset"] == "LCSTS"


# ---- Result processing ----


def _make_sample(
    *,
    status: str = "success",
    prediction: str = "\u6d4b\u8bd5\u6458\u8981",
    reference: str = "\u6d4b\u8bd5\u6458\u8981",
) -> dict[str, Any]:
    return {
        "id": "x",
        "status": status,
        "prediction": prediction,
        "reference": reference,
        "input_tokens": 10,
        "output_tokens": 5,
    }


def test_process_success() -> None:
    mod = load_result_module()
    raw = {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metrics": {
            "samples": [
                _make_sample(),
                _make_sample(
                    prediction="\u4e0d\u540c\u6458\u8981",
                    reference="\u53e6\u4e00\u4e2a\u6458\u8981",
                ),
            ],
        },
        "metadata": {"model": "m"},
    }
    final = mod.process(raw)
    assert final["status"] == "success"
    lcsts_metrics = final["metrics"]["task_view"]["lcsts"]
    assert "rouge1" in lcsts_metrics
    assert "rouge2" in lcsts_metrics
    assert "rougeL" in lcsts_metrics
    assert lcsts_metrics["rouge1"]["unit"] == "score"
    assert lcsts_metrics["scored_samples"]["value"] == 2
    assert lcsts_metrics["total_samples"]["value"] == 2


def test_process_identical_texts() -> None:
    """Identical prediction and reference should score ~100."""
    mod = load_result_module()
    raw = {
        "status": "success",
        "metrics": {
            "samples": [
                _make_sample(
                    prediction="\u4eca\u5929\u5929\u6c14\u5f88\u597d",
                    reference="\u4eca\u5929\u5929\u6c14\u5f88\u597d",
                ),
            ],
        },
        "metadata": {},
    }
    final = mod.process(raw)
    lcsts_m = final["metrics"]["task_view"]["lcsts"]
    assert lcsts_m["rouge1"]["value"] == pytest.approx(
        100.0
    )
    assert lcsts_m["rouge2"]["value"] == pytest.approx(
        100.0
    )
    assert lcsts_m["rougeL"]["value"] == pytest.approx(
        100.0
    )


def test_process_partial_failed() -> None:
    mod = load_result_module()
    raw = {
        "status": "success",
        "metrics": {
            "samples": [
                _make_sample(),
                _make_sample(status="service_failed"),
            ],
        },
        "metadata": {},
    }
    final = mod.process(raw)
    assert final["status"] == "partial_failed"
    assert final["error"] is not None


def test_process_all_failed() -> None:
    mod = load_result_module()
    raw = {
        "status": "success",
        "metrics": {
            "samples": [
                _make_sample(status="service_failed"),
            ],
        },
        "metadata": {},
    }
    with pytest.raises(ValueError, match="all samples"):
        mod.process(raw)


def test_process_raw_failed() -> None:
    mod = load_result_module()
    raw = {
        "status": "failed",
        "metrics": {},
        "metadata": {"measurement": "m"},
        "error": {
            "type": "RuntimeError",
            "message": "boom",
        },
    }
    final = mod.process(raw)
    assert final["status"] == "failed"


# ---- Registry discovery ----


def test_registry_discovers_lcsts() -> None:
    registry = BenchmarkRegistry()
    assert "lcsts" in registry.list_benchmarks(
        "inference"
    )
    config = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "lcsts"
        / "lcsts.yaml"
    )
    request = RunRequest(
        run_id="inference-lcsts-test",
        module="inference",
        benchmark="lcsts",
        config=config,
        model_path=None,
        model_name=None,
        output_dir=ROOT / "runs",
    )
    run = registry.resolve(request)
    assert run.benchmark.result_handler.name == "result.py"
    assert run.parameters["eval_mode"] == "gen"
    assert run.parameters["max_tokens"] == 512
    assert run.parameters["dataset_path"] == "data/lcsts/test.jsonl"
