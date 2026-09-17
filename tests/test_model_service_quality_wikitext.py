"""Tests for the wikitext model service quality scenario."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

from luban_meter.benchmarking.model_service_quality.common import metrics
from luban_meter.benchmarking.model_service_quality.wikitext import (
    collect_raw as wikitext,
)
from luban_meter.core.run_contracts import RunRequest
from luban_meter.core.scenario_registry import ScenarioRegistry

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "src"
    / "luban_meter"
    / "benchmarking"
    / "model_service_quality"
    / "wikitext"
    / "calculate_metrics.py"
)

MODULE_COUNT = 0


def load_result_module():
    global MODULE_COUNT
    MODULE_COUNT += 1
    spec = importlib.util.spec_from_file_location(
        f"wikitext_result_test_{MODULE_COUNT}", RESULT_PATH
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
                        "id": "doc1",
                        "text": "Hello world test.",
                        "raw_text": "Hello world test.",
                        "bytes": 17,
                        "words": 3,
                    }
                ),
                json.dumps(
                    {
                        "id": "doc2",
                        "text": "Another paragraph.",
                        "raw_text": "Another paragraph.",
                        "bytes": 18,
                        "words": 2,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    return {
        "dataset_path": str(dataset),
        "max_samples": 2,
        "max_context_length": 512,
    }


# ---- Parameter validation ----


def test_validate_parameters(tmp_path: Path) -> None:
    cfg = wikitext.validate_parameters(base_parameters(tmp_path))
    assert cfg["eval_mode"] == "loss"
    assert cfg["prompt_format"] == "base"
    assert cfg["max_context_length"] == 512
    assert cfg["stride"] is None
    assert cfg["max_samples"] == 2


def test_validate_max_samples_null(
    tmp_path: Path,
) -> None:
    params = base_parameters(tmp_path)
    params["max_samples"] = None
    cfg = wikitext.validate_parameters(params)
    assert cfg["max_samples"] is None


def test_validate_rejects_eval_mode_gen(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        wikitext.validate_parameters(
            {
                **base_parameters(tmp_path),
                "eval_mode": "gen",
            }
        )


def test_validate_rejects_chat_format(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        wikitext.validate_parameters(
            {
                **base_parameters(tmp_path),
                "prompt_format": "chat",
            }
        )


def test_validate_rejects_few_shot(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        wikitext.validate_parameters(
            {
                **base_parameters(tmp_path),
                "few_shot": 4,
            }
        )


def test_validate_rejects_bad_stride(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        wikitext.validate_parameters(
            {
                **base_parameters(tmp_path),
                "stride": 9999,
            }
        )
    with pytest.raises(ValueError):
        wikitext.validate_parameters(
            {
                **base_parameters(tmp_path),
                "stride": 0,
            }
        )


def test_validate_stride_within_context(
    tmp_path: Path,
) -> None:
    cfg = wikitext.validate_parameters(
        {
            **base_parameters(tmp_path),
            "stride": 256,
        }
    )
    assert cfg["stride"] == 256


# ---- Dataset validation ----


def test_validate_records_ok() -> None:
    records = [
        {
            "id": "a",
            "text": "hi",
            "raw_text": "hi",
            "bytes": 2,
            "words": 1,
        },
    ]
    wikitext.validate_records(records)


def test_validate_records_duplicate_id() -> None:
    records = [
        {
            "id": "a",
            "text": "x",
            "raw_text": "x",
            "bytes": 1,
            "words": 1,
        },
        {
            "id": "a",
            "text": "y",
            "raw_text": "y",
            "bytes": 1,
            "words": 1,
        },
    ]
    with pytest.raises(ValueError, match="duplicate"):
        wikitext.validate_records(records)


def test_validate_records_missing_field() -> None:
    with pytest.raises(ValueError, match="text"):
        wikitext.validate_records(
            [
                {
                    "id": "a",
                    "raw_text": "x",
                    "bytes": 1,
                    "words": 1,
                }
            ]
        )


# ---- Metrics helpers ----


def test_bits_per_byte() -> None:
    bpb = metrics.bits_per_byte(-10.0, 20)
    assert bpb == pytest.approx(-(-10.0) / 20 / math.log(2))


def test_bits_per_byte_rejects_zero() -> None:
    with pytest.raises(ValueError):
        metrics.bits_per_byte(-1.0, 0)


# ---- Result processing ----


def _make_sample(
    status: str = "success",
    sum_lp: float = -50.0,
    tok_cnt: int = 20,
    byte_cnt: int = 100,
) -> dict[str, Any]:
    return {
        "id": "x",
        "status": status,
        "sum_logprob": sum_lp,
        "token_count": tok_cnt,
        "bytes": byte_cnt,
        "words": 10,
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
                    sum_lp=-30.0,
                    tok_cnt=15,
                    byte_cnt=80,
                ),
            ],
        },
        "metadata": {"model": "m"},
    }
    final = mod.process(raw)
    assert final["status"] == "success"
    wt = final["metrics"]["task_view"]["wikitext"]
    total_lp = -50.0 + (-30.0)
    total_tok = 20 + 15
    total_byt = 100 + 80
    expected_ml = -total_lp / total_tok
    expected_ppl = math.exp(expected_ml)
    expected_bpb = -total_lp / total_byt / math.log(2)
    assert wt["mean_loss"]["value"] == pytest.approx(expected_ml)
    assert wt["perplexity"]["value"] == pytest.approx(expected_ppl)
    assert wt["bits_per_byte"]["value"] == pytest.approx(expected_bpb)
    assert wt["count"]["value"] == total_tok
    assert wt["count_bytes"]["value"] == total_byt


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


def test_registry_discovers_wikitext() -> None:
    registry = ScenarioRegistry()
    assert "wikitext" in registry.list_scenarios("model_service_quality")
    config = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmarking"
        / "model_service_quality"
        / "wikitext"
        / "wikitext.yaml"
    )
    request = RunRequest(
        run_id="model_service_quality-wikitext-test",
        category_name="model_service_quality",
        scenario_name="wikitext",
        config_path=config,
        model_path=None,
        model_name=None,
        output_dir=ROOT / "runs",
    )
    run = registry.resolve(request)
    assert (
        run.scenario_definition.processor_path.name == "calculate_metrics.py"
    )
    assert run.parameters["eval_mode"] == "loss"
    assert run.parameters["max_context_length"] == 2048
    assert run.parameters["stride"] is None
