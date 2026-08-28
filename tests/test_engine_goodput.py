"""Tests for vLLM Engine-internal SLO and Goodput semantics."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from luban_meter.core.registry import BenchmarkRegistry

ROOT = Path(__file__).parents[1]
ENGINE_RESULT_HANDLER = (
    ROOT / "src/luban_meter/benchmark/generate/vllm-engine-offline/result.py"
)
ENGINE_BENCHMARK_ENTRY = (
    ROOT / "src/luban_meter/benchmark/generate/vllm-engine-offline/benchmark.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_record(
    *,
    request_index: int,
    scheduled_ts: float,
    first_token_ts: float,
    last_token_ts: float,
    internal_ttft_seconds: float,
    output_tokens: int = 3,
    input_tokens: int = 4,
    request_batch_size: int = 2,
) -> dict:
    return {
        "request_index": request_index,
        "round_index": 0,
        "input_length": input_tokens,
        "output_length": output_tokens,
        "request_batch_size": request_batch_size,
        "actual_prompt_tokens": input_tokens,
        "actual_output_tokens": output_tokens,
        "internal_ttft_seconds": internal_ttft_seconds,
        "scheduled_ts": scheduled_ts,
        "first_token_ts": first_token_ts,
        "last_token_ts": last_token_ts,
    }


def make_round(
    *,
    start: float = 10.0,
    output_tokens: int = 3,
) -> dict:
    if output_tokens == 1:
        requests = [
            make_record(
                request_index=0,
                scheduled_ts=start,
                first_token_ts=start + 0.1,
                last_token_ts=start + 0.1,
                internal_ttft_seconds=0.1,
                output_tokens=1,
            ),
            make_record(
                request_index=1,
                scheduled_ts=start,
                first_token_ts=start + 0.15,
                last_token_ts=start + 0.15,
                internal_ttft_seconds=0.15,
                output_tokens=1,
            ),
        ]
    else:
        requests = [
            make_record(
                request_index=0,
                scheduled_ts=start,
                first_token_ts=start + 0.1,
                last_token_ts=start + 0.3,
                internal_ttft_seconds=0.1,
                output_tokens=output_tokens,
            ),
            make_record(
                request_index=1,
                scheduled_ts=start,
                first_token_ts=start + 0.15,
                last_token_ts=start + 0.4,
                internal_ttft_seconds=0.15,
                output_tokens=output_tokens,
            ),
        ]
    return {"round_index": 0, "requests": requests}


def make_case(*, output_tokens: int = 3, rounds: list[dict] | None = None) -> dict:
    return {
        "input_length": 4,
        "output_length": output_tokens,
        "request_batch_size": 2,
        "rounds": rounds if rounds is not None else [make_round(output_tokens=output_tokens)],
    }


def make_raw_result(case: dict, engine_slo: dict | None = None) -> dict:
    metadata = {"measurement": "vllm_engine_offline"}
    if engine_slo is not None:
        metadata["engine_slo_config"] = engine_slo
    return {
        "environment": {
            "kv_cache": {
                "num_gpu_blocks": 10,
                "block_size": 16,
                "kv_cache_size_tokens": 160,
                "kv_cache_max_concurrency": 2.0,
            }
        },
        "metrics": {"cases": [case]},
        "metadata": metadata,
    }


class EngineSloConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark = load_module(
            "engine_benchmark_slo", ENGINE_BENCHMARK_ENTRY
        )

    def test_absent_config_is_backward_compatible(self) -> None:
        self.assertIsNone(self.benchmark.engine_slo_config({}))
        self.assertIsNone(self.benchmark.engine_slo_config({"engine_slo": None}))

    def test_registry_exposes_offline_benchmark_name(self) -> None:
        benchmarks = BenchmarkRegistry().list_benchmarks("generate")
        self.assertIn("vllm-engine-offline", benchmarks)
        self.assertNotIn("vllm-engine-stage", benchmarks)

    def test_valid_partial_config_is_normalized(self) -> None:
        config = self.benchmark.engine_slo_config(
            {
                "engine_slo": {
                    "internal_ttft_ms": 200,
                    "mean_decode_step_latency_ms": 50.5,
                }
            }
        )
        self.assertEqual(
            config,
            {
                "internal_ttft_ms": 200.0,
                "mean_decode_step_latency_ms": 50.5,
            },
        )

    def test_invalid_configs_are_rejected(self) -> None:
        invalid = (
            {"engine_slo": "invalid"},
            {"engine_slo": {}},
            {"engine_slo": {"internal_ttft_ms": 0}},
            {"engine_slo": {"internal_ttft_ms": float("nan")}},
            {"engine_slo": {"internal_ttft_ms": float("inf")}},
            {"engine_slo": {"unknown_ms": 10}},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.benchmark.engine_slo_config(parameters)

    def test_run_benchmark_preserves_config_in_raw_metadata(self) -> None:
        fake_engine = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=32),
            get_tokenizer=lambda: SimpleNamespace(
                encode=lambda text, add_special_tokens=False: [1, 2]
            ),
        )
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: kwargs
        parameters = {
            "warmup_rounds": 0,
            "rounds": 1,
            "input_lengths": [4],
            "output_lengths": [1],
            "request_batch_sizes": [1],
            "seed_prompt": "test",
            "engine_slo": {"internal_ttft_ms": 200},
        }
        record = make_record(
            request_index=0,
            scheduled_ts=10.0,
            first_token_ts=10.1,
            last_token_ts=10.1,
            internal_ttft_seconds=0.1,
            output_tokens=1,
            request_batch_size=1,
        )

        with (
            patch.dict(sys.modules, {"vllm": fake_vllm}),
            patch.object(self.benchmark, "build_engine", return_value=fake_engine),
            patch.object(self.benchmark, "generate_batch", return_value=[record]),
            patch.object(
                self.benchmark,
                "kv_cache_environment",
                return_value={"num_gpu_blocks": 1},
            ),
        ):
            raw_result = self.benchmark.run_benchmark(
                {"model_name": "test-model"}, parameters
            )

        self.assertEqual(
            raw_result["metadata"]["engine_slo_config"],
            {"internal_ttft_ms": 200.0},
        )
        self.assertEqual(
            raw_result["metadata"]["measurement"], "vllm_engine_offline"
        )

    def test_failure_result_uses_offline_measurement_name(self) -> None:
        failure = self.benchmark.failure_result(RuntimeError("failed"))
        self.assertEqual(
            failure["metadata"]["measurement"], "vllm_engine_offline"
        )


class EngineGoodputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result_module = load_module(
            "engine_result_goodput", ENGINE_RESULT_HANDLER
        )

    def test_no_goodput_when_engine_slo_is_absent(self) -> None:
        result = self.result_module.process(make_raw_result(make_case()))
        self.assertNotIn("engine_goodput", result["metrics"]["cases"][0])

    def test_classifies_requests_with_and_semantics(self) -> None:
        raw_result = make_raw_result(
            make_case(),
            {
                "internal_ttft_ms": 200,
                "prefill_latency_ms": 200,
                "mean_decode_step_latency_ms": 110,
                "engine_execution_latency_ms": 500,
            },
        )
        result = self.result_module.process(raw_result)
        goodput = result["metrics"]["cases"][0]["engine_goodput"]

        self.assertEqual(goodput["status"], "applicable")
        self.assertEqual(goodput["measurement_boundary"], "vllm_engine_internal")
        self.assertEqual(
            goodput["duration_basis"], "sum_of_formal_round_engine_windows"
        )
        self.assertEqual(goodput["engine_active_duration"]["value"], 0.4)
        self.assertEqual(goodput["engine_slo_satisfied_count"]["value"], 1)
        self.assertEqual(goodput["engine_slo_violated_count"]["value"], 1)
        self.assertEqual(goodput["engine_slo_satisfied_rate"]["value"], 0.5)
        self.assertEqual(
            goodput["engine_goodput_request_throughput"]["value"], 2.5
        )
        self.assertEqual(
            goodput["engine_goodput_output_token_throughput"]["value"], 7.5
        )

    def test_uses_sum_of_round_windows_not_timestamp_envelope(self) -> None:
        raw_result = make_raw_result(
            make_case(rounds=[make_round(start=10.0), make_round(start=20.0)]),
            {"engine_execution_latency_ms": 1000},
        )
        result = self.result_module.process(raw_result)
        goodput = result["metrics"]["cases"][0]["engine_goodput"]

        self.assertEqual(goodput["engine_active_duration"]["value"], 0.8)
        self.assertEqual(goodput["evaluated_request_count"]["value"], 4)
        self.assertEqual(
            goodput["engine_goodput_request_throughput"]["value"], 5.0
        )

    def test_decode_only_target_is_not_applicable_for_single_token_case(self) -> None:
        raw_result = make_raw_result(
            make_case(output_tokens=1),
            {"mean_decode_step_latency_ms": 100},
        )
        result = self.result_module.process(raw_result)
        goodput = result["metrics"]["cases"][0]["engine_goodput"]

        self.assertEqual(goodput["status"], "not_applicable")
        self.assertEqual(goodput["applicable_dimensions"], [])
        self.assertEqual(
            goodput["not_applicable_dimensions"],
            ["mean_decode_step_latency_ms"],
        )
        self.assertNotIn("engine_goodput_request_throughput", goodput)

    def test_single_token_case_uses_other_applicable_targets(self) -> None:
        raw_result = make_raw_result(
            make_case(output_tokens=1),
            {
                "internal_ttft_ms": 120,
                "mean_decode_step_latency_ms": 100,
            },
        )
        result = self.result_module.process(raw_result)
        goodput = result["metrics"]["cases"][0]["engine_goodput"]

        self.assertEqual(goodput["status"], "applicable")
        self.assertEqual(goodput["engine_slo_satisfied_count"]["value"], 1)
        self.assertEqual(goodput["engine_slo_violated_count"]["value"], 1)
        self.assertEqual(
            goodput["not_applicable_dimensions"],
            ["mean_decode_step_latency_ms"],
        )

    def test_invalid_raw_engine_slo_is_rejected(self) -> None:
        raw_result = make_raw_result(make_case(), {"unknown_ms": 100})
        with self.assertRaisesRegex(ValueError, "unsupported thresholds"):
            self.result_module.process(raw_result)

    def test_non_finite_raw_metric_is_rejected(self) -> None:
        raw_result = make_raw_result(
            make_case(), {"internal_ttft_ms": 200}
        )
        raw_result["metrics"]["cases"][0]["rounds"][0]["requests"][0][
            "internal_ttft_seconds"
        ] = float("nan")
        with self.assertRaisesRegex(ValueError, "positive number"):
            self.result_module.process(raw_result)


if __name__ == "__main__":
    unittest.main()
