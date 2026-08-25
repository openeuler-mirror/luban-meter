"""Tests for SLO config, circuit breaker, and Goodput computation.

Covers the M1 milestone:
- slo config validation in benchmark.py
- circuit breaker triggering based on Case P99 E2EL
- Goodput computation in result.py
- backward compatibility when slo config is absent
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SERVING_RESULT_HANDLER = (
    ROOT / "src/luban_meter/benchmark/generate/serving-online/result.py"
)
SERVING_BENCHMARK_ENTRY = (
    ROOT / "src/luban_meter/benchmark/generate/serving-online/benchmark.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_request_record(
    *,
    status: str = "success",
    ttft_ms: float = 100.0,
    e2el_ms: float = 500.0,
    output_tokens: int = 3,
    input_tokens: int = 10,
    itl_samples_ms: list[float] | None = None,
    start_offset_ms: float = 0.0,
    dispatch_delay_ms: float = 0.0,
    duration_ms: float | None = None,
) -> dict:
    if itl_samples_ms is None:
        itl_samples_ms = [100.0, 100.0]
    if duration_ms is None:
        duration_ms = e2el_ms
    return {
        "status": status,
        "start_offset_ms": start_offset_ms,
        "dispatch_delay_ms": dispatch_delay_ms,
        "duration_ms": duration_ms,
        "ttft_ms": ttft_ms,
        "e2el_ms": e2el_ms,
        "itl_samples_ms": itl_samples_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def make_raw_case(
    *,
    input_length: int = 10,
    output_length: int = 3,
    request_rate: float = 2.0,
    duration_seconds: float = 10.0,
    max_concurrency: int = 8,
    peak_concurrency: int = 2,
    requests: list[dict] | None = None,
) -> dict:
    if requests is None:
        requests = []
    return {
        "input_length": input_length,
        "output_length": output_length,
        "request_rate": request_rate,
        "benchmark_duration_seconds": duration_seconds,
        "maximum_request_concurrency": max_concurrency,
        "peak_concurrent_requests": peak_concurrency,
        "requests": requests,
    }


class SloConfigTest(unittest.TestCase):
    """Verify slo config validation in benchmark.py."""

    def setUp(self) -> None:
        self.benchmark = load_module(
            "serving_benchmark_slo", SERVING_BENCHMARK_ENTRY
        )

    def test_absent_slo_returns_none(self) -> None:
        self.assertIsNone(self.benchmark.slo_config({}))

    def test_none_slo_returns_none(self) -> None:
        self.assertIsNone(self.benchmark.slo_config({"slo": None}))

    def test_invalid_slo_type_raises(self) -> None:
        with self.assertRaisesRegex(TypeError, "slo must be an object"):
            self.benchmark.slo_config({"slo": "invalid"})

    def test_empty_slo_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one threshold"):
            self.benchmark.slo_config({"slo": {}})

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a positive number"):
            self.benchmark.slo_config({"slo": {"p99_ms": -1}})

    def test_valid_slo_returns_config(self) -> None:
        slo = self.benchmark.slo_config(
            {"slo": {"p99_ms": 2000, "ttft_ms": 500, "tpot_ms": 50, "e2el_ms": 8000}}
        )
        self.assertEqual(
            slo,
            {"p99_ms": 2000.0, "ttft_ms": 500.0, "tpot_ms": 50.0, "e2el_ms": 8000.0},
        )

    def test_partial_slo_returns_config(self) -> None:
        slo = self.benchmark.slo_config({"slo": {"p99_ms": 2000}})
        self.assertEqual(slo, {"p99_ms": 2000.0})


class CaseP99E2elTest(unittest.TestCase):
    """Verify P99 E2EL computation from a Case."""

    def setUp(self) -> None:
        self.benchmark = load_module(
            "serving_benchmark_p99", SERVING_BENCHMARK_ENTRY
        )

    def test_returns_none_for_fewer_than_10_samples(self) -> None:
        case = make_raw_case(
            requests=[make_request_record() for _ in range(9)]
        )
        self.assertIsNone(self.benchmark.case_p99_e2el_ms(case))

    def test_returns_none_for_no_successful_requests(self) -> None:
        case = make_raw_case(
            requests=[make_request_record(status="failed") for _ in range(10)]
        )
        self.assertIsNone(self.benchmark.case_p99_e2el_ms(case))

    def test_computes_p99_from_successful_e2el(self) -> None:
        case = make_raw_case(
            requests=[
                make_request_record(e2el_ms=float(i) * 10)
                for i in range(1, 11)
            ]
        )
        p99 = self.benchmark.case_p99_e2el_ms(case)
        self.assertIsNotNone(p99)
        self.assertGreater(p99, 0)


class CircuitBreakerTest(unittest.TestCase):
    """Verify circuit breaker triggering in benchmark.run_benchmark."""

    def setUp(self) -> None:
        self.benchmark = load_module(
            "serving_benchmark_cb", SERVING_BENCHMARK_ENTRY
        )

    def _patch_run_case(self, p99_values: list[float]):
        """Patch run_case to return cases with controlled P99 values.

        Each entry in p99_values produces a Case whose successful requests
        have E2EL samples such that percentile(e2el, 0.99) == p99_values[i].
        """
        call_index = [0]

        def fake_run_case(**kwargs):
            idx = call_index[0]
            call_index[0] += 1
            p99 = p99_values[idx] if idx < len(p99_values) else p99_values[-1]
            requests = [
                make_request_record(e2el_ms=p99, ttft_ms=50.0)
                for _ in range(10)
            ]
            return {
                "input_length": kwargs["input_length"],
                "output_length": kwargs["output_length"],
                "request_rate": kwargs["request_rate"],
                "benchmark_duration_seconds": 1.0,
                "maximum_request_concurrency": kwargs["max_concurrency"],
                "peak_concurrent_requests": 1,
                "requests": requests,
            }

        return fake_run_case

    def test_no_circuit_breaker_without_p99_threshold(self) -> None:
        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "warmup": 0,
            "rounds": 10,
            "max_concurrency": 8,
            "input_lengths": [4],
            "output_lengths": [2],
            "request_rates": [1.0, 2.0],
            "seed_prompt": "test",
            "request_timeout": 5,
        }
        request = {"model_name": "test-model"}

        with patch.object(self.benchmark, "run_case", side_effect=self._patch_run_case([100, 200])):
            with patch.object(self.benchmark, "tokenize_seed_prompt", return_value=([1, 2], 32)):
                with patch.object(self.benchmark, "discover_model", return_value="test-model"):
                    raw_result = self.benchmark.run_benchmark(request, parameters)

        self.assertNotIn("circuit_breaker", raw_result["metadata"])
        self.assertNotIn("slo_config", raw_result["metadata"])
        self.assertEqual(len(raw_result["metrics"]["cases"]), 2)

    def test_circuit_breaker_triggers_on_p99_exceeded(self) -> None:
        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "warmup": 0,
            "rounds": 10,
            "max_concurrency": 8,
            "input_lengths": [4],
            "output_lengths": [2],
            "request_rates": [1.0, 2.0, 4.0],
            "seed_prompt": "test",
            "request_timeout": 5,
            "slo": {"p99_ms": 150},
        }
        request = {"model_name": "test-model"}

        with patch.object(self.benchmark, "run_case", side_effect=self._patch_run_case([100, 200, 300])):
            with patch.object(self.benchmark, "tokenize_seed_prompt", return_value=([1, 2], 32)):
                with patch.object(self.benchmark, "discover_model", return_value="test-model"):
                    raw_result = self.benchmark.run_benchmark(request, parameters)

        self.assertIn("circuit_breaker", raw_result["metadata"])
        cb = raw_result["metadata"]["circuit_breaker"]
        self.assertTrue(cb["triggered"])
        self.assertEqual(cb["threshold_p99_ms"], 150.0)
        self.assertGreater(cb["actual_p99_ms"], 150)
        self.assertEqual(cb["triggered_at_case"]["request_rate"], 2.0)
        self.assertEqual(cb["remaining_cases_skipped"], 1)
        self.assertEqual(len(raw_result["metrics"]["cases"]), 2)

    def test_circuit_breaker_not_triggered_below_threshold(self) -> None:
        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "warmup": 0,
            "rounds": 10,
            "max_concurrency": 8,
            "input_lengths": [4],
            "output_lengths": [2],
            "request_rates": [1.0, 2.0],
            "seed_prompt": "test",
            "request_timeout": 5,
            "slo": {"p99_ms": 1000},
        }
        request = {"model_name": "test-model"}

        with patch.object(self.benchmark, "run_case", side_effect=self._patch_run_case([100, 200])):
            with patch.object(self.benchmark, "tokenize_seed_prompt", return_value=([1, 2], 32)):
                with patch.object(self.benchmark, "discover_model", return_value="test-model"):
                    raw_result = self.benchmark.run_benchmark(request, parameters)

        self.assertNotIn("circuit_breaker", raw_result["metadata"])
        self.assertEqual(len(raw_result["metrics"]["cases"]), 2)

    def test_circuit_breaker_not_triggered_for_few_samples(self) -> None:
        """When successful samples < 10, P99 is not computed, breaker skips."""
        def fake_run_case(**kwargs):
            return {
                "input_length": kwargs["input_length"],
                "output_length": kwargs["output_length"],
                "request_rate": kwargs["request_rate"],
                "benchmark_duration_seconds": 1.0,
                "maximum_request_concurrency": kwargs["max_concurrency"],
                "peak_concurrent_requests": 1,
                "requests": [make_request_record(e2el_ms=10000) for _ in range(5)],
            }

        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "warmup": 0,
            "rounds": 5,
            "max_concurrency": 8,
            "input_lengths": [4],
            "output_lengths": [2],
            "request_rates": [1.0, 2.0],
            "seed_prompt": "test",
            "request_timeout": 5,
            "slo": {"p99_ms": 100},
        }
        request = {"model_name": "test-model"}

        with patch.object(self.benchmark, "run_case", side_effect=fake_run_case):
            with patch.object(self.benchmark, "tokenize_seed_prompt", return_value=([1, 2], 32)):
                with patch.object(self.benchmark, "discover_model", return_value="test-model"):
                    raw_result = self.benchmark.run_benchmark(request, parameters)

        self.assertNotIn("circuit_breaker", raw_result["metadata"])
        self.assertEqual(len(raw_result["metrics"]["cases"]), 2)

    def test_circuit_breaker_skipped_cases_recorded(self) -> None:
        """Verify skipped_cases list contains every remaining case after trigger."""
        def fake_run_case(**kwargs):
            p99_map = {(4, 2, 1.0): 100, (4, 2, 2.0): 200, (4, 2, 4.0): 300}
            key = (kwargs["input_length"], kwargs["output_length"], kwargs["request_rate"])
            p99 = p99_map.get(key, 200)
            requests = [
                make_request_record(e2el_ms=p99, ttft_ms=50.0)
                for _ in range(10)
            ]
            return {
                "input_length": kwargs["input_length"],
                "output_length": kwargs["output_length"],
                "request_rate": kwargs["request_rate"],
                "benchmark_duration_seconds": 1.0,
                "maximum_request_concurrency": kwargs["max_concurrency"],
                "peak_concurrent_requests": 1,
                "requests": requests,
            }

        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "warmup": 0,
            "rounds": 10,
            "max_concurrency": 8,
            "input_lengths": [4],
            "output_lengths": [2],
            "request_rates": [1.0, 2.0, 4.0],
            "seed_prompt": "test",
            "request_timeout": 5,
            "slo": {"p99_ms": 150},
        }
        request = {"model_name": "test-model"}

        with patch.object(self.benchmark, "run_case", side_effect=fake_run_case):
            with patch.object(self.benchmark, "tokenize_seed_prompt", return_value=([1, 2], 32)):
                with patch.object(self.benchmark, "discover_model", return_value="test-model"):
                    raw_result = self.benchmark.run_benchmark(request, parameters)

        self.assertIn("skipped_cases", raw_result["metadata"])
        skipped = raw_result["metadata"]["skipped_cases"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["request_rate"], 4.0)
        self.assertEqual(skipped[0]["skipped_reason"], "circuit_breaker_triggered")
        self.assertEqual(
            raw_result["metadata"]["circuit_breaker"]["remaining_cases_skipped"], 1
        )


class GoodputComputationTest(unittest.TestCase):
    """Verify Goodput computation in result.py."""

    def setUp(self) -> None:
        self.result_module = load_module(
            "serving_result_goodput", SERVING_RESULT_HANDLER
        )

    def _run_process(self, raw_result: dict) -> dict:
        return self.result_module.process(raw_result)

    def test_no_goodput_field_when_slo_absent(self) -> None:
        raw_case = make_raw_case(
            requests=[
                make_request_record(ttft_ms=100, e2el_ms=500, output_tokens=3)
                for _ in range(2)
            ],
        )
        raw_result = {"metrics": {"cases": [raw_case]}, "metadata": {}}
        result = self._run_process(raw_result)
        service_view = result["metrics"]["cases"][0]["service_view"]
        self.assertNotIn("goodput", service_view)

    def test_p99_only_slo_does_not_emit_goodput(self) -> None:
        """Only p99_ms configured: no Goodput dimensions, no goodput output."""
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=500, output_tokens=3, input_tokens=10
                )
                for _ in range(10)
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {"slo_config": {"p99_ms": 2000}},
        }
        result = self._run_process(raw_result)
        service_view = result["metrics"]["cases"][0]["service_view"]
        self.assertNotIn("goodput", service_view)

    def test_tpot_only_single_token_not_applicable(self) -> None:
        """Only tpot_ms configured and all output_tokens==1: status=not_applicable."""
        raw_case = make_raw_case(
            input_length=10,
            output_length=1,
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=200, output_tokens=1, input_tokens=10
                )
                for _ in range(10)
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {"slo_config": {"tpot_ms": 50}},
        }
        result = self._run_process(raw_result)
        service_view = result["metrics"]["cases"][0]["service_view"]
        self.assertIn("goodput", service_view)
        goodput = service_view["goodput"]
        self.assertEqual(goodput["status"], "not_applicable")
        self.assertIn("tpot_ms", goodput["not_applicable_dimensions"])

    def test_all_satisfied_goodput_equals_request_throughput(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=300, output_tokens=3, input_tokens=10
                )
                for _ in range(10)
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "tpot_ms": 100, "e2el_ms": 1000},
            },
        }
        result = self._run_process(raw_result)
        service_view = result["metrics"]["cases"][0]["service_view"]
        goodput = service_view["goodput"]
        self.assertEqual(goodput["slo_satisfied_count"]["value"], 10)
        self.assertEqual(goodput["slo_violated_count"]["value"], 0)
        self.assertEqual(goodput["slo_satisfied_rate"]["value"], 1.0)
        self.assertEqual(
            goodput["goodput_request_throughput"]["value"],
            service_view["request_throughput"]["value"],
        )

    def test_all_violated_goodput_is_zero(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=600, e2el_ms=9000, output_tokens=3, input_tokens=10
                )
                for _ in range(10)
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "e2el_ms": 1000},
            },
        }
        result = self._run_process(raw_result)
        goodput = result["metrics"]["cases"][0]["service_view"]["goodput"]
        self.assertEqual(goodput["slo_satisfied_count"]["value"], 0)
        self.assertEqual(goodput["slo_violated_count"]["value"], 10)
        self.assertEqual(goodput["slo_satisfied_rate"]["value"], 0.0)
        self.assertEqual(goodput["goodput_request_throughput"]["value"], 0.0)

    def test_partial_satisfaction(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=500, output_tokens=3, input_tokens=10
                ),
                make_request_record(
                    ttft_ms=600, e2el_ms=9000, output_tokens=3, input_tokens=10
                ),
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "e2el_ms": 1000},
            },
        }
        result = self._run_process(raw_result)
        goodput = result["metrics"]["cases"][0]["service_view"]["goodput"]
        self.assertEqual(goodput["slo_satisfied_count"]["value"], 1)
        self.assertEqual(goodput["slo_violated_count"]["value"], 1)
        self.assertEqual(goodput["slo_satisfied_rate"]["value"], 0.5)

    def test_tpot_skipped_for_single_token_output(self) -> None:
        """output_tokens=1: TPOT marked not_applicable, TTFT/E2EL check normally."""
        raw_case = make_raw_case(
            input_length=10,
            output_length=1,
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=200, output_tokens=1, input_tokens=10
                )
                for _ in range(10)
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "tpot_ms": 10, "e2el_ms": 1000},
            },
        }
        result = self._run_process(raw_result)
        goodput = result["metrics"]["cases"][0]["service_view"]["goodput"]
        self.assertEqual(goodput["slo_satisfied_count"]["value"], 10)
        self.assertEqual(goodput["slo_violated_count"]["value"], 0)
        self.assertEqual(goodput["status"], "applicable")
        self.assertIn("tpot_ms", goodput["not_applicable_dimensions"])
        self.assertNotIn("tpot_ms", goodput["applicable_dimensions"])
        self.assertIn("ttft_ms", goodput["applicable_dimensions"])
        self.assertIn("e2el_ms", goodput["applicable_dimensions"])

    def test_tpot_violation(self) -> None:
        """Even when TTFT and E2EL are within SLO, TPOT violation fails the request."""
        # decode_duration = e2el - ttft = 1000 - 100 = 900
        # tpot = 900 / (3 - 1) = 450 > 50 → violation
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=1000, output_tokens=3, input_tokens=10
                )
                for _ in range(2)
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "tpot_ms": 50, "e2el_ms": 2000},
            },
        }
        result = self._run_process(raw_result)
        goodput = result["metrics"]["cases"][0]["service_view"]["goodput"]
        self.assertEqual(goodput["slo_violated_count"]["value"], 2)

    def test_failed_requests_not_in_goodput(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=500, output_tokens=3, input_tokens=10
                ),
                make_request_record(status="failed"),
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "e2el_ms": 1000},
            },
        }
        result = self._run_process(raw_result)
        goodput = result["metrics"]["cases"][0]["service_view"]["goodput"]
        self.assertEqual(goodput["slo_satisfied_count"]["value"], 1)
        self.assertEqual(goodput["slo_violated_count"]["value"], 0)

    def test_multiple_cases_independent_goodput(self) -> None:
        raw_case_1 = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=500, output_tokens=3, input_tokens=10
                )
                for _ in range(10)
            ],
        )
        raw_case_2 = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=600, e2el_ms=9000, output_tokens=3, input_tokens=10
                )
                for _ in range(10)
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case_1, raw_case_2]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "e2el_ms": 1000},
            },
        }
        result = self._run_process(raw_result)
        goodput_1 = result["metrics"]["cases"][0]["service_view"]["goodput"]
        goodput_2 = result["metrics"]["cases"][1]["service_view"]["goodput"]
        self.assertEqual(goodput_1["slo_satisfied_count"]["value"], 10)
        self.assertEqual(goodput_2["slo_violated_count"]["value"], 10)

    def test_backward_compatible_with_old_raw_result(self) -> None:
        """Old raw_result.json without slo_config metadata works normally."""
        raw_case = make_raw_case(
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=500, output_tokens=3, input_tokens=10
                )
                for _ in range(2)
            ],
        )
        raw_result = {"metrics": {"cases": [raw_case]}}
        result = self._run_process(raw_result)
        service_view = result["metrics"]["cases"][0]["service_view"]
        self.assertNotIn("goodput", service_view)
        self.assertIn("request_throughput", service_view)

    def test_partial_slo_config(self) -> None:
        """Only ttft_ms configured: only TTFT dimension is checked."""
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100, e2el_ms=500, output_tokens=3, input_tokens=10
                ),
                make_request_record(
                    ttft_ms=600, e2el_ms=700, output_tokens=3, input_tokens=10
                ),
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200},
            },
        }
        result = self._run_process(raw_result)
        goodput = result["metrics"]["cases"][0]["service_view"]["goodput"]
        self.assertEqual(goodput["slo_satisfied_count"]["value"], 1)
        self.assertEqual(goodput["slo_violated_count"]["value"], 1)


class GoodputThroughputValueTest(unittest.TestCase):
    """Verify Goodput throughput values are correctly computed."""

    def setUp(self) -> None:
        self.result_module = load_module(
            "serving_result_goodput_values", SERVING_RESULT_HANDLER
        )

    def test_goodput_output_token_throughput(self) -> None:
        """Verify goodput_output_token_throughput uses only SLO-satisfied tokens."""
        raw_case = make_raw_case(
            input_length=10,
            output_length=3,
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100,
                    e2el_ms=500,
                    output_tokens=3,
                    input_tokens=10,
                ),
                make_request_record(
                    ttft_ms=600,
                    e2el_ms=9000,
                    output_tokens=3,
                    input_tokens=10,
                ),
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {
                "slo_config": {"ttft_ms": 200, "e2el_ms": 1000},
            },
        }
        result = self.result_module.process(raw_result)
        service_view = result["metrics"]["cases"][0]["service_view"]
        goodput = service_view["goodput"]

        # Only 1 satisfied request with 3 output tokens over 10s = 0.3 token/s
        self.assertEqual(goodput["goodput_request_throughput"]["value"], 0.1)
        self.assertEqual(goodput["goodput_output_token_throughput"]["value"], 0.3)

        # service_view output_token_throughput uses ALL successful tokens
        # 2 successful × 3 tokens / 10s = 0.6 token/s
        self.assertEqual(service_view["output_token_throughput"]["value"], 0.6)


if __name__ == "__main__":
    unittest.main()
