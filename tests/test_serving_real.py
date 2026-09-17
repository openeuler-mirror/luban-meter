"""Tests for the online_serving benchmark module (dataset mode).

Covers:
- ShareGPT and JSONL dataset loading
- Arrival process scheduling (constant, poisson, gamma)
- Result processing with variable input/output lengths
- SLO / Goodput computation with variable lengths
- Circuit breaker triggering
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SERVING_REAL_BENCHMARK = ROOT / (
    "src/luban_meter/benchmarking/generation_performance"
    "/online_serving/collect_raw.py"
)
SERVING_REAL_RESULT = ROOT / (
    "src/luban_meter/benchmarking/generation_performance"
    "/online_serving/calculate_metrics.py"
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
        "last_output_latency_ms": ttft_ms if output_tokens == 1 else e2el_ms,
        "e2el_ms": e2el_ms,
        "itl_samples_ms": itl_samples_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def make_raw_case(
    *,
    request_rate: float = 2.0,
    arrival_process: str = "gamma",
    burstiness: float | None = 1.0,
    num_prompts: int = 10,
    max_tokens: int = 2048,
    duration_seconds: float = 10.0,
    max_concurrency: int = 8,
    peak_concurrency: int = 2,
    requests: list[dict] | None = None,
) -> dict:
    if requests is None:
        requests = []
    case = {
        "request_rate": request_rate,
        "arrival_process": arrival_process,
        "num_prompts": num_prompts,
        "max_tokens": max_tokens,
        "benchmark_duration_seconds": duration_seconds,
        "maximum_request_concurrency": max_concurrency,
        "peak_concurrent_requests": peak_concurrency,
        "requests": requests,
    }
    if burstiness is not None:
        case["burstiness"] = burstiness
    return case


class ShareGPTLoadingTest(unittest.TestCase):
    """Verify ShareGPT and JSONL dataset loading."""

    def setUp(self) -> None:
        self.collector_module = load_module(
            "serving_real_bench_ds", SERVING_REAL_BENCHMARK
        )

    def _make_sharegpt_file(self, conversations: list) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(conversations, tmp)
        tmp.close()
        return tmp.name

    def _make_jsonl_file(self, records: list) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp.close()
        return tmp.name

    def test_loads_first_human_message(self) -> None:
        path = self._make_sharegpt_file(
            [
                {
                    "id": "1",
                    "conversations": [
                        {"from": "human", "value": "What is Python?"},
                        {"from": "gpt", "value": "Python is a language."},
                    ],
                },
                {
                    "id": "2",
                    "conversations": [
                        {"from": "human", "value": "Hello there!"},
                        {"from": "gpt", "value": "Hi! How can I help?"},
                    ],
                },
            ]
        )
        prompts = self.collector_module.data_loader.load_prompts(
            path, num_prompts=10, dataset_format="sharegpt"
        )
        self.assertEqual(len(prompts), 2)
        self.assertIn(prompts[0], ("What is Python?", "Hello there!"))

    def test_skips_conversations_without_human(self) -> None:
        path = self._make_sharegpt_file(
            [
                {
                    "id": "1",
                    "conversations": [
                        {"from": "system", "value": "You are helpful."},
                        {"from": "human", "value": "Hi"},
                        {"from": "gpt", "value": "Hello!"},
                    ],
                },
                {
                    "id": "2",
                    "conversations": [
                        {"from": "gpt", "value": "No human here."},
                    ],
                },
            ]
        )
        prompts = self.collector_module.data_loader.load_prompts(
            path, num_prompts=10, dataset_format="sharegpt"
        )
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0], "Hi")

    def test_respects_num_prompts_limit(self) -> None:
        conversations = [
            {
                "id": str(i),
                "conversations": [
                    {"from": "human", "value": f"Question {i}"},
                    {"from": "gpt", "value": f"Answer {i}"},
                ],
            }
            for i in range(50)
        ]
        path = self._make_sharegpt_file(conversations)
        prompts = self.collector_module.data_loader.load_prompts(
            path, num_prompts=5, dataset_format="sharegpt"
        )
        self.assertEqual(len(prompts), 5)

    def test_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.collector_module.data_loader.load_prompts(
                "/nonexistent/path.json", 10, dataset_format="sharegpt"
            )

    def test_raises_on_no_valid_prompts(self) -> None:
        path = self._make_sharegpt_file(
            [
                {
                    "id": "1",
                    "conversations": [{"from": "gpt", "value": "No human."}],
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "no valid prompts"):
            self.collector_module.data_loader.load_prompts(
                path, 10, dataset_format="sharegpt"
            )

    def test_loads_jsonl_prompts(self) -> None:
        path = self._make_jsonl_file(
            [
                {"prompt": "What is the capital of India?"},
                {"prompt": "Explain quantum computing."},
                {"other": "not a prompt"},
            ]
        )
        prompts = self.collector_module.data_loader.load_prompts(
            path, num_prompts=10, dataset_format="jsonl"
        )
        self.assertEqual(len(prompts), 2)
        self.assertIn("What is the capital of India?", prompts)

    def test_jsonl_custom_prompt_field(self) -> None:
        path = self._make_jsonl_file(
            [
                {"question": "What is 2+2?"},
                {"question": "Define AI."},
            ]
        )
        prompts = self.collector_module.data_loader.load_prompts(
            path,
            num_prompts=10,
            dataset_format="jsonl",
            prompt_field="question",
        )
        self.assertEqual(len(prompts), 2)
        self.assertIn("What is 2+2?", prompts)

    def test_jsonl_raises_on_missing_field(self) -> None:
        path = self._make_jsonl_file(
            [
                {"other": "no prompt field"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "no valid prompts"):
            self.collector_module.data_loader.load_prompts(
                path, 10, dataset_format="jsonl"
            )

    def test_unsupported_format_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dataset_format"):
            self.collector_module.data_loader.load_prompts(
                "/tmp/x.json", 10, dataset_format="unknown"
            )


class ArrivalSchedulingTest(unittest.TestCase):
    """Verify arrival process scheduling."""

    def setUp(self) -> None:
        self.collector_module = load_module(
            "serving_real_bench_arr", SERVING_REAL_BENCHMARK
        )

    def test_constant_spacing(self) -> None:
        offsets = self.collector_module.schedule_arrival_times(
            arrival_process="constant",
            request_rate=10.0,
            num_requests=5,
            burstiness=1.0,
            seed=0,
        )
        self.assertEqual(len(offsets), 5)
        self.assertEqual(offsets[0], 0.0)
        for i in range(1, 5):
            self.assertAlmostEqual(offsets[i] - offsets[i - 1], 0.1, places=6)

    def test_poisson_spacing_positive_and_increasing(self) -> None:
        offsets = self.collector_module.schedule_arrival_times(
            arrival_process="poisson",
            request_rate=5.0,
            num_requests=20,
            burstiness=1.0,
            seed=42,
        )
        self.assertEqual(len(offsets), 20)
        self.assertTrue(all(o >= 0 for o in offsets))
        for i in range(1, len(offsets)):
            self.assertGreater(offsets[i], offsets[i - 1])

    def test_gamma_spacing_positive_and_increasing(self) -> None:
        offsets = self.collector_module.schedule_arrival_times(
            arrival_process="gamma",
            request_rate=5.0,
            num_requests=20,
            burstiness=0.5,
            seed=42,
        )
        self.assertEqual(len(offsets), 20)
        self.assertTrue(all(o >= 0 for o in offsets))
        for i in range(1, len(offsets)):
            self.assertGreater(offsets[i], offsets[i - 1])

    def test_invalid_arrival_process_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "arrival_process"):
            self.collector_module.schedule_arrival_times(
                arrival_process="invalid",
                request_rate=1.0,
                num_requests=5,
                burstiness=1.0,
                seed=0,
            )


class ResultProcessingTest(unittest.TestCase):
    """Verify result processing with variable token lengths."""

    def setUp(self) -> None:
        self.result_module = load_module(
            "serving_real_result", SERVING_REAL_RESULT
        )

    def test_variable_lengths_summarized(self) -> None:
        """Variable lengths summarized in distribution."""
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=50,
                    e2el_ms=300,
                    input_tokens=100,
                    output_tokens=10,
                ),
                make_request_record(
                    ttft_ms=80,
                    e2el_ms=600,
                    input_tokens=500,
                    output_tokens=50,
                ),
                make_request_record(
                    ttft_ms=120,
                    e2el_ms=1200,
                    input_tokens=2000,
                    output_tokens=200,
                ),
            ],
        )
        raw_result = {"metrics": {"cases": [raw_case]}, "metadata": {}}
        result = self.result_module.process(raw_result)

        case = result["metrics"]["cases"][0]
        rv = case["request_view"]
        # Input tokens distribution
        self.assertEqual(rv["input_tokens"]["count"], 3)
        self.assertEqual(rv["input_tokens"]["min"], 100.0)
        self.assertEqual(rv["input_tokens"]["max"], 2000.0)
        # Output tokens distribution
        self.assertEqual(rv["output_tokens"]["count"], 3)
        self.assertEqual(rv["output_tokens"]["min"], 10.0)
        self.assertEqual(rv["output_tokens"]["max"], 200.0)
        # TTFT distribution
        self.assertEqual(rv["ttft"]["count"], 3)
        self.assertEqual(rv["ttft"]["min"], 50.0)
        self.assertEqual(rv["ttft"]["max"], 120.0)

    def test_case_metadata_preserved(self) -> None:
        raw_case = make_raw_case(
            request_rate=4.0,
            arrival_process="poisson",
            burstiness=None,
            num_prompts=50,
            max_tokens=1024,
            duration_seconds=5.0,
            requests=[
                make_request_record(input_tokens=10, output_tokens=5),
                make_request_record(input_tokens=20, output_tokens=8),
            ],
        )
        raw_result = {"metrics": {"cases": [raw_case]}, "metadata": {}}
        result = self.result_module.process(raw_result)

        case = result["metrics"]["cases"][0]
        self.assertEqual(case["request_rate"], 4.0)
        self.assertEqual(case["arrival_process"], "poisson")
        self.assertEqual(case["num_prompts"], 50)
        self.assertEqual(case["max_tokens"], 1024)

    def test_failed_requests_excluded_from_token_stats(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    input_tokens=100, output_tokens=10, ttft_ms=50, e2el_ms=300
                ),
                make_request_record(status="failed"),
            ],
        )
        raw_result = {"metrics": {"cases": [raw_case]}, "metadata": {}}
        result = self.result_module.process(raw_result)

        case = result["metrics"]["cases"][0]
        rv = case["request_view"]
        self.assertEqual(rv["input_tokens"]["count"], 1)
        self.assertEqual(rv["output_tokens"]["count"], 1)
        self.assertEqual(case["request_outcome"], "partial_failed")

    def test_service_view_aggregates_correctly(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    input_tokens=100, output_tokens=10, ttft_ms=50, e2el_ms=300
                ),
                make_request_record(
                    input_tokens=200, output_tokens=20, ttft_ms=80, e2el_ms=600
                ),
            ],
        )
        raw_result = {"metrics": {"cases": [raw_case]}, "metadata": {}}
        result = self.result_module.process(raw_result)

        sv = result["metrics"]["cases"][0]["service_view"]
        self.assertEqual(sv["total_requests"]["value"], 2)
        self.assertEqual(sv["successful_requests"]["value"], 2)
        self.assertEqual(sv["total_input_tokens"]["value"], 300)
        self.assertEqual(sv["total_output_tokens"]["value"], 30)
        self.assertEqual(sv["request_throughput"]["value"], 0.2)

    def test_goodput_with_variable_lengths(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(
                    ttft_ms=100,
                    e2el_ms=300,
                    input_tokens=10,
                    output_tokens=3,
                ),
                make_request_record(
                    ttft_ms=600,
                    e2el_ms=9000,
                    input_tokens=500,
                    output_tokens=100,
                ),
            ],
        )
        raw_result = {
            "metrics": {"cases": [raw_case]},
            "metadata": {"slo_config": {"ttft_ms": 200, "e2el_ms": 1000}},
        }
        result = self.result_module.process(raw_result)

        goodput = result["metrics"]["cases"][0]["service_view"]["goodput"]
        self.assertEqual(goodput["slo_satisfied_count"]["value"], 1)
        self.assertEqual(goodput["slo_violated_count"]["value"], 1)
        self.assertEqual(goodput["slo_satisfied_rate"]["value"], 0.5)

    def test_no_goodput_without_slo(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[make_request_record()],
        )
        raw_result = {"metrics": {"cases": [raw_case]}, "metadata": {}}
        result = self.result_module.process(raw_result)
        self.assertNotIn(
            "goodput",
            result["metrics"]["cases"][0]["service_view"],
        )

    def test_overall_status_reflects_failures(self) -> None:
        raw_case = make_raw_case(
            duration_seconds=10.0,
            requests=[
                make_request_record(),
                make_request_record(status="failed"),
            ],
        )
        raw_result = {"metrics": {"cases": [raw_case]}, "metadata": {}}
        result = self.result_module.process(raw_result)
        self.assertEqual(result["status"], "partial_failed")
        self.assertEqual(
            result["metadata"]["request_outcome"], "partial_failed"
        )
        self.assertEqual(result["metadata"]["total_failed_requests"], 1)


class CircuitBreakerTest(unittest.TestCase):
    """Verify circuit breaker in serving-real benchmark."""

    def setUp(self) -> None:
        self.collector_module = load_module(
            "serving_real_bench_cb", SERVING_REAL_BENCHMARK
        )

    def test_circuit_breaker_triggers_on_p99_exceeded(self) -> None:
        def fake_run_case(**kwargs):
            rate = kwargs["request_rate"]
            p99 = 100 if rate == 1.0 else 300
            requests = [
                make_request_record(e2el_ms=p99, ttft_ms=50.0)
                for _ in range(10)
            ]
            return {
                "request_rate": rate,
                "arrival_process": kwargs.get("arrival_process", "gamma"),
                "burstiness": kwargs.get("burstiness", 1.0),
                "num_prompts": kwargs.get("rounds", 10),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "benchmark_duration_seconds": 1.0,
                "maximum_request_concurrency": kwargs["max_concurrency"],
                "peak_concurrent_requests": 1,
                "requests": requests,
            }

        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "workload_mode": "dataset",
            "warmup": 0,
            "rounds": 10,
            "max_concurrency": 8,
            "dataset_path": "/fake/path.json",
            "num_prompts": 10,
            "arrival_process": "gamma",
            "request_rates": [1.0, 4.0, 16.0],
            "burstiness": 1.0,
            "max_tokens": 2048,
            "seed": 0,
            "request_timeout": 5,
            "slo": {"ttft_ms": 200, "e2el_ms": 1000},
            "circuit_breaker": 150,
        }
        request = {"model_name": "test-model"}

        with (
            patch.object(
                self.collector_module,
                "run_case_dataset",
                side_effect=fake_run_case,
            ),
            patch.object(
                self.collector_module.data_loader,
                "load_prompts",
                return_value=["p"] * 10,
            ),
            patch.object(
                self.collector_module.http_client,
                "discover_served_model_name",
                return_value="test-model",
            ),
        ):
            raw_result = self.collector_module.collect_raw_result(
                request, parameters
            )

        self.assertIn("circuit_breaker", raw_result["metadata"])
        cb = raw_result["metadata"]["circuit_breaker"]
        self.assertTrue(cb["triggered"])
        self.assertEqual(cb["threshold_p99_ms"], 150.0)
        self.assertEqual(cb["remaining_cases_skipped"], 1)
        self.assertEqual(len(raw_result["metrics"]["cases"]), 2)

    def test_no_circuit_breaker_without_p99_threshold(self) -> None:
        def fake_run_case(**kwargs):
            return {
                "request_rate": kwargs["request_rate"],
                "arrival_process": kwargs.get("arrival_process"),
                "burstiness": kwargs.get("burstiness"),
                "num_prompts": 10,
                "max_tokens": 2048,
                "benchmark_duration_seconds": 1.0,
                "maximum_request_concurrency": kwargs["max_concurrency"],
                "peak_concurrent_requests": 1,
                "requests": [make_request_record() for _ in range(10)],
            }

        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "workload_mode": "dataset",
            "warmup": 0,
            "rounds": 10,
            "max_concurrency": 8,
            "dataset_path": "/fake/path.json",
            "num_prompts": 10,
            "arrival_process": "poisson",
            "request_rates": [1.0, 4.0],
            "burstiness": 1.0,
            "max_tokens": 2048,
            "seed": 0,
            "request_timeout": 5,
        }
        request = {"model_name": "test-model"}

        with (
            patch.object(
                self.collector_module,
                "run_case_dataset",
                side_effect=fake_run_case,
            ),
            patch.object(
                self.collector_module.data_loader,
                "load_prompts",
                return_value=["p"] * 10,
            ),
            patch.object(
                self.collector_module.http_client,
                "discover_served_model_name",
                return_value="test-model",
            ),
        ):
            raw_result = self.collector_module.collect_raw_result(
                request, parameters
            )

        self.assertNotIn("circuit_breaker", raw_result["metadata"])
        self.assertEqual(len(raw_result["metrics"]["cases"]), 2)


if __name__ == "__main__":
    unittest.main()
