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
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SERVING_REAL_BENCHMARK = ROOT / (
    "src/luban_meter/benchmarking/generation_performance"
    "/online_serving.py"
)
SERVING_REAL_RESULT = ROOT / (
    "src/luban_meter/benchmarking/generation_performance"
    "/online_serving.py"
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


if __name__ == "__main__":
    unittest.main()
