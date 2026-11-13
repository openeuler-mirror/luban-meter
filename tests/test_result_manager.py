import importlib.util
import io
import unittest
from pathlib import Path

import httpx

from luban_meter.benchmarking.generation_performance.common.streaming import (
    collect_completion_stream,
)

ROOT = Path(__file__).parents[1]
SERVING_RESULT_HANDLER = ROOT / (
    "src/luban_meter/benchmarking/generation_performance/online_serving.py"
)
SERVING_BENCHMARK_ENTRY = ROOT / (
    "src/luban_meter/benchmarking/generation_performance/online_serving.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResultManagerTest(unittest.TestCase):
    pass


class StreamingCollectionTest(unittest.TestCase):
    pass

    def test_skips_empty_events_and_reads_usage(self) -> None:
        stream = io.BytesIO(
            b'data: {"choices": [{"text": ""}]}\n\n'
            b'data: {"choices": [{"text": "A"}]}\n\n'
            b'data: {"choices": [{"text": "BC"}]}\n\n'
            b'data: {"choices": [], "usage": '
            b'{"prompt_tokens": 3, "completion_tokens": 3}}\n\n'
            b"data: [DONE]\n\n"
        )
        times = iter((10.0, 10.2))

        observation = collect_completion_stream(
            stream, clock=lambda: next(times)
        )

        self.assertEqual(observation.generated_text, "ABC")
        self.assertEqual(observation.event_times, (10.0, 10.2))
        self.assertEqual(observation.input_tokens, 3)
        self.assertEqual(observation.output_tokens, 3)

    def test_missing_usage_preserves_stream_observation(self) -> None:
        stream = io.BytesIO(
            b'data: {"choices": [{"text": "A"}]}\n\ndata: [DONE]\n\n'
        )

        observation = collect_completion_stream(stream, clock=lambda: 10.0)
        self.assertIsNone(observation.input_tokens)
        self.assertIsNone(observation.output_tokens)
        self.assertEqual(observation.generated_text, "A")


class ServingMetricsTest(unittest.TestCase):
    pass

    def test_calculates_all_request_and_service_metrics(self) -> None:
        raw_result = {
            "metrics": {
                "cases": [
                    {
                        "input_length": 10,
                        "output_length": 3,
                        "request_rate": 2.0,
                        "benchmark_duration_seconds": 2.0,
                        "maximum_request_concurrency": 2,
                        "peak_concurrent_requests": 2,
                        "requests": [
                            {
                                "status": "success",
                                "start_offset_ms": 0.0,
                                "dispatch_delay_ms": 0.1,
                                "duration_ms": 1000.0,
                                "ttft_ms": 100.0,
                                "last_output_latency_ms": 300.0,
                                "e2el_ms": 1000.0,
                                "itl_samples_ms": [90.0, 110.0],
                                "input_tokens": 10,
                                "output_tokens": 3,
                            },
                            {
                                "status": "success",
                                "start_offset_ms": 500.0,
                                "dispatch_delay_ms": 0.2,
                                "duration_ms": 500.0,
                                "ttft_ms": 50.0,
                                "last_output_latency_ms": 150.0,
                                "e2el_ms": 500.0,
                                "itl_samples_ms": [100.0],
                                "input_tokens": 10,
                                "output_tokens": 3,
                            },
                            {
                                "status": "failed",
                                "start_offset_ms": 1000.0,
                                "dispatch_delay_ms": 0.3,
                                "duration_ms": 250.0,
                            },
                        ],
                    }
                ],
            },
            "metadata": {"model": "test-model"},
        }

        processor = load_module(
            "serving_result", SERVING_RESULT_HANDLER
        ).process
        result = processor(raw_result)
        case = result["metrics"]["cases"][0]
        request_view = case["request_view"]
        service_view = case["service_view"]

        self.assertEqual(len(request_view), 12)
        self.assertEqual(len(service_view), 18)
        self.assertEqual(request_view["ttft"]["count"], 2)
        self.assertEqual(request_view["itl"]["count"], 2)
        self.assertEqual(request_view["itl"]["mean"], 75.0)
        self.assertEqual(request_view["itl"]["weight_sum"], 4)
        self.assertEqual(request_view["stream_event_itl"]["count"], 3)
        self.assertEqual(request_view["tpot"]["count"], 2)
        self.assertEqual(request_view["tpot"]["unit"], "ms/token")
        self.assertEqual(request_view["dispatch_delay"]["mean"], 0.2)
        self.assertEqual(service_view["offered_request_rate"]["value"], 2.0)
        self.assertEqual(
            service_view["achieved_request_start_rate"]["value"], 2.0
        )
        self.assertEqual(service_view["successful_requests"]["value"], 2)
        self.assertEqual(service_view["failed_requests"]["value"], 1)
        self.assertEqual(service_view["average_concurrency"]["value"], 0.875)
        self.assertEqual(service_view["total_input_tokens"]["value"], 20)
        self.assertEqual(service_view["total_output_tokens"]["value"], 6)
        self.assertEqual(service_view["request_throughput"]["value"], 1.0)
        self.assertEqual(service_view["total_token_throughput"]["value"], 13.0)
        self.assertEqual(case["request_outcome"], "partial_failed")
        self.assertEqual(result["status"], "partial_failed")


class FakeStreamingResponse(io.BytesIO):
    pass

    def __init__(self, prompt_tokens: int, output_tokens: int) -> None:
        super().__init__(
            b'data: {"choices": [{"text": "A"}]}\n\n'
            b'data: {"choices": [{"text": "BC"}]}\n\n'
            + (
                'data: {"choices": [], "usage": '
                f'{{"prompt_tokens": {prompt_tokens}, '
                f'"completion_tokens": {output_tokens}}}}}\n\n'
            ).encode()
            + b"data: [DONE]\n\n"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FakeJsonResponse(io.BytesIO):
    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ServingCollectionTest(unittest.TestCase):
    pass

    def test_preserves_actual_counts_when_output_is_short(self) -> None:
        benchmark = load_module(
            "serving_benchmark_mismatch", SERVING_BENCHMARK_ENTRY
        )
        now = benchmark.time.perf_counter()
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, stream=httpx.ByteStream(
                    FakeStreamingResponse(4, 2).getvalue()
                )
            )
        )
        with httpx.Client(transport=transport) as client:
            record = benchmark.execute_request(
                client=client,
                request_index=0,
                payload={
                    "model": "test-model",
                    "prompt": [1, 2, 3, 4],
                    "add_special_tokens": False,
                    "max_tokens": 3,
                    "min_tokens": 3,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "seed": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                endpoint="completion",
                service_url="http://127.0.0.1:8000",
                api_key="",
                timeout=5,
                benchmark_start=now,
                scheduled_time=now,
                tracker=benchmark.ActiveRequestTracker(),
                stream_collector=collect_completion_stream,
            )

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["output_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
