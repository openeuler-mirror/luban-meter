import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from luban_meter.core.models import (
    BenchmarkSpec,
    RawRunArtifacts,
    ResolvedRun,
    RunRequest,
)
from luban_meter.result.manager import ResultManager
from luban_meter.vendors.nvidia.benchmark.generate.common.streaming import (
    collect_completion_stream,
)

ROOT = Path(__file__).parents[1]
SERVING_RESULT_HANDLER = (
    ROOT
    / "src/luban_meter/vendors/nvidia/benchmark/generate/serving-online/result.py"
)
SERVING_BENCHMARK_ENTRY = (
    ROOT
    / "src/luban_meter/vendors/nvidia/benchmark/generate/serving-online/benchmark.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResultManagerTest(unittest.TestCase):
    def test_uses_the_registered_tool_result_handler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            raw_result = raw_dir / "raw_result.json"
            raw_result.write_text(
                json.dumps(
                    {
                        "schema_version": "luban-meter.raw/v1",
                        "status": "success",
                        "metrics": {"samples_ms": [10.0, 14.0]},
                        "metadata": {"source": "test"},
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )

            processor = root / "result.py"
            processor.write_text(
                "def process(raw_result):\n"
                "    samples = raw_result['metrics']['samples_ms']\n"
                "    return {\n"
                "        'environment': {'runtime': {'name': 'test'}},\n"
                "        'metrics': {'mean_ms': sum(samples) / len(samples)},\n"
                "        'metadata': {'processed': True},\n"
                "    }\n",
                encoding="utf-8",
            )
            benchmark = root / "benchmark.py"
            benchmark.write_text("", encoding="utf-8")

            request = RunRequest(
                run_id="generate-result-test",
                module="generate",
                vendor="ascend",
                benchmark="ttft",
                config=root / "ttft.yaml",
                model_path=None,
                model_name=None,
                output_dir=root,
            )
            run = ResolvedRun(
                request=request,
                benchmark=BenchmarkSpec(
                    vendor="ascend",
                    module="generate",
                    benchmark="ttft",
                    benchmark_entry=benchmark,
                    result_handler=processor,
                ),
                parameters={"concurrency": 1},
            )
            artifacts = RawRunArtifacts(
                raw_result=raw_result,
                stdout_log=raw_dir / "stdout.log",
                stderr_log=raw_dir / "stderr.log",
                artifact_dir=raw_dir / "artifacts",
            )

            result = ResultManager().process(run, artifacts)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.vendor, "ascend")
            self.assertEqual(result.benchmark, "ttft")
            self.assertEqual(result.metrics["mean_ms"], 12.0)
            self.assertEqual(result.environment["runtime"]["name"], "test")
            self.assertTrue(result.metadata["processed"])
            self.assertEqual(result.metadata["source"], "test")


class NvidiaStreamingCollectionTest(unittest.TestCase):
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

        observation = collect_completion_stream(stream, clock=lambda: next(times))

        self.assertEqual(observation.generated_text, "ABC")
        self.assertEqual(observation.event_times, (10.0, 10.2))
        self.assertEqual(observation.input_tokens, 3)
        self.assertEqual(observation.output_tokens, 3)

    def test_requires_stream_usage(self) -> None:
        stream = io.BytesIO(
            b'data: {"choices": [{"text": "A"}]}\n\n'
            b"data: [DONE]\n\n"
        )

        with self.assertRaisesRegex(RuntimeError, "did not report token usage"):
            collect_completion_stream(stream, clock=lambda: 10.0)


class NvidiaServingMetricsTest(unittest.TestCase):
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
            "nvidia_serving_result", SERVING_RESULT_HANDLER
        ).process
        result = processor(raw_result)
        case = result["metrics"]["cases"][0]
        request_view = case["request_view"]
        service_view = case["service_view"]

        self.assertEqual(len(request_view), 9)
        self.assertEqual(len(service_view), 15)
        self.assertEqual(request_view["ttft"]["count"], 2)
        self.assertEqual(request_view["itl"]["count"], 3)
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
    def __init__(self, prompt_tokens: int, output_tokens: int) -> None:
        super().__init__(
            b'data: {"choices": [{"text": "A"}]}\n\n'
            b'data: {"choices": [{"text": "BC"}]}\n\n'
            + (
                'data: {"choices": [], "usage": '
                f'{{"prompt_tokens": {prompt_tokens}, '
                f'"completion_tokens": {output_tokens}}}}}\n\n'
            ).encode()
            +
            b"data: [DONE]\n\n"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FakeJsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class NvidiaServingCollectionTest(unittest.TestCase):
    def test_collects_complete_serving_scenario(self) -> None:
        benchmark = load_module(
            "nvidia_serving_benchmark", SERVING_BENCHMARK_ENTRY
        )
        parameters = {
            "service_url": "http://127.0.0.1:8000",
            "warmup": 0,
            "rounds": 2,
            "max_concurrency": 2,
            "input_lengths": [4],
            "output_lengths": [1, 3],
            "request_rates": [100000.0, 200000.0],
            "seed_prompt": "test",
            "request_timeout": 5,
        }
        request = {"model_name": "test-model"}
        completion_payloads = []

        def fake_urlopen(http_request, **kwargs):
            if http_request.full_url.endswith("/tokenize"):
                return FakeJsonResponse(
                    json.dumps(
                        {"tokens": [11, 12], "count": 2, "max_model_len": 32}
                    ).encode()
                )
            payload = json.loads(http_request.data)
            completion_payloads.append(payload)
            return FakeStreamingResponse(len(payload["prompt"]), payload["max_tokens"])

        with patch.object(
            benchmark.urllib.request,
            "urlopen",
            side_effect=fake_urlopen,
        ):
            raw_result = benchmark.run_benchmark(request, parameters)
        result = load_module(
            "nvidia_serving_result_complete", SERVING_RESULT_HANDLER
        ).process(raw_result)

        cases = result["metrics"]["cases"]
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0]["input_length"], 4)
        self.assertEqual(cases[0]["output_length"], 1)
        self.assertEqual(cases[0]["request_rate"], 100000.0)
        self.assertEqual(cases[1]["request_rate"], 200000.0)
        self.assertEqual(cases[0]["request_view"]["ttft"]["count"], 2)
        self.assertEqual(cases[0]["request_view"]["tpot"]["count"], 0)
        self.assertEqual(cases[2]["request_view"]["tpot"]["count"], 2)
        self.assertEqual(cases[3]["service_view"]["total_requests"]["value"], 2)
        self.assertEqual(result["metadata"]["total_successful_requests"], 8)
        self.assertEqual(len(completion_payloads), 8)
        self.assertTrue(all(len(payload["prompt"]) == 4 for payload in completion_payloads))
        self.assertTrue(
            all(
                payload["min_tokens"] == payload["max_tokens"]
                and payload["ignore_eos"] is True
                and payload["add_special_tokens"] is False
                for payload in completion_payloads
            )
        )

    def test_rejects_a_response_with_the_wrong_exact_token_count(self) -> None:
        benchmark = load_module(
            "nvidia_serving_benchmark_mismatch", SERVING_BENCHMARK_ENTRY
        )
        now = benchmark.time.perf_counter()
        with patch.object(
            benchmark.urllib.request,
            "urlopen",
            return_value=FakeStreamingResponse(4, 2),
        ):
            record = benchmark.execute_request(
                request_index=0,
                prompt_token_ids=[1, 2, 3, 4],
                output_length=3,
                service_url="http://127.0.0.1:8000",
                model="test-model",
                api_key="",
                timeout=5,
                benchmark_start=now,
                scheduled_time=now,
                tracker=benchmark.ActiveRequestTracker(),
            )

        self.assertEqual(record["status"], "failed")
        self.assertIn("expected 3 output tokens", record["error"]["message"])


if __name__ == "__main__":
    unittest.main()
