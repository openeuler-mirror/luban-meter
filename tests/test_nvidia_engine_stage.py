import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
ENGINE_STAGE_DIR = (
    ROOT
    / "src/luban_meter/vendors/nvidia/benchmark/generate/vllm-engine-stage"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("benchmark must disable special tokens")
        return [11, 12, 13]


class FakeLLM:
    instances: ClassVar[list["FakeLLM"]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_config = types.SimpleNamespace(max_model_len=32)
        cache_config = types.SimpleNamespace(
            num_gpu_blocks=100,
            block_size=16,
            kv_cache_size_tokens=1600,
            kv_cache_max_concurrency=12.5,
        )
        self.llm_engine = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(cache_config=cache_config)
        )
        self.calls = 0
        self.__class__.instances.append(self)

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, sampling, use_tqdm=False):
        self.calls += 1
        outputs = []
        for index, prompt in enumerate(prompts):
            scheduled = 100.0 + self.calls + index * 0.001
            first = scheduled + 0.01
            last = first + 0.005 * (sampling.max_tokens - 1)
            metrics = types.SimpleNamespace(
                first_token_latency=0.012,
                scheduled_ts=scheduled,
                first_token_ts=first,
                last_token_ts=last,
            )
            candidate = types.SimpleNamespace(
                token_ids=list(range(sampling.max_tokens))
            )
            outputs.append(
                types.SimpleNamespace(
                    metrics=metrics,
                    prompt_token_ids=prompt["prompt_token_ids"],
                    outputs=[candidate],
                )
            )
        return outputs


class NvidiaEngineStageCollectionTest(unittest.TestCase):
    def test_collects_matrix_and_cache_environment(self) -> None:
        benchmark = load_module(
            "nvidia_engine_stage_benchmark", ENGINE_STAGE_DIR / "benchmark.py"
        )
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        FakeLLM.instances.clear()

        parameters = {
            "warmup_rounds": 0,
            "rounds": 1,
            "input_lengths": [4],
            "output_lengths": [1, 3],
            "request_batch_sizes": [2],
            "seed_prompt": "test",
        }
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            raw = benchmark.run_benchmark(
                {"model_path": "/models/test"}, parameters
            )

        self.assertEqual(len(FakeLLM.instances), 1)
        engine = FakeLLM.instances[0]
        self.assertFalse(engine.kwargs["enable_prefix_caching"])
        self.assertFalse(engine.kwargs["enable_chunked_prefill"])
        self.assertFalse(engine.kwargs["disable_log_stats"])
        self.assertEqual(
            raw["environment"]["kv_cache"],
            {
                "num_gpu_blocks": 100,
                "block_size": 16,
                "kv_cache_size_tokens": 1600,
                "kv_cache_max_concurrency": 12.5,
            },
        )
        self.assertEqual(len(raw["metrics"]["cases"]), 2)
        first_record = raw["metrics"]["cases"][0]["rounds"][0]["requests"][0]
        self.assertEqual(first_record["actual_prompt_tokens"], 4)
        self.assertEqual(first_record["actual_output_tokens"], 1)


class NvidiaEngineStageResultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = load_module(
            "nvidia_engine_stage_result", ENGINE_STAGE_DIR / "result.py"
        ).process

    @staticmethod
    def record(
        *, prompt_tokens: int, output_tokens: int, offset: float
    ) -> dict[str, float | int]:
        scheduled = 10.0 + offset
        first = scheduled + 0.1
        last = first + 0.02 * (output_tokens - 1)
        return {
            "input_length": prompt_tokens,
            "output_length": output_tokens,
            "request_batch_size": 2,
            "actual_prompt_tokens": prompt_tokens,
            "actual_output_tokens": output_tokens,
            "internal_ttft_seconds": 0.15,
            "scheduled_ts": scheduled,
            "first_token_ts": first,
            "last_token_ts": last,
        }

    def raw_result(self, cases):
        return {
            "environment": {
                "kv_cache": {
                    "num_gpu_blocks": 100,
                    "block_size": 16,
                    "kv_cache_size_tokens": 1600,
                    "kv_cache_max_concurrency": 12.5,
                }
            },
            "metrics": {"cases": cases},
            "metadata": {"model": "test"},
        }

    def test_separates_prefill_only_and_decode_metrics(self) -> None:
        cases = [
            {
                "input_length": 10,
                "output_length": 1,
                "request_batch_size": 2,
                "rounds": [
                    {
                        "requests": [
                            self.record(
                                prompt_tokens=10, output_tokens=1, offset=0.0
                            ),
                            self.record(
                                prompt_tokens=10, output_tokens=1, offset=0.05
                            ),
                        ]
                    }
                ],
            },
            {
                "input_length": 10,
                "output_length": 3,
                "request_batch_size": 2,
                "rounds": [
                    {
                        "requests": [
                            self.record(
                                prompt_tokens=10, output_tokens=3, offset=0.0
                            ),
                            self.record(
                                prompt_tokens=10, output_tokens=3, offset=0.01
                            ),
                        ]
                    }
                ],
            },
        ]

        result = self.processor(self.raw_result(cases))
        prefill_case, decode_case = result["metrics"]["cases"]

        self.assertEqual(
            result["environment"]["kv_cache"]["kv_cache_size_tokens"], 1600
        )
        self.assertEqual(
            prefill_case["request_metrics"]["prefill_latency"]["mean"], 100.0
        )
        self.assertEqual(
            prefill_case["request_metrics"]["decode_latency"]["count"], 0
        )
        self.assertEqual(
            prefill_case["batch_metrics"][
                "aggregate_prefill_token_throughput"
            ]["mean"],
            133.333,
        )
        self.assertEqual(
            decode_case["request_metrics"]["decode_latency"]["mean"], 40.0
        )
        self.assertEqual(
            decode_case["request_metrics"]["mean_decode_step_latency"]["mean"],
            20.0,
        )
        self.assertEqual(
            decode_case["request_metrics"]["per_sequence_decode_rate"]["mean"],
            50.0,
        )
        self.assertEqual(
            decode_case["batch_metrics"][
                "aggregate_decode_token_throughput"
            ]["mean"],
            80.0,
        )
        self.assertEqual(
            decode_case["batch_metrics"][
                "aggregate_prefill_token_throughput"
            ]["count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
