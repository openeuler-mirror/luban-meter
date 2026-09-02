"""Measure offline vLLM Engine prefill and decode performance."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from luban_meter.benchmark.generate.common.device_monitor import (
    print_hardware_info,
)

ENGINE_SLO_DIMENSIONS = (
    "internal_ttft_ms",
    "prefill_latency_ms",
    "mean_decode_step_latency_ms",
    "engine_execution_latency_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM offline-engine benchmark")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_request(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request")
    parameters = payload.get("parameters")
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be an object")
    return dict(request), dict(parameters)


def positive_integer(parameters: Mapping[str, Any], name: str, default: int) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def non_negative_integer(
    parameters: Mapping[str, Any], name: str, default: int
) -> int:
    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def positive_integer_list(
    parameters: Mapping[str, Any], name: str, default: Sequence[int]
) -> list[int]:
    value = parameters.get(name, default)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    result = list(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def bounded_fraction(
    parameters: Mapping[str, Any], name: str, default: float
) -> float:
    value = parameters.get(name, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 < value <= 1
    ):
        raise ValueError(f"{name} must be greater than 0 and at most 1")
    return float(value)


def string_value(parameters: Mapping[str, Any], name: str, default: str) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def boolean_value(parameters: Mapping[str, Any], name: str, default: bool) -> bool:
    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def fixed_value(parameters: Mapping[str, Any], name: str, expected: Any) -> Any:
    value = parameters.get(name, expected)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r} for comparable measurements")
    return value


def engine_slo_config(parameters: Mapping[str, Any]) -> dict[str, float] | None:
    raw = parameters.get("engine_slo")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("engine_slo must be an object")

    unsupported = [name for name in raw if name not in ENGINE_SLO_DIMENSIONS]
    if unsupported:
        names = ", ".join(sorted(repr(name) for name in unsupported))
        raise ValueError(f"engine_slo contains unsupported thresholds: {names}")

    config: dict[str, float] = {}
    for name in ENGINE_SLO_DIMENSIONS:
        if name not in raw or raw[name] is None:
            continue
        value = raw[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"engine_slo.{name} must be a positive number")
        config[name] = float(value)
    if not config:
        raise ValueError("engine_slo must contain at least one threshold")
    return config


def model_reference(request: Mapping[str, Any]) -> str:
    model = request.get("model_path") or request.get("model_name")
    if not isinstance(model, str) or not model:
        raise ValueError("vllm-engine-offline requires --model-path or --model-name")
    return model


def build_engine(request: Mapping[str, Any], parameters: Mapping[str, Any]) -> Any:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vLLM is required for vllm-engine-offline") from exc

    kwargs: dict[str, Any] = {
        "model": model_reference(request),
        "tensor_parallel_size": positive_integer(
            parameters, "tensor_parallel_size", 1
        ),
        "gpu_memory_utilization": bounded_fraction(
            parameters, "gpu_memory_utilization", 0.9
        ),
        "dtype": string_value(parameters, "dtype", "auto"),
        "trust_remote_code": boolean_value(
            parameters, "trust_remote_code", False
        ),
        "enforce_eager": boolean_value(parameters, "enforce_eager", False),
        # Required for RequestOutput.metrics and the engine timestamps below.
        "disable_log_stats": False,
        # Cache hits and chunked prefill change the stage semantics.
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    }
    max_model_len = parameters.get("max_model_len")
    if max_model_len is not None:
        kwargs["max_model_len"] = positive_integer(
            parameters, "max_model_len", 4096
        )
    return LLM(**kwargs)


def exact_prompt_token_ids(tokenizer: Any, seed_prompt: str, length: int) -> list[int]:
    token_ids = tokenizer.encode(seed_prompt, add_special_tokens=False)
    if (
        not isinstance(token_ids, Sequence)
        or isinstance(token_ids, (str, bytes))
        or not token_ids
        or any(not isinstance(item, int) for item in token_ids)
    ):
        raise RuntimeError("the model tokenizer returned no usable prompt tokens")
    repeats = math.ceil(length / len(token_ids))
    return (list(token_ids) * repeats)[:length]


def positive_attribute(value: Any, name: str, *, integer: bool) -> int | float:
    if integer:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"vLLM cache_config.{name} is unavailable")
        return value
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise RuntimeError(f"vLLM cache_config.{name} is unavailable")
    return float(value)


def kv_cache_environment(engine: Any) -> dict[str, int | float]:
    llm_engine = getattr(engine, "llm_engine", None)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        raise RuntimeError("vLLM cache_config is unavailable after initialization")
    return {
        "num_gpu_blocks": positive_attribute(
            getattr(cache_config, "num_gpu_blocks", None),
            "num_gpu_blocks",
            integer=True,
        ),
        "block_size": positive_attribute(
            getattr(cache_config, "block_size", None), "block_size", integer=True
        ),
        "kv_cache_size_tokens": positive_attribute(
            getattr(cache_config, "kv_cache_size_tokens", None),
            "kv_cache_size_tokens",
            integer=True,
        ),
        "kv_cache_max_concurrency": positive_attribute(
            getattr(cache_config, "kv_cache_max_concurrency", None),
            "kv_cache_max_concurrency",
            integer=False,
        ),
    }


def engine_max_model_len(engine: Any) -> int:
    model_config = getattr(engine, "model_config", None)
    value = getattr(model_config, "max_model_len", None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("vLLM model_config.max_model_len is unavailable")
    return value


def metric_timestamp(metrics: Any, name: str) -> float:
    value = getattr(metrics, name, None)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise RuntimeError(f"vLLM RequestOutput.metrics.{name} is unavailable")
    return float(value)


def collect_request_record(output: Any, request_index: int) -> dict[str, Any]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        raise RuntimeError("vLLM RequestOutput.metrics is unavailable")
    prompt_token_ids = getattr(output, "prompt_token_ids", None)
    candidates = getattr(output, "outputs", None)
    if (
        not isinstance(prompt_token_ids, Sequence)
        or isinstance(prompt_token_ids, (str, bytes))
        or not isinstance(candidates, Sequence)
        or len(candidates) != 1
    ):
        raise RuntimeError("vLLM returned an unexpected RequestOutput")
    output_token_ids = getattr(candidates[0], "token_ids", None)
    if not isinstance(output_token_ids, Sequence) or isinstance(
        output_token_ids, (str, bytes)
    ):
        raise TypeError("vLLM returned no output token IDs")

    first_token_latency = metric_timestamp(metrics, "first_token_latency")
    scheduled_ts = metric_timestamp(metrics, "scheduled_ts")
    first_token_ts = metric_timestamp(metrics, "first_token_ts")
    last_token_ts = metric_timestamp(metrics, "last_token_ts")
    if not scheduled_ts <= first_token_ts <= last_token_ts:
        raise RuntimeError("vLLM returned an invalid engine-internal timeline")
    return {
        "request_index": request_index,
        "actual_prompt_tokens": len(prompt_token_ids),
        "actual_output_tokens": len(output_token_ids),
        "internal_ttft_seconds": first_token_latency,
        "scheduled_ts": scheduled_ts,
        "first_token_ts": first_token_ts,
        "last_token_ts": last_token_ts,
    }


def generate_batch(
    engine: Any,
    sampling: Any,
    prompt_token_ids: list[int],
    request_batch_size: int,
) -> list[dict[str, Any]]:
    prompts = [
        {"prompt_token_ids": list(prompt_token_ids)}
        for _ in range(request_batch_size)
    ]
    outputs = engine.generate(prompts, sampling, use_tqdm=False)
    if not isinstance(outputs, Sequence) or len(outputs) != request_batch_size:
        actual = len(outputs) if isinstance(outputs, Sequence) else "non-sequence"
        raise RuntimeError(
            f"vLLM returned {actual} outputs for {request_batch_size} requests"
        )
    return [
        collect_request_record(output, request_index)
        for request_index, output in enumerate(outputs)
    ]


def run_benchmark(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    print_hardware_info()
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required for vllm-engine-offline") from exc

    warmup_rounds = non_negative_integer(parameters, "warmup_rounds", 2)
    rounds = positive_integer(parameters, "rounds", 10)
    input_lengths = positive_integer_list(
        parameters, "input_lengths", [128, 512, 2048, 8192]
    )
    output_lengths = positive_integer_list(
        parameters, "output_lengths", [1, 32, 128]
    )
    request_batch_sizes = positive_integer_list(
        parameters, "request_batch_sizes", [1, 4, 8]
    )
    seed_prompt = string_value(
        parameters,
        "seed_prompt",
        "Please summarize the benefits of open source software.",
    )
    if not seed_prompt:
        raise ValueError("seed_prompt must not be empty")
    fixed_value(parameters, "temperature", 0.0)
    fixed_value(parameters, "ignore_eos", True)
    fixed_value(parameters, "seed", 0)
    engine_slo = engine_slo_config(parameters)

    engine = build_engine(request, parameters)
    max_model_len = engine_max_model_len(engine)
    for input_length in input_lengths:
        for output_length in output_lengths:
            if input_length + output_length > max_model_len:
                raise ValueError(
                    f"input_length {input_length} + output_length {output_length} "
                    f"exceeds vLLM max_model_len {max_model_len}"
                )

    tokenizer = engine.get_tokenizer()
    prompt_tokens = {
        length: exact_prompt_token_ids(tokenizer, seed_prompt, length)
        for length in input_lengths
    }
    cases: list[dict[str, Any]] = []
    for input_length in input_lengths:
        for output_length in output_lengths:
            sampling = SamplingParams(
                max_tokens=output_length,
                min_tokens=output_length,
                temperature=0.0,
                ignore_eos=True,
                detokenize=False,
                seed=0,
            )
            for request_batch_size in request_batch_sizes:
                for _ in range(warmup_rounds):
                    generate_batch(
                        engine,
                        sampling,
                        prompt_tokens[input_length],
                        request_batch_size,
                    )
                formal_rounds = []
                for round_index in range(rounds):
                    records = generate_batch(
                        engine,
                        sampling,
                        prompt_tokens[input_length],
                        request_batch_size,
                    )
                    if any(
                        record["actual_prompt_tokens"] != input_length
                        or record["actual_output_tokens"] != output_length
                        for record in records
                    ):
                        raise RuntimeError(
                            "vLLM did not honor the configured token lengths"
                        )
                    for record in records:
                        record.update(
                            {
                                "round_index": round_index,
                                "input_length": input_length,
                                "output_length": output_length,
                                "request_batch_size": request_batch_size,
                            }
                        )
                    formal_rounds.append(
                        {"round_index": round_index, "requests": records}
                    )
                cases.append(
                    {
                        "input_length": input_length,
                        "output_length": output_length,
                        "request_batch_size": request_batch_size,
                        "rounds": formal_rounds,
                    }
                )

    metadata: dict[str, Any] = {
        "measurement": "vllm_engine_offline",
        "engine": "vllm",
        "model": model_reference(request),
        "max_model_len": max_model_len,
        "warmup_rounds_per_case": warmup_rounds,
        "formal_rounds_per_case": rounds,
        "prefix_caching": False,
        "chunked_prefill": False,
        "detokenize": False,
    }
    if engine_slo is not None:
        metadata["engine_slo_config"] = engine_slo

    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "environment": {"kv_cache": kv_cache_environment(engine)},
        "metrics": {"cases": cases},
        "metadata": metadata,
        "artifacts": {},
    }


def failure_result(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "failed",
        "metrics": {},
        "metadata": {
            "measurement": "vllm_engine_offline",
            "engine": "vllm",
        },
        "artifacts": {},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main() -> None:
    args = parse_args()
    try:
        request, parameters = load_request(args.request)
        result = run_benchmark(request, parameters)
    except Exception as exc:  # noqa: BLE001
        result = failure_result(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
