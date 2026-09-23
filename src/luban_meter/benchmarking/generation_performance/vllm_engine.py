"""Engine collection and scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.benchmarking.generation_performance.common.statistics import (
    scalar,
    summarize,
)

ENGINE_SLO_DIMENSIONS = (
    "internal_ttft_ms",
    "prefill_latency_ms",
    "mean_decode_step_latency_ms",
    "engine_execution_latency_ms",
)


def positive_integer(
    parameters: Mapping[str, Any], name: str, default: int
) -> int:
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
        raise ValueError(
            f"{name} must be a non-empty list of positive integers"
        )
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


def string_value(
    parameters: Mapping[str, Any], name: str, default: str
) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def boolean_value(
    parameters: Mapping[str, Any], name: str, default: bool
) -> bool:
    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def fixed_value(
    parameters: Mapping[str, Any], name: str, expected: Any
) -> Any:
    value = parameters.get(name, expected)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(
            f"{name} must be {expected!r} for comparable measurements"
        )
    return value


def engine_slo_config(
    parameters: Mapping[str, Any],
) -> dict[str, float] | None:
    raw = parameters.get("engine_slo")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("engine_slo must be an object")

    unsupported = [name for name in raw if name not in ENGINE_SLO_DIMENSIONS]
    if unsupported:
        names = ", ".join(sorted(repr(name) for name in unsupported))
        raise ValueError(
            f"engine_slo contains unsupported thresholds: {names}"
        )

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


def resolve_model_reference(request: Mapping[str, Any]) -> str:
    model_reference = request.get("model_path") or request.get("model_name")
    if not isinstance(model_reference, str) or not model_reference:
        raise ValueError(
            "offline_vllm_engine requires --model-path or --model-name"
        )
    return model_reference


def build_engine(
    request: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Any:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vLLM is required for offline_vllm_engine") from exc

    kwargs: dict[str, Any] = {
        "model": resolve_model_reference(request),
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
        # Required for RequestOutput.metrics and the engine timestamps
        # below.
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
        raise RuntimeError(
            "vLLM cache_config is unavailable after initialization"
        )
    return {
        "num_gpu_blocks": positive_attribute(
            getattr(cache_config, "num_gpu_blocks", None),
            "num_gpu_blocks",
            integer=True,
        ),
        "block_size": positive_attribute(
            getattr(cache_config, "block_size", None),
            "block_size",
            integer=True,
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
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    request_batch_size = len(prompts)
    outputs = engine.generate(prompts, sampling, use_tqdm=False)
    if not isinstance(outputs, Sequence) or len(outputs) != request_batch_size:
        actual = (
            len(outputs) if isinstance(outputs, Sequence) else "non-sequence"
        )
        raise RuntimeError(
            f"vLLM returned {actual} outputs for {request_batch_size} requests"
        )
    return [
        collect_request_record(output, request_index)
        for request_index, output in enumerate(outputs)
    ]


DECODE_SLO_DIMENSION = "mean_decode_step_latency_ms"


def object_list(value: Any, name: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise TypeError(f"{name} must be a list of objects")
    return list(value)


def result_positive_integer(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_number(record: Mapping[str, Any], name: str) -> float:
    value = record.get(name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def validate_engine_slo_config(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("raw metadata.engine_slo_config must be an object")

    unsupported = [name for name in value if name not in ENGINE_SLO_DIMENSIONS]
    if unsupported:
        names = ", ".join(sorted(repr(name) for name in unsupported))
        raise ValueError(
            "raw metadata.engine_slo_config contains unsupported "
            f"thresholds: {names}"
        )

    config = {
        name: positive_number(value, name)
        for name in ENGINE_SLO_DIMENSIONS
        if name in value
    }
    if not config:
        raise ValueError(
            (
                "raw metadata.engine_slo_config must contain at "
                "least one threshold"
            )
        )
    return config


def compute_engine_goodput(
    *,
    observations: list[Mapping[str, Any]],
    engine_active_duration_seconds: float,
    engine_slo_config: Mapping[str, float],
) -> dict[str, Any]:
    if not observations:
        raise ValueError("engine Goodput requires request observations")
    if (
        not math.isfinite(engine_active_duration_seconds)
        or engine_active_duration_seconds <= 0
    ):
        raise ValueError("engine Goodput duration must be positive")

    decode_applicable = any(
        observation.get(DECODE_SLO_DIMENSION) is not None
        for observation in observations
    )
    applicable_dimensions = [
        name
        for name in ENGINE_SLO_DIMENSIONS
        if name in engine_slo_config
        and (name != DECODE_SLO_DIMENSION or decode_applicable)
    ]
    not_applicable_dimensions = (
        [DECODE_SLO_DIMENSION]
        if DECODE_SLO_DIMENSION in engine_slo_config and not decode_applicable
        else []
    )

    common = {
        "measurement_boundary": "vllm_engine_internal",
        "duration_basis": "sum_of_formal_round_engine_windows",
        "engine_slo_config": {
            name: engine_slo_config[name]
            for name in ENGINE_SLO_DIMENSIONS
            if name in engine_slo_config
        },
        "applicable_dimensions": applicable_dimensions,
        "not_applicable_dimensions": not_applicable_dimensions,
        "engine_active_duration": scalar(
            engine_active_duration_seconds, "s", precision=6
        ),
    }
    if not applicable_dimensions:
        return {
            "status": "not_applicable",
            "reason": "mean_decode_step_latency_ms is the only configured "
            "dimension and output_length == 1",
            **common,
        }

    satisfied: list[Mapping[str, Any]] = []
    violated: list[Mapping[str, Any]] = []
    for observation in observations:
        request_violated = False
        for name in applicable_dimensions:
            value = observation.get(name)
            if value is None and name == DECODE_SLO_DIMENSION:
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(
                    f"engine Goodput observation {name} must be positive"
                )
            if float(value) > engine_slo_config[name]:
                request_violated = True
                break
        if request_violated:
            violated.append(observation)
        else:
            satisfied.append(observation)

    satisfied_count = len(satisfied)
    violated_count = len(violated)
    evaluated_count = satisfied_count + violated_count
    satisfied_output_tokens = sum(
        result_positive_integer(observation, "output_tokens")
        for observation in satisfied
    )
    return {
        "status": "applicable",
        **common,
        "evaluated_request_count": scalar(evaluated_count, "request"),
        "engine_slo_satisfied_count": scalar(satisfied_count, "request"),
        "engine_slo_violated_count": scalar(violated_count, "request"),
        "engine_slo_satisfied_rate": scalar(
            satisfied_count / evaluated_count, "ratio", precision=4
        ),
        "engine_goodput_request_throughput": scalar(
            satisfied_count / engine_active_duration_seconds, "req/s"
        ),
        "engine_goodput_output_token_throughput": scalar(
            satisfied_output_tokens / engine_active_duration_seconds, "token/s"
        ),
    }


def validate_environment(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    environment = raw_result.get("environment")
    if not isinstance(environment, Mapping):
        raise TypeError("raw environment must be an object")
    kv_cache = environment.get("kv_cache")
    if not isinstance(kv_cache, Mapping):
        raise TypeError("raw environment.kv_cache must be an object")
    return {
        "kv_cache": {
            "num_gpu_blocks": result_positive_integer(
                kv_cache, "num_gpu_blocks"
            ),
            "block_size": result_positive_integer(kv_cache, "block_size"),
            "kv_cache_size_tokens": result_positive_integer(
                kv_cache, "kv_cache_size_tokens"
            ),
            "kv_cache_max_concurrency": positive_number(
                kv_cache, "kv_cache_max_concurrency"
            ),
        }
    }


def process_case(
    raw_case: Mapping[str, Any],
    engine_slo_config: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    request_batch_size = result_positive_integer(
        raw_case, "request_batch_size"
    )
    rounds = object_list(raw_case.get("rounds"), "case rounds")
    if not rounds:
        raise ValueError("case rounds must not be empty")

    internal_ttft: list[float] = []
    prefill_latency: list[float] = []
    prefill_rate: list[float] = []
    decode_latency: list[float] = []
    mean_decode_step_latency: list[float] = []
    per_sequence_decode_rate: list[float] = []
    engine_execution_latency: list[float] = []
    aggregate_prefill_rate: list[float] = []
    aggregate_decode_rate: list[float] = []
    goodput_observations: list[dict[str, Any]] = []
    engine_active_duration_seconds = 0.0

    for raw_round in rounds:
        requests = object_list(raw_round.get("requests"), "round requests")
        if len(requests) != request_batch_size:
            raise ValueError(
                (
                    "round request count must equal the configured "
                    "request_batch_size"
                )
            )

        round_prompt_tokens = 0
        round_decode_tokens = 0
        scheduled_times: list[float] = []
        first_token_times: list[float] = []
        last_token_times: list[float] = []
        for record in requests:
            if record["request_batch_size"] != request_batch_size:
                raise ValueError("request batch size does not match case")
            prompt_tokens = result_positive_integer(
                record, "actual_prompt_tokens"
            )
            output_tokens = result_positive_integer(
                record, "actual_output_tokens"
            )

            ttft_seconds = positive_number(record, "internal_ttft_seconds")
            scheduled_ts = positive_number(record, "scheduled_ts")
            first_token_ts = positive_number(record, "first_token_ts")
            last_token_ts = positive_number(record, "last_token_ts")
            if not scheduled_ts <= first_token_ts <= last_token_ts:
                raise ValueError(
                    (
                        "request timestamps must satisfy scheduled <= "
                        "first <= last"
                    )
                )

            prefill_seconds = first_token_ts - scheduled_ts
            execution_seconds = last_token_ts - scheduled_ts
            if prefill_seconds <= 0 or execution_seconds <= 0:
                raise ValueError("engine stage durations must be positive")
            internal_ttft.append(ttft_seconds * 1000)
            prefill_latency.append(prefill_seconds * 1000)
            prefill_rate.append(prompt_tokens / prefill_seconds)
            engine_execution_latency.append(execution_seconds * 1000)

            mean_decode_step_ms: float | None = None
            if output_tokens > 1:
                decode_seconds = last_token_ts - first_token_ts
                if decode_seconds <= 0:
                    raise ValueError(
                        (
                            "decode duration must be positive when output "
                            "length exceeds 1"
                        )
                    )
                follow_on_tokens = output_tokens - 1
                decode_latency.append(decode_seconds * 1000)
                mean_decode_step_ms = decode_seconds * 1000 / follow_on_tokens
                mean_decode_step_latency.append(mean_decode_step_ms)
                per_sequence_decode_rate.append(
                    follow_on_tokens / decode_seconds
                )
                round_decode_tokens += follow_on_tokens

            goodput_observations.append(
                {
                    "output_tokens": output_tokens,
                    "internal_ttft_ms": ttft_seconds * 1000,
                    "prefill_latency_ms": prefill_seconds * 1000,
                    "mean_decode_step_latency_ms": mean_decode_step_ms,
                    "engine_execution_latency_ms": execution_seconds * 1000,
                }
            )

            round_prompt_tokens += prompt_tokens
            scheduled_times.append(scheduled_ts)
            first_token_times.append(first_token_ts)
            last_token_times.append(last_token_ts)

        engine_window = max(last_token_times) - min(scheduled_times)
        if engine_window <= 0:
            raise ValueError("formal round engine window must be positive")
        engine_active_duration_seconds += engine_window

        if all(r["actual_output_tokens"] == 1 for r in requests):
            prefill_window = max(first_token_times) - min(scheduled_times)
            if prefill_window <= 0:
                raise ValueError("aggregate prefill window must be positive")
            aggregate_prefill_rate.append(round_prompt_tokens / prefill_window)
        else:
            decode_window = max(last_token_times) - min(first_token_times)
            if decode_window <= 0:
                raise ValueError("aggregate decode window must be positive")
            aggregate_decode_rate.append(round_decode_tokens / decode_window)

    expected_samples = len(rounds) * request_batch_size
    if len(internal_ttft) != expected_samples:
        raise ValueError("case request sample count is inconsistent")
    case_result = {
        "request_batch_size": request_batch_size,
        "request_metrics": {
            "internal_ttft": summarize(internal_ttft, "ms"),
            "prefill_latency": summarize(prefill_latency, "ms"),
            "prefill_token_throughput": summarize(prefill_rate, "token/s"),
            "decode_latency": summarize(decode_latency, "ms"),
            "mean_decode_step_latency": summarize(
                mean_decode_step_latency, "ms/token"
            ),
            "per_sequence_decode_rate": summarize(
                per_sequence_decode_rate, "token/s/sequence"
            ),
            "engine_execution_latency": summarize(
                engine_execution_latency, "ms"
            ),
        },
        "batch_metrics": {
            "aggregate_prefill_token_throughput": summarize(
                aggregate_prefill_rate, "token/s"
            ),
            "aggregate_decode_token_throughput": summarize(
                aggregate_decode_rate, "token/s"
            ),
        },
    }
    if engine_slo_config is not None:
        case_result["engine_goodput"] = compute_engine_goodput(
            observations=goodput_observations,
            engine_active_duration_seconds=engine_active_duration_seconds,
            engine_slo_config=engine_slo_config,
        )
    return case_result


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")
    raw_cases = object_list(raw_metrics.get("cases"), "raw cases")
    if not raw_cases:
        raise ValueError("raw cases must not be empty")

    metadata = raw_result.get("metadata")
    engine_slo_config: Mapping[str, float] | None = None
    if (
        isinstance(metadata, Mapping)
        and metadata.get("engine_slo_config") is not None
    ):
        engine_slo_config = validate_engine_slo_config(
            metadata["engine_slo_config"]
        )
    return {
        "environment": validate_environment(raw_result),
        "metrics": {
            "cases": [
                process_case(case, engine_slo_config) for case in raw_cases
            ]
        },
        "metadata": {
            **(dict(metadata) if isinstance(metadata, Mapping) else {}),
        },
    }
