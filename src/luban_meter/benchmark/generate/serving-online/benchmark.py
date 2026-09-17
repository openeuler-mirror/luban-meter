"""Online serving benchmark with random and dataset workload modes.

**random** mode sends exact-length token-ID prompts through
``/v1/completions`` at a constant rate, varying input/output lengths.

**dataset** mode sends real conversational prompts from a ShareGPT
dataset through ``/v1/chat/completions`` and schedules their arrival
using Poisson, Gamma, or constant inter-arrival distributions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from luban_meter.benchmark.generate.common.circuit_breaker import (
    case_p99_e2el_ms,
    check_stop_constraints,
    circuit_breaker_config,
    slo_config,
)
from luban_meter.benchmark.generate.common.data_loader import (
    exact_prompt_token_ids,
    load_prompts,
    tokenize_seed_prompt,
)
from luban_meter.benchmark.generate.common.http_client import (
    discover_model,
    request_headers,
)
from luban_meter.benchmark.generate.common.parameters import (
    fixed_value,
    non_negative_integer,
    optional_positive_number,
    positive_integer,
    positive_integer_list,
    positive_number,
    positive_number_list,
    string_value,
)
from luban_meter.benchmark.generate.common.scheduler import (
    schedule_arrival_times,
)
from luban_meter.benchmark.generate.common.streaming import (
    collect_chat_stream,
    collect_completion_stream,
)

# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online serving benchmark"
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_request(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request")
    parameters = payload.get("parameters")
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be an object")
    return dict(request), dict(parameters)


# ── Request execution ───────────────────────────────────────────────────


class ActiveRequestTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._peak = 0

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)

    def exit(self) -> None:
        with self._lock:
            self._active -= 1


def _build_request_record(
    *,
    request_index: int,
    started: float,
    ended: float,
    benchmark_start: float,
    scheduled_time: float,
    observation: Any | None,
    error: Exception | None,
) -> dict[str, Any]:
    """Build a success or failure request record."""
    if error is not None:
        return {
            "request_index": request_index,
            "status": "failed",
            "scheduled_offset_ms": round(
                (scheduled_time - benchmark_start) * 1000, 3
            ),
            "start_offset_ms": round(
                (started - benchmark_start) * 1000, 3
            ),
            "dispatch_delay_ms": round(
                (started - scheduled_time) * 1000, 3
            ),
            "end_offset_ms": round(
                (ended - benchmark_start) * 1000, 3
            ),
            "scheduled_latency_ms": round(
                (ended - scheduled_time) * 1000, 3
            ),
            "duration_ms": round((ended - started) * 1000, 3),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    event_times = observation.event_times
    ttft_ms = (event_times[0] - started) * 1000
    itl_samples_ms = [
        (current - previous) * 1000
        for previous, current in itertools.pairwise(event_times)
    ]
    return {
        "request_index": request_index,
        "status": "success",
        "scheduled_offset_ms": round(
            (scheduled_time - benchmark_start) * 1000, 3
        ),
        "start_offset_ms": round(
            (started - benchmark_start) * 1000, 3
        ),
        "dispatch_delay_ms": round(
            (started - scheduled_time) * 1000, 3
        ),
        "end_offset_ms": round(
            (ended - benchmark_start) * 1000, 3
        ),
        "scheduled_latency_ms": round(
            (ended - scheduled_time) * 1000, 3
        ),
        "duration_ms": round((ended - started) * 1000, 3),
        "ttft_ms": round(ttft_ms, 3),
        "last_output_latency_ms": round(
            (event_times[-1] - started) * 1000, 3
        ),
        "e2el_ms": round((ended - started) * 1000, 3),
        "itl_samples_ms": [round(value, 3) for value in itl_samples_ms],
        "input_tokens": observation.input_tokens,
        "output_tokens": observation.output_tokens,
        "stream_event_count": len(event_times),
    }


def execute_request(
    *,
    request_index: int,
    payload: dict[str, Any],
    endpoint: str,
    service_url: str,
    api_key: str,
    timeout: float,
    benchmark_start: float,
    scheduled_time: float,
    tracker: ActiveRequestTracker,
    stream_collector: Callable[[Any], Any],
    expected_input_tokens: int | None = None,
    expected_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Execute a streaming HTTP request and return a result record.

    ``endpoint`` is ``"/v1/completions"`` or ``"/v1/chat/completions"``.
    ``stream_collector`` is ``collect_completion_stream`` or
    ``collect_chat_stream``.
    """
    url = f"{service_url.rstrip('/')}/{endpoint.lstrip('/')}"
    http_request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(api_key),
        method="POST",
    )
    started = time.perf_counter()
    tracker.enter()
    try:
        with urllib.request.urlopen(
            http_request, timeout=timeout
        ) as response:
            observation = stream_collector(response)
            ended = time.perf_counter()

        if (
            expected_input_tokens is not None
            and observation.input_tokens != expected_input_tokens
        ):
            raise RuntimeError(
                f"expected {expected_input_tokens} input tokens, "
                f"service reported {observation.input_tokens}"
            )
        if (
            expected_output_tokens is not None
            and observation.output_tokens != expected_output_tokens
        ):
            raise RuntimeError(
                f"expected {expected_output_tokens} output tokens, "
                f"service reported {observation.output_tokens}"
            )
        return _build_request_record(
            request_index=request_index,
            started=started,
            ended=ended,
            benchmark_start=benchmark_start,
            scheduled_time=scheduled_time,
            observation=observation,
            error=None,
        )
    except urllib.error.HTTPError as exc:
        ended = time.perf_counter()
        detail = exc.read().decode("utf-8", errors="replace")
        error: Exception = RuntimeError(
            f"HTTP {exc.code}: {detail}"
        )
        return _build_request_record(
            request_index=request_index,
            started=started,
            ended=ended,
            benchmark_start=benchmark_start,
            scheduled_time=scheduled_time,
            observation=None,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        ended = time.perf_counter()
        return _build_request_record(
            request_index=request_index,
            started=started,
            ended=ended,
            benchmark_start=benchmark_start,
            scheduled_time=scheduled_time,
            observation=None,
            error=exc,
        )
    finally:
        tracker.exit()


# ── Case runners ─────────────────────────────────────────────────────────


def run_case_random(
    *,
    input_length: int,
    output_length: int,
    request_rate: float,
    seed_tokens: Sequence[int],
    warmup: int,
    rounds: int,
    max_concurrency: int,
    service_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    """Run a case with exact-length token-ID prompts (random mode)."""
    for index in range(warmup):
        now = time.perf_counter()
        result = execute_request(
            request_index=index,
            payload={
                "model": model,
                "prompt": exact_prompt_token_ids(
                    seed_tokens, input_length, index
                ),
                "add_special_tokens": False,
                "max_tokens": output_length,
                "min_tokens": output_length,
                "temperature": 0.0,
                "ignore_eos": True,
                "seed": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            endpoint="/v1/completions",
            service_url=service_url,
            api_key=api_key,
            timeout=timeout,
            benchmark_start=now,
            scheduled_time=now,
            tracker=ActiveRequestTracker(),
            stream_collector=collect_completion_stream,
            expected_input_tokens=input_length,
            expected_output_tokens=output_length,
        )
        if result["status"] != "success":
            message = result["error"]["message"]
            raise RuntimeError(
                f"warmup failed for input={input_length}, "
                f"output={output_length}, "
                f"rate={request_rate}: {message}"
            )

    tracker = ActiveRequestTracker()
    benchmark_start = time.perf_counter()
    request_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=min(max_concurrency, rounds)
    ) as executor:
        futures = []
        for index in range(rounds):
            scheduled_time = benchmark_start + index / request_rate
            remaining = scheduled_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            futures.append(
                executor.submit(
                    execute_request,
                    request_index=index,
                    payload={
                        "model": model,
                        "prompt": exact_prompt_token_ids(
                            seed_tokens, input_length, index
                        ),
                        "add_special_tokens": False,
                        "max_tokens": output_length,
                        "min_tokens": output_length,
                        "temperature": 0.0,
                        "ignore_eos": True,
                        "seed": 0,
                        "stream": True,
                        "stream_options": {
                            "include_usage": True
                        },
                    },
                    endpoint="/v1/completions",
                    service_url=service_url,
                    api_key=api_key,
                    timeout=timeout,
                    benchmark_start=benchmark_start,
                    scheduled_time=scheduled_time,
                    tracker=tracker,
                    stream_collector=collect_completion_stream,
                    expected_input_tokens=input_length,
                    expected_output_tokens=output_length,
                )
            )
        for future in as_completed(futures):
            request_results.append(future.result())
    benchmark_end = time.perf_counter()
    request_results.sort(key=lambda item: item["request_index"])
    return {
        "input_length": input_length,
        "output_length": output_length,
        "request_rate": request_rate,
        "benchmark_duration_seconds": round(
            benchmark_end - benchmark_start, 6
        ),
        "maximum_request_concurrency": max_concurrency,
        "peak_concurrent_requests": tracker.peak,
        "requests": request_results,
    }


def run_case_dataset(
    *,
    prompts: list[str],
    request_rate: float,
    arrival_process: str,
    burstiness: float,
    max_tokens: int,
    seed: int,
    warmup: int,
    rounds: int,
    max_concurrency: int,
    service_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    """Run a case with real conversational prompts (dataset mode)."""
    num_requests = min(rounds, len(prompts))

    for i in range(min(warmup, len(prompts))):
        warmup_prompt = prompts[i % len(prompts)]
        now = time.perf_counter()
        result = execute_request(
            request_index=i,
            payload={
                "model": model,
                "messages": [
                    {"role": "user", "content": warmup_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            endpoint="/v1/chat/completions",
            service_url=service_url,
            api_key=api_key,
            timeout=timeout,
            benchmark_start=now,
            scheduled_time=now,
            tracker=ActiveRequestTracker(),
            stream_collector=collect_chat_stream,
        )
        if result["status"] != "success":
            message = result["error"]["message"]
            raise RuntimeError(
                f"warmup failed for rate={request_rate}, "
                f"arrival={arrival_process}: {message}"
            )

    arrival_offsets = schedule_arrival_times(
        arrival_process=arrival_process,
        request_rate=request_rate,
        num_requests=num_requests,
        burstiness=burstiness,
        seed=seed,
    )

    tracker = ActiveRequestTracker()
    benchmark_start = time.perf_counter()
    request_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=min(max_concurrency, num_requests)
    ) as executor:
        futures = []
        for i in range(num_requests):
            scheduled_time = benchmark_start + arrival_offsets[i]
            remaining = scheduled_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            futures.append(
                executor.submit(
                    execute_request,
                    request_index=i,
                    payload={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompts[i],
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "seed": 0,
                        "stream": True,
                        "stream_options": {
                            "include_usage": True
                        },
                    },
                    endpoint="/v1/chat/completions",
                    service_url=service_url,
                    api_key=api_key,
                    timeout=timeout,
                    benchmark_start=benchmark_start,
                    scheduled_time=scheduled_time,
                    tracker=tracker,
                    stream_collector=collect_chat_stream,
                )
            )
        for future in as_completed(futures):
            request_results.append(future.result())
    benchmark_end = time.perf_counter()
    request_results.sort(key=lambda item: item["request_index"])
    return {
        "request_rate": request_rate,
        "arrival_process": arrival_process,
        "burstiness": burstiness if arrival_process == "gamma" else None,
        "num_prompts": num_requests,
        "max_tokens": max_tokens,
        "benchmark_duration_seconds": round(
            benchmark_end - benchmark_start, 6
        ),
        "maximum_request_concurrency": max_concurrency,
        "peak_concurrent_requests": tracker.peak,
        "requests": request_results,
    }


# ── Benchmark orchestrator ──────────────────────────────────────────────


def run_benchmark(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    workload_mode = string_value(parameters, "workload_mode", "random")
    if workload_mode not in ("random", "dataset"):
        raise ValueError(
            "workload_mode must be 'random' or 'dataset'"
        )

    service_url = string_value(
        parameters, "service_url", "http://127.0.0.1:8000"
    ).rstrip("/")
    if not service_url:
        raise ValueError("service_url must not be empty")

    api_key = string_value(parameters, "api_key", "")
    timeout = positive_number(parameters, "request_timeout", 120)
    warmup = non_negative_integer(parameters, "warmup", 2)
    rounds = positive_integer(parameters, "rounds", 100)
    max_concurrency = positive_integer(
        parameters, "max_concurrency", 128
    )
    max_duration = optional_positive_number(parameters, "max_duration")
    max_error_rate = optional_positive_number(
        parameters, "max_error_rate"
    )
    request_rates = positive_number_list(
        parameters, "request_rates", [1.0, 4.0, 16.0]
    )
    seed = non_negative_integer(parameters, "seed", 0)

    slo = slo_config(parameters)
    cb_threshold = circuit_breaker_config(parameters)

    model = request.get("model_name")
    if not isinstance(model, str) or not model:
        model = discover_model(service_url, api_key, timeout)

    if workload_mode == "random":
        return _run_mode(
            parameters=parameters,
            service_url=service_url,
            api_key=api_key,
            timeout=timeout,
            warmup=warmup,
            rounds=rounds,
            max_concurrency=max_concurrency,
            request_rates=request_rates,
            seed=seed,
            slo=slo,
            cb_threshold=cb_threshold,
            model=model,
            max_duration=max_duration,
            max_error_rate=max_error_rate,
            mode_specific=_random_mode_setup(
                parameters, model, service_url, api_key, timeout
            ),
        )
    else:
        return _run_mode(
            parameters=parameters,
            service_url=service_url,
            api_key=api_key,
            timeout=timeout,
            warmup=warmup,
            rounds=rounds,
            max_concurrency=max_concurrency,
            request_rates=request_rates,
            seed=seed,
            slo=slo,
            cb_threshold=cb_threshold,
            model=model,
            max_duration=max_duration,
            max_error_rate=max_error_rate,
            mode_specific=_dataset_mode_setup(parameters),
        )


def _random_mode_setup(
    parameters: dict[str, Any],
    model: str,
    service_url: str,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    """Parse random-mode parameters and tokenize seed prompt."""
    input_lengths = positive_integer_list(
        parameters, "input_lengths", [128, 512, 2048]
    )
    output_lengths = positive_integer_list(
        parameters, "output_lengths", [1, 32, 128]
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

    seed_tokens, max_model_len = tokenize_seed_prompt(
        service_url, model, seed_prompt, api_key, timeout
    )
    for input_length in input_lengths:
        for output_length in output_lengths:
            if input_length + output_length > max_model_len:
                raise ValueError(
                    f"input_length {input_length} + output_length "
                    f"{output_length} exceeds service max_model_len "
                    f"{max_model_len}"
                )

    all_specs = [
        {
            "input_length": il,
            "output_length": ol,
            "request_rate": rr,
        }
        for il in input_lengths
        for ol in output_lengths
        for rr in positive_number_list(
            parameters, "request_rates", [1.0, 4.0, 16.0]
        )
    ]
    return {
        "specs": all_specs,
        "seed_tokens": seed_tokens,
        "max_model_len": max_model_len,
        "metadata": {
            "measurement": "client_streaming_serving_exact_length_fixed_rate",
            "protocol": "openai_compatible_completions",
            "workload_mode": "random",
            "max_model_len": max_model_len,
            "temperature": 0.0,
            "ignore_eos": True,
            "seed": 0,
        },
    }


def _dataset_mode_setup(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Parse dataset-mode parameters and load prompts."""
    dataset_path = string_value(parameters, "dataset_path", "")
    if not dataset_path:
        raise ValueError("dataset_path must not be empty")
    num_prompts = positive_integer(parameters, "num_prompts", 1000)

    dataset_format = string_value(
        parameters, "dataset_format", "sharegpt"
    )
    prompt_field = string_value(parameters, "prompt_field", "prompt")

    arrival_process = string_value(
        parameters, "arrival_process", "gamma"
    )
    if arrival_process not in ("constant", "poisson", "gamma"):
        raise ValueError(
            "arrival_process must be 'constant', 'poisson', "
            "or 'gamma'"
        )

    burstiness = positive_number(parameters, "burstiness", 1.0)
    max_tokens = positive_integer(parameters, "max_tokens", 2048)
    seed = non_negative_integer(parameters, "seed", 0)

    prompts = load_prompts(
        dataset_path=dataset_path,
        num_prompts=num_prompts,
        dataset_format=dataset_format,
        prompt_field=prompt_field,
        seed=seed,
    )

    request_rates = positive_number_list(
        parameters, "request_rates", [1.0, 4.0, 16.0]
    )
    all_specs = [
        {
            "request_rate": rate,
            "arrival_process": arrival_process,
        }
        for rate in request_rates
    ]
    return {
        "specs": all_specs,
        "prompts": prompts,
        "dataset_path": dataset_path,
        "num_prompts": num_prompts,
        "arrival_process": arrival_process,
        "burstiness": burstiness,
        "max_tokens": max_tokens,
        "metadata": {
            "measurement": "client_streaming_serving_real_workload",
            "protocol": "openai_compatible_chat_completions",
            "workload_mode": "dataset",
            "dataset_format": dataset_format,
            "dataset_path": dataset_path,
            "num_prompts": num_prompts,
            "arrival_process": arrival_process,
            "burstiness": burstiness,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
        },
    }


def _run_mode(
    *,
    parameters: dict[str, Any],
    service_url: str,
    api_key: str,
    timeout: float,
    warmup: int,
    rounds: int,
    max_concurrency: int,
    request_rates: list[float],
    seed: int,
    slo: dict[str, float] | None,
    cb_threshold: float | None,
    model: str,
    max_duration: float | None,
    max_error_rate: float | None,
    mode_specific: dict[str, Any],
) -> dict[str, Any]:
    """Unified orchestrator for both random and dataset modes."""
    specs = mode_specific["specs"]
    metadata = dict(mode_specific["metadata"])

    p99_threshold = cb_threshold
    circuit_breaker: dict[str, Any] | None = None
    skipped_cases: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for spec in specs:
        if circuit_breaker is not None:
            skip_spec = dict(spec)
            skip_spec["skipped_reason"] = "circuit_breaker_triggered"
            skipped_cases.append(skip_spec)
            continue

        if "input_length" in spec:
            case = run_case_random(
                input_length=spec["input_length"],
                output_length=spec["output_length"],
                request_rate=spec["request_rate"],
                seed_tokens=mode_specific["seed_tokens"],
                warmup=warmup,
                rounds=rounds,
                max_concurrency=max_concurrency,
                service_url=service_url,
                model=model,
                api_key=api_key,
                timeout=timeout,
            )
        else:
            case = run_case_dataset(
                prompts=mode_specific["prompts"],
                request_rate=spec["request_rate"],
                arrival_process=spec["arrival_process"],
                burstiness=mode_specific["burstiness"],
                max_tokens=mode_specific["max_tokens"],
                seed=seed,
                warmup=warmup,
                rounds=rounds,
                max_concurrency=max_concurrency,
                service_url=service_url,
                model=model,
                api_key=api_key,
                timeout=timeout,
            )
        cases.append(case)

        if p99_threshold is not None:
            actual_p99 = case_p99_e2el_ms(case)
            if actual_p99 is not None and actual_p99 > p99_threshold:
                circuit_breaker = {
                    "triggered": True,
                    "threshold_p99_ms": p99_threshold,
                    "actual_p99_ms": actual_p99,
                    "triggered_at_case": dict(spec),
                    "remaining_cases_skipped": 0,
                }

        if circuit_breaker is None:
            stop_reason = check_stop_constraints(
                case,
                max_duration=max_duration,
                max_error_rate=max_error_rate,
            )
            if stop_reason is not None:
                circuit_breaker = {
                    "triggered": True,
                    "reason": stop_reason,
                    "triggered_at_case": dict(spec),
                    "remaining_cases_skipped": 0,
                }

    if circuit_breaker is not None:
        circuit_breaker["remaining_cases_skipped"] = len(
            skipped_cases
        )

    metadata.update(
        {
            "service_url": service_url,
            "model": model,
            "rounds_per_case": rounds,
            "warmup_requests_per_case": warmup,
            "case_count": len(cases),
            "max_concurrency": max_concurrency,
        }
    )
    if slo is not None:
        metadata["slo_config"] = slo
    if circuit_breaker is not None:
        metadata["circuit_breaker"] = circuit_breaker
        metadata["skipped_cases"] = skipped_cases
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
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
            "measurement": "client_streaming_serving",
            "protocol": "openai_compatible",
        },
        "artifacts": {},
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
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
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
