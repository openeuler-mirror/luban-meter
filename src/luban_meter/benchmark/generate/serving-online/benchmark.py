"""Collect exact-length online serving cases at fixed request rates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from luban_meter.benchmark.generate.common.statistics import percentile
from luban_meter.benchmark.generate.common.streaming import (
    collect_completion_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online serving benchmark")
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


def positive_number_list(
    parameters: Mapping[str, Any], name: str, default: Sequence[float]
) -> list[float]:
    value = parameters.get(name, default)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or item <= 0
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a non-empty list of positive numbers")
    result = [float(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def positive_number(
    parameters: Mapping[str, Any], name: str, default: float
) -> float:
    value = parameters.get(name, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def string_value(parameters: Mapping[str, Any], name: str, default: str) -> str:
    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def fixed_value(parameters: Mapping[str, Any], name: str, expected: Any) -> Any:
    value = parameters.get(name, expected)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r} for exact-length cases")
    return value


def optional_positive_number(
    parameters: Mapping[str, Any], name: str
) -> float | None:
    value = parameters.get(name)
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def slo_config(parameters: Mapping[str, Any]) -> dict[str, float] | None:
    raw = parameters.get("slo")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("slo must be an object")
    config: dict[str, float] = {}
    for name in ("p99_ms", "ttft_ms", "tpot_ms", "e2el_ms"):
        threshold = optional_positive_number(raw, name)
        if threshold is not None:
            config[name] = threshold
    if not config:
        raise ValueError("slo must contain at least one threshold")
    return config


def request_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(
    url: str, payload: Mapping[str, Any], api_key: str, timeout: float
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"POST {url} returned a non-object response")
    return value


def discover_model(service_url: str, api_key: str, timeout: float) -> str:
    url = f"{service_url.rstrip('/')}/v1/models"
    request = urllib.request.Request(url, headers=request_headers(api_key))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"model discovery failed with HTTP {exc.code}: {detail}"
        ) from exc
    models = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(models, list) or not models:
        raise RuntimeError(f"serving endpoint returned no models from {url}")
    model = models[0].get("id") if isinstance(models[0], Mapping) else None
    if not isinstance(model, str) or not model:
        raise RuntimeError(f"serving endpoint returned an invalid model from {url}")
    return model


def tokenize_seed_prompt(
    service_url: str,
    model: str,
    seed_prompt: str,
    api_key: str,
    timeout: float,
) -> tuple[list[int], int]:
    response = post_json(
        f"{service_url.rstrip('/')}/tokenize",
        {
            "model": model,
            "prompt": seed_prompt,
            "add_special_tokens": False,
        },
        api_key,
        timeout,
    )
    tokens = response.get("tokens")
    count = response.get("count")
    max_model_len = response.get("max_model_len")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens)
        or count != len(tokens)
    ):
        raise RuntimeError("/tokenize returned invalid token IDs or count")
    if (
        not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
        or max_model_len <= 0
    ):
        raise RuntimeError("/tokenize returned an invalid max_model_len")
    return tokens, max_model_len


def exact_prompt_token_ids(
    seed_tokens: Sequence[int], length: int, request_index: int
) -> list[int]:
    offset = request_index % len(seed_tokens)
    rotated = [*seed_tokens[offset:], *seed_tokens[:offset]]
    repeats = math.ceil(length / len(rotated))
    return (rotated * repeats)[:length]


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


def execute_request(
    *,
    request_index: int,
    prompt_token_ids: list[int],
    output_length: int,
    service_url: str,
    model: str,
    api_key: str,
    timeout: float,
    benchmark_start: float,
    scheduled_time: float,
    tracker: ActiveRequestTracker,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_token_ids,
        "add_special_tokens": False,
        "max_tokens": output_length,
        "min_tokens": output_length,
        "temperature": 0.0,
        "ignore_eos": True,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{service_url.rstrip('/')}/v1/completions"
    http_request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(api_key),
        method="POST",
    )
    started = time.perf_counter()
    tracker.enter()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            observation = collect_completion_stream(response)
            ended = time.perf_counter()

        if observation.input_tokens != len(prompt_token_ids):
            raise RuntimeError(
                f"expected {len(prompt_token_ids)} input tokens, "
                f"service reported {observation.input_tokens}"
            )
        if observation.output_tokens != output_length:
            raise RuntimeError(
                f"expected {output_length} output tokens, "
                f"service reported {observation.output_tokens}"
            )
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
            "start_offset_ms": round((started - benchmark_start) * 1000, 3),
            "dispatch_delay_ms": round(max(0.0, started - scheduled_time) * 1000, 3),
            "end_offset_ms": round((ended - benchmark_start) * 1000, 3),
            "duration_ms": round((ended - started) * 1000, 3),
            "ttft_ms": round(ttft_ms, 3),
            "e2el_ms": round((ended - started) * 1000, 3),
            "itl_samples_ms": [round(value, 3) for value in itl_samples_ms],
            "input_tokens": observation.input_tokens,
            "output_tokens": observation.output_tokens,
            "stream_event_count": len(event_times),
        }
    except urllib.error.HTTPError as exc:
        ended = time.perf_counter()
        detail = exc.read().decode("utf-8", errors="replace")
        error: Exception = RuntimeError(f"HTTP {exc.code}: {detail}")
    except Exception as exc:  # noqa: BLE001
        ended = time.perf_counter()
        error = exc
    finally:
        tracker.exit()

    return {
        "request_index": request_index,
        "status": "failed",
        "scheduled_offset_ms": round((scheduled_time - benchmark_start) * 1000, 3),
        "start_offset_ms": round((started - benchmark_start) * 1000, 3),
        "dispatch_delay_ms": round(max(0.0, started - scheduled_time) * 1000, 3),
        "end_offset_ms": round((ended - benchmark_start) * 1000, 3),
        "duration_ms": round((ended - started) * 1000, 3),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def case_p99_e2el_ms(case: Mapping[str, Any]) -> float | None:
    """Compute P99 E2EL from successful requests in a case.

    Returns None when fewer than 10 successful samples are available, because
    P99 is statistically unstable for small sample sizes.
    """
    successful = [
        record
        for record in case.get("requests", [])
        if isinstance(record, Mapping) and record.get("status") == "success"
    ]
    if len(successful) < 10:
        return None
    e2el_values = [
        float(record["e2el_ms"])
        for record in successful
        if isinstance(record.get("e2el_ms"), (int, float))
        and not isinstance(record.get("e2el_ms"), bool)
    ]
    if len(e2el_values) < 10:
        return None
    return round(percentile(e2el_values, 0.99), 3)


def run_case(
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
    for index in range(warmup):
        now = time.perf_counter()
        result = execute_request(
            request_index=index,
            prompt_token_ids=exact_prompt_token_ids(seed_tokens, input_length, index),
            output_length=output_length,
            service_url=service_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            benchmark_start=now,
            scheduled_time=now,
            tracker=ActiveRequestTracker(),
        )
        if result["status"] != "success":
            message = result["error"]["message"]
            raise RuntimeError(
                f"warmup failed for input={input_length}, output={output_length}, "
                f"rate={request_rate}: {message}"
            )

    tracker = ActiveRequestTracker()
    benchmark_start = time.perf_counter()
    request_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_concurrency, rounds)) as executor:
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
                    prompt_token_ids=exact_prompt_token_ids(
                        seed_tokens, input_length, index
                    ),
                    output_length=output_length,
                    service_url=service_url,
                    model=model,
                    api_key=api_key,
                    timeout=timeout,
                    benchmark_start=benchmark_start,
                    scheduled_time=scheduled_time,
                    tracker=tracker,
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
        "benchmark_duration_seconds": round(benchmark_end - benchmark_start, 6),
        "maximum_request_concurrency": max_concurrency,
        "peak_concurrent_requests": tracker.peak,
        "requests": request_results,
    }


def run_benchmark(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    service_url = string_value(
        parameters, "service_url", "http://127.0.0.1:8000"
    ).rstrip("/")
    if not service_url:
        raise ValueError("service_url must not be empty")

    api_key = string_value(parameters, "api_key", "")
    timeout = positive_number(parameters, "request_timeout", 120)
    warmup = non_negative_integer(parameters, "warmup", 2)
    rounds = positive_integer(parameters, "rounds", 100)
    max_concurrency = positive_integer(parameters, "max_concurrency", 128)
    input_lengths = positive_integer_list(
        parameters, "input_lengths", [128, 512, 2048]
    )
    output_lengths = positive_integer_list(
        parameters, "output_lengths", [1, 32, 128]
    )
    request_rates = positive_number_list(
        parameters, "request_rates", [1.0, 4.0, 16.0]
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

    slo = slo_config(parameters)

    model = request.get("model_name")
    if not isinstance(model, str) or not model:
        model = discover_model(service_url, api_key, timeout)
    seed_tokens, max_model_len = tokenize_seed_prompt(
        service_url, model, seed_prompt, api_key, timeout
    )
    for input_length in input_lengths:
        for output_length in output_lengths:
            if input_length + output_length > max_model_len:
                raise ValueError(
                    f"input_length {input_length} + output_length {output_length} "
                    f"exceeds service max_model_len {max_model_len}"
                )

    p99_threshold = slo.get("p99_ms") if slo is not None else None
    circuit_breaker: dict[str, Any] | None = None
    skipped_cases: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    # Build a flat list of all case specs for deterministic iteration.
    all_specs: list[dict[str, Any]] = [
        {"input_length": il, "output_length": ol, "request_rate": rr}
        for il in input_lengths
        for ol in output_lengths
        for rr in request_rates
    ]
    for idx, spec in enumerate(all_specs):
        if circuit_breaker is not None:
            skipped_cases.append(
                {
                    "input_length": spec["input_length"],
                    "output_length": spec["output_length"],
                    "request_rate": spec["request_rate"],
                    "skipped_reason": "circuit_breaker_triggered",
                }
            )
            continue

        case = run_case(
            input_length=spec["input_length"],
            output_length=spec["output_length"],
            request_rate=spec["request_rate"],
            seed_tokens=seed_tokens,
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
                    "triggered_at_case": {
                        "input_length": spec["input_length"],
                        "output_length": spec["output_length"],
                        "request_rate": spec["request_rate"],
                    },
                    "remaining_cases_skipped": 0,
                }

    if circuit_breaker is not None:
        circuit_breaker["remaining_cases_skipped"] = len(skipped_cases)

    metadata: dict[str, Any] = {
        "measurement": "client_streaming_serving_exact_length_fixed_rate",
        "protocol": "openai_compatible_completions",
        "service_url": service_url,
        "model": model,
        "max_model_len": max_model_len,
        "rounds_per_case": rounds,
        "warmup_requests_per_case": warmup,
        "case_count": len(cases),
        "max_concurrency": max_concurrency,
        "temperature": 0.0,
        "ignore_eos": True,
        "seed": 0,
    }
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
            "measurement": "client_streaming_serving_exact_length_fixed_rate",
            "protocol": "openai_compatible_completions",
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
