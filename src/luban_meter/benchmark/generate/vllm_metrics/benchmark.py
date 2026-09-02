"""Scrape vLLM /metrics endpoint and collect server-side indicators."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from luban_meter.benchmark.generate.common.device_monitor import (
    print_hardware_info,
)
from luban_meter.benchmark.generate.common.prometheus import (
    parse_prometheus_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM /metrics scraper"
    )
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


def request_headers(api_key: str) -> dict[str, str]:
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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


def scrape_metrics(
    service_url: str,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    url = f"{service_url.rstrip('/')}/metrics"
    http_request = urllib.request.Request(
        url, headers=request_headers(api_key)
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GET /metrics failed with HTTP {exc.code}: {detail}"
        ) from exc
    return parse_prometheus_text(body)


def run_benchmark(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    print_hardware_info()
    service_url = string_value(
        parameters, "service_url", "http://127.0.0.1:8000"
    ).rstrip("/")
    if not service_url:
        raise ValueError("service_url must not be empty")

    api_key = string_value(parameters, "api_key", "")
    timeout = positive_number(parameters, "request_timeout", 10)
    collect_interval = positive_number(parameters, "collect_interval", 1.0)
    collect_duration = positive_number(parameters, "collect_duration", 60.0)

    model = request.get("model_name")
    if not isinstance(model, str) or not model:
        model = discover_model(service_url, api_key, timeout)

    snapshots: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    snapshot_index = 0

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= collect_duration:
            break

        snapshot_start = time.perf_counter()
        try:
            metrics = scrape_metrics(service_url, api_key, timeout)
            error = None
        except Exception as exc:  # noqa: BLE001
            metrics = {}
            error = {"type": type(exc).__name__, "message": str(exc)}

        snapshots.append(
            {
                "index": snapshot_index,
                "elapsed_seconds": round(elapsed, 6),
                "metrics": metrics,
                "error": error,
            }
        )
        snapshot_index += 1

        remaining = collect_interval - (time.perf_counter() - snapshot_start)
        if remaining > 0:
            time.sleep(remaining)

    total_elapsed = round(time.perf_counter() - start_time, 6)
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metadata": {
            "measurement": "vllm_metrics_scrape",
            "service_url": service_url,
            "model": model,
            "collect_interval": collect_interval,
            "collect_duration": collect_duration,
            "snapshot_count": len(snapshots),
            "total_elapsed_seconds": total_elapsed,
        },
        "metrics": {
            "snapshots": snapshots,
        },
    }


def main() -> None:
    args = parse_args()
    request_data, parameters = load_request(args.request)
    result = run_benchmark(request_data, parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()