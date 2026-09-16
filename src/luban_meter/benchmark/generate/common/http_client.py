"""HTTP client helpers for generate benchmarks."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any


def request_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(
    url: str,
    payload: Mapping[str, Any],
    api_key: str,
    timeout: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout
        ) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST {url} failed with HTTP {exc.code}: {detail}"
        ) from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"POST {url} returned a non-object response")
    return value


def discover_model(
    service_url: str, api_key: str, timeout: float
) -> str:
    url = f"{service_url.rstrip('/')}/v1/models"
    request = urllib.request.Request(
        url, headers=request_headers(api_key)
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout
        ) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"model discovery failed with HTTP {exc.code}: {detail}"
        ) from exc
    models = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(models, list) or not models:
        raise RuntimeError(
            f"serving endpoint returned no models from {url}"
        )
    model = models[0].get("id") if isinstance(models[0], Mapping) else None
    if not isinstance(model, str) or not model:
        raise RuntimeError(
            f"serving endpoint returned an invalid model from {url}"
        )
    return model
