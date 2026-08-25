"""OpenAI-compatible HTTP client for inference benchmarks."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ServiceError(RuntimeError):
    """Raised when the inference service rejects or fails a request."""


class OpenAIClient:
    """Minimal OpenAI-compatible client using only the standard library."""

    def __init__(
        self,
        service_url: str,
        model: str = "",
        api_key: str = "",
        timeout: float = 60.0,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _post(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.service_url}{path}",
            data=data,
            headers=headers,
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ServiceError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            raise ServiceError(f"request to {path} failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return body, elapsed_ms

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Call /v1/chat/completions; the service applies the chat template."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        body, elapsed_ms = self._post("/v1/chat/completions", payload)
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ServiceError("chat completion returned no choices")
        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}
        return {
            "text": message.get("content") or "",
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "latency_ms": elapsed_ms,
        }

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Call /v1/completions with plain text (base-model transport)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        body, elapsed_ms = self._post("/v1/completions", payload)
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ServiceError("completion returned no choices")
        usage = body.get("usage") or {}
        return {
            "text": choices[0].get("text") or "",
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "latency_ms": elapsed_ms,
        }

    def completion_logprobs(self, prompt: str) -> dict[str, Any]:
        """Call /v1/completions with echo+logprobs to score prompt tokens."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
            "stream": False,
        }
        body, elapsed_ms = self._post("/v1/completions", payload)
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ServiceError("completion returned no choices")
        logprobs = choices[0].get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        token_logprobs = logprobs.get("token_logprobs") or []
        if not tokens or len(tokens) != len(token_logprobs):
            raise ServiceError("completion logprobs payload is incomplete")
        usage = body.get("usage") or {}
        return {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "input_tokens": usage.get("prompt_tokens"),
            "latency_ms": elapsed_ms,
        }

    def tokenize(self, text: str) -> list[int]:
        """Call /tokenize to obtain token IDs for a text."""
        body, _ = self._post("/tokenize", {"model": self.model, "prompt": text})
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ServiceError("tokenize returned no tokens")
        return list(tokens)

    def discover_model(self) -> str:
        """Fetch the first served model name from /v1/models."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.service_url}/v1/models",
            headers=headers,
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            raise ServiceError(f"request to /v1/models failed: {exc}") from exc
        models = body.get("data") or []
        if not models or not isinstance(models[0], dict):
            raise ServiceError("service returned no models")
        model = models[0].get("id")
        if not isinstance(model, str) or not model:
            raise ServiceError("service returned no model id")
        return model
