"""OpenAI-compatible HTTP client for model service quality benchmarking."""

from __future__ import annotations

import json
import time
from contextlib import closing
from typing import Any

import httpx

from luban_meter.runtime.http_client import create_http_client
from luban_meter.runtime.request_params import (
    append_request_params,
    validate_request_params,
)
from luban_meter.runtime.thinking import (
    reasoning_text,
    reasoning_token_count,
)


class ServiceError(RuntimeError):
    """Raised when the model service rejects or fails a request."""


class OpenAIClient:
    """Minimal OpenAI-compatible client."""

    def __init__(
        self,
        service_url: str,
        served_model_name: str = "",
        api_key: str = "",
        timeout: float = 60.0,
        request_runner=None,
        *,
        max_connections: int = 100,
        keepalive_expiry: float = 5.0,
        request_params: dict[str, Any] | None = None,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.request_params = validate_request_params(
            {} if request_params is None else request_params
        )
        self.served_model_name = served_model_name
        self.api_key = api_key
        self.timeout = timeout
        self.request_runner = request_runner
        self.http = create_http_client(
            timeout,
            max_connections=max_connections,
            keepalive_expiry=keepalive_expiry,
        )

    def close(self) -> None:
        self.http.close()

    def _post(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        if path in ("/v1/chat/completions", "/v1/completions"):
            payload = append_request_params(payload, self.request_params)
        if self.request_runner is not None and "completions" in path:
            def send(ticket):
                body, elapsed = self._post_direct(path, payload)
                usage = body.get("usage")
                if hasattr(ticket, "usage") and isinstance(usage, dict):
                    ticket.usage.update({
                        key: usage.get(key)
                        for key in ("prompt_tokens", "completion_tokens")
                    })
                return body, elapsed

            return self.request_runner(send)
        return self._post_direct(path, payload)

    def _post_direct(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = self.http.build_request(
            "POST",
            f"{self.service_url}{path}",
            content=data,
            headers=headers,
        )
        start = time.perf_counter()
        try:
            with closing(self.http.send(request, stream=True)) as response:
                content = response.read()
                response.raise_for_status()
                body = json.loads(content.decode("utf-8"))
        except httpx.HTTPStatusError as exc:
            detail = exc.response.read().decode("utf-8", "replace")[:500]
            raise ServiceError(
                f"HTTP {exc.response.status_code} from {path}: {detail}"
            ) from exc
        except (
            httpx.HTTPError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
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
        """Call /v1/chat/completions; the service applies the chat
        template.
        """
        payload: dict[str, Any] = {
            "model": self.served_model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload["stop"] = [] if stop is None else stop
        body, elapsed_ms = self._post("/v1/chat/completions", payload)
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ServiceError("chat completion returned no choices")
        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}
        try:
            thought = reasoning_text(message)
            thought_tokens = reasoning_token_count(usage)
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        return {
            "reasoning_text": thought,
            "reasoning_tokens": thought_tokens,
            "has_content": bool(message.get("content")),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": usage,
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
            "model": self.served_model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload["stop"] = [] if stop is None else stop
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
        """Call /v1/completions with echo+logprobs to score prompt
        tokens.
        """
        payload = {
            "model": self.served_model_name,
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
        body, _ = self._post(
            "/tokenize", {"model": self.served_model_name, "prompt": text}
        )
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ServiceError("tokenize returned no tokens")
        return list(tokens)

    def discover_served_model_name(self) -> str:
        """Fetch the first served model name from /v1/models."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = self.http.build_request(
            "GET",
            f"{self.service_url}/v1/models",
            headers=headers,
        )
        try:
            with closing(self.http.send(request, stream=True)) as response:
                content = response.read()
                response.raise_for_status()
                body = json.loads(content.decode("utf-8"))
        except (
            httpx.HTTPError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise ServiceError(f"request to /v1/models failed: {exc}") from exc
        models = body.get("data") or []
        if not models or not isinstance(models[0], dict):
            raise ServiceError("service returned no models")
        served_model_name = models[0].get("id")
        if not isinstance(served_model_name, str) or not served_model_name:
            raise ServiceError("service returned no model id")
        return served_model_name
