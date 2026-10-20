"""Case-scoped HTTP/1.1 clients and byte-preserving responses."""

from __future__ import annotations

import math
import ssl
import sys
from http.cookiejar import CookieJar, DefaultCookiePolicy

import httpx

from luban_meter.runtime.load_profiles import profiles
from luban_meter.runtime.request_scheduler import request_capacity


def http_settings(parameters):
    """Validate pool limits against formal admission without I/O."""
    profile = profiles.get(parameters["profile"])(parameters)
    capacity = request_capacity(
        profile.workers, profile.create_strategy().concurrency_limit
    )
    maximum = parameters.get("http_max_connections")
    if maximum is None:
        maximum = capacity
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < capacity
    ):
        raise ValueError(
            "http_max_connections must be an integer >= effective "
            f"request capacity ({capacity})"
        )
    expiry = parameters.get("http_keepalive_expiry", 5.0)
    if (
        isinstance(expiry, bool)
        or not isinstance(expiry, (int, float))
        or not math.isfinite(expiry)
        or expiry <= 0
    ):
        raise ValueError("http_keepalive_expiry must be finite and positive")
    return {
        "client": "httpx",
        "client_version": httpx.__version__,
        "http2": False,
        "max_connections": maximum,
        "max_keepalive_connections": maximum,
        "keepalive_expiry": expiry,
    }


def _check_redirect(response):
    # urllib does not replay POST bodies on 307/308 redirects.
    if (
        response.status_code in (307, 308)
        and response.request.method not in ("GET", "HEAD")
    ):
        response.read()
        response.raise_for_status()


def create_http_client(timeout, *, max_connections, keepalive_expiry=5.0):
    """Create before threads start; close after their requests finish.

    Retain system certificate trust and environment proxy discovery.
    Disable cookie persistence and compression negotiation so pooling
    does not introduce a session or change the response representation.
    HTTPX's default transport does not retry failed requests.
    """
    return httpx.Client(
        http1=True,
        http2=False,
        timeout=httpx.Timeout(timeout),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=keepalive_expiry,
        ),
        verify=ssl.create_default_context(),
        trust_env=True,
        follow_redirects=True,
        max_redirects=10,
        cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": f"Python-urllib/{sys.version_info.major}."
            f"{sys.version_info.minor}",
        },
        event_hooks={"response": [_check_redirect]},
    )


def iter_response_lines(response):
    """Yield byte lines preserving strict UTF-8 and SSE semantics.

    Keep the iterator alive after the collector returns at [DONE], then
    exhaust it to finish the HTTP message and release the connection.
    No fixed chunk size is used: it could buffer output and distort TTFT.
    """
    pending = bytearray()
    for chunk in response.iter_raw():
        parts = chunk.split(b"\n")
        pending.extend(parts[0])
        for part in parts[1:]:
            yield bytes(pending) + b"\n"
            pending.clear()
            pending.extend(part)
    if pending:
        yield bytes(pending)
