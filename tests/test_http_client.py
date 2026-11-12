"""Transport changes preserve request facts and parameter contracts."""

import json
from types import SimpleNamespace

import httpx
import pytest

from luban_meter.benchmarking.generation_performance import online_serving
from luban_meter.benchmarking.generation_performance.common.streaming import (
    collect_chat_stream,
    collect_completion_stream,
)
from luban_meter.benchmarking.model_service_quality.common import client
from luban_meter.core.planning import build_plan
from luban_meter.runtime import http_client
from luban_meter.runtime.load_profiles import Profile
from luban_meter.runtime.request_scheduler import RequestScheduler
from luban_meter.runtime.scheduling_strategies import SynchronousStrategy


def online_parameters(**overrides):
    return build_plan({
        "scenario": "online_serving",
        "model": "mock",
        "profile": "concurrent",
        "max_concurrency": 4,
        **overrides,
    })["tasks"][0]["cases"][0]["parameters"]


@pytest.mark.parametrize("limit", [None, 4, 128])
def test_http_defaults_and_explicit_limits(limit):
    parameters = online_parameters(http_max_connections=limit)
    settings = http_client.http_settings(parameters)
    assert settings["max_connections"] == (limit or 4)
    assert settings["max_keepalive_connections"] == (limit or 4)
    assert settings["keepalive_expiry"] == 5.0
    assert settings["http2"] is False


@pytest.mark.parametrize("scenario", ["online_serving", "gsm8k"])
@pytest.mark.parametrize(
    "name, value",
    [
        ("http_max_connections", 0),
        ("http_max_connections", -1),
        ("http_max_connections", True),
        ("http_max_connections", 2.5),
        ("http_max_connections", 3),
        ("http_keepalive_expiry", 0),
        ("http_keepalive_expiry", -1),
        ("http_keepalive_expiry", True),
        ("http_keepalive_expiry", None),
        ("http_keepalive_expiry", float("inf")),
        ("http_keepalive_expiry", float("nan")),
    ],
)
def test_http_invalid_parameters_rejected_before_run(
    tmp_path, scenario, name, value
):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n')
    with pytest.raises(ValueError, match=name):
        online_parameters(
            scenario=scenario, dataset_path=str(dataset), **{name: value}
        )


def test_http_limit_uses_strategy_capacity_not_worker_count(monkeypatch):
    class LimitedProfile(Profile):
        def create_strategy(self):
            return SynchronousStrategy()

    parameters = {"profile": "limited", "max_concurrency": 8}
    monkeypatch.setattr(
        http_client, "profiles", SimpleNamespace(get=lambda _: LimitedProfile)
    )
    settings = http_client.http_settings(parameters)
    scheduler = RequestScheduler(
        LimitedProfile(parameters), parameters, lambda: False, lambda *a: None
    )
    assert settings["max_connections"] == scheduler.capacity == 1
    assert http_client.http_settings({
        **parameters, "http_max_connections": 1,
    })["max_connections"] == 1


class TimedStream(httpx.SyncByteStream):
    def __init__(self, clock, chunks, failure=None):
        self.clock = clock
        self.chunks = chunks
        self.failure = failure
        self.closed = False

    def __iter__(self):
        for timestamp, chunk in self.chunks:
            self.clock.now = timestamp
            yield chunk
        if self.failure:
            raise self.failure

    def close(self):
        self.clock.now = 30.0
        self.closed = True


def run_stream(monkeypatch, chunks, *, collector=None, failure=None):
    clock = SimpleNamespace(now=10.0)
    clock.perf_counter = lambda: clock.now
    monkeypatch.setattr(online_serving, "time", clock)
    stream = TimedStream(clock, chunks, failure)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream)
    )
    if collector is None:
        collector = collect_completion_stream
    with httpx.Client(transport=transport) as http:
        build_request = http.build_request

        def build(*args, **kwargs):
            # Request construction belongs before the timer starts.
            clock.now = 20.0
            return build_request(*args, **kwargs)

        monkeypatch.setattr(http, "build_request", build)
        record = online_serving.execute_request(
            client=http, request_index=0, payload={},
            endpoint="completion", service_url="http://mock",
            api_key="", timeout=1.0, benchmark_start=0.0,
            scheduled_time=19.0, tracker=online_serving.ActiveRequestTracker(),
            stream_collector=lambda lines: collector(
                lines, clock=clock.perf_counter
            ),
        )
    assert stream.closed
    return record


@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("line_ending", [b"\n", b"\r", b"\r\n"])
def test_stream_timestamps_exclude_drain_and_close(
    monkeypatch, cleanup_failure, line_ending
):
    record = run_stream(
        monkeypatch,
        [
            (timestamp, chunk.replace(b"\n", line_ending))
            for timestamp, chunk in [
                (20.5, b': keepalive\n\ndata: {"choices":[{"text":""}]}\n'),
                (21.0, b'data: {"choices":[{"text":"A"}]}\n'),
                (22.0, b'data: {"choices":[{"text":"BC"}]}\n'),
                (23.0, b'data: {"choices":[],"usage":'
                 b'{"prompt_tokens":4,"completion_tokens":3}}\n'),
                (24.0, b'data: [DONE]\n'),
                (29.0, b'data: {"choices":[{"text":"ignored"}]}\n'),
            ]
        ],
        failure=httpx.ReadTimeout("cleanup") if cleanup_failure else None,
    )
    assert record["status"] == "success"
    assert record["submitted_at_monotonic_s"] == 20.0
    assert record["dispatch_delay_ms"] == 1000.0
    assert record["ttft_ms"] == 1000.0
    assert record["last_output_latency_ms"] == 2000.0
    assert record["e2el_ms"] == record["duration_ms"] == 4000.0
    assert record["scheduled_latency_ms"] == 5000.0
    assert record["itl_samples_ms"] == [1000.0]
    assert record["output_tokens"] == 3
    assert record["stream_event_count"] == 2
    assert online_serving.guidellm_token_latencies(record) == (
        2000 / 3, 500.0,
    )
    goodput = online_serving.compute_goodput(
        successful=[record], failed_count=0, duration_seconds=10,
        slo_config={"ttft_ms": 1000, "tpot_ms": 500, "e2el_ms": 4000},
    )
    assert goodput["slo_satisfied_count"]["value"] == 1
    if cleanup_failure:
        assert record["connection_cleanup_error"]["type"] == "ReadTimeout"
    else:
        assert "connection_cleanup_error" not in record


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("line_ending", ["\n", "\r", "\r\n"])
def test_utf8_fragmentation_and_line_boundaries(
    monkeypatch, chat, line_ending
):
    choice = {"delta": {"content": "你好"}} if chat else {"text": "你好"}
    body = (
        ': heartbeat' + line_ending * 2 + 'data: '
        + json.dumps({"choices": [choice]}, ensure_ascii=False)
        + line_ending * 2 + 'data: [DONE]' + line_ending * 2
    ).encode()
    record = run_stream(
        monkeypatch,
        [(21.0, bytes([byte])) for byte in body],
        collector=collect_chat_stream if chat else collect_completion_stream,
    )
    assert record["status"] == "success"
    assert record["generated_text"] == "你好"
    assert record["stream_event_count"] == 1
    assert record["ttft_ms"] == 1000.0
    assert record["input_tokens"] is None
    assert record["output_tokens"] is None


@pytest.mark.parametrize(
    "body, failure, error_type",
    [
        (b'data: {"choices":[{"text":"\xff"}]}\n', None,
         "UnicodeDecodeError"),
        (b'data: broken\n', None, "JSONDecodeError"),
        (b'data: [DONE]\n', None, "RuntimeError"),
        (b': heartbeat\n', httpx.ReadTimeout("before output"), "ReadTimeout"),
    ],
)
def test_stream_failures_release_response(
    monkeypatch, body, failure, error_type
):
    record = run_stream(monkeypatch, [(21.0, body)], failure=failure)
    assert record["status"] == "failed"
    assert record["error"]["type"] == error_type
    assert "ttft_ms" not in record


def test_quality_latency_includes_json_decode(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    clock.perf_counter = lambda: clock.now
    monkeypatch.setattr(client, "time", clock)
    stream = TimedStream(clock, [(21.0, b'{"choices":[]}')])
    http = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream)
    ))
    monkeypatch.setattr(client, "create_http_client", lambda *a, **k: http)
    service = client.OpenAIClient("http://mock")
    decode = json.loads

    def loads(value):
        clock.now = 31.0
        return decode(value)

    monkeypatch.setattr(client, "json", SimpleNamespace(
        dumps=json.dumps, loads=loads,
        JSONDecodeError=json.JSONDecodeError,
    ))
    try:
        body, latency = service._post_direct("/v1/completions", {})
        assert body == {"choices": []}
        assert latency == 21000.0
        assert stream.closed
    finally:
        service.close()


@pytest.mark.parametrize("separator", [b"\n", b"\r", b"\r\n"])
def test_response_lines_preserve_bytes_across_all_split_positions(separator):
    text = "你好\u2028世界".encode()
    body = separator.join([text, b"", b"last", b"tail"])
    expected = [text + b"\n", b"\n", b"last\n", b"tail"]
    for split in range(len(body) + 1):
        response = SimpleNamespace(
            iter_raw=lambda: iter([body[:split], b"", body[split:]])
        )
        assert list(http_client.iter_response_lines(response)) == expected


def test_response_lines_yield_at_cr_without_reading_next_chunk():
    def chunks():
        yield b"data: first\r"
        pytest.fail("reader waited for the next chunk after a complete line")

    response = SimpleNamespace(iter_raw=chunks)
    lines = http_client.iter_response_lines(response)
    assert next(lines) == b"data: first\n"
    lines.close()


def completion_usage_event(usage):
    return (
        "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
    ).encode()


@pytest.mark.parametrize("details, state", [
    (None, "missing"),
    ({}, "missing"),
    ({"cached_tokens": None}, "missing"),
    ({"cached_tokens": 0}, "valid"),
    ({"cached_tokens": 2}, "valid"),
    ({"cached_tokens": -1}, "invalid"),
    ({"cached_tokens": True}, "invalid"),
    ({"cached_tokens": 1.5}, "invalid"),
    ({"cached_tokens": 5}, "invalid"),
])
def test_completion_usage_tail_preserves_evidence_and_timing(
    monkeypatch, details, state
):
    from luban_meter.benchmarking.generation_performance.common import (
        cache_usage,
    )

    usage = {
        "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7,
        "prompt_tokens_details": details,
        "completion_tokens_details": {"vendor_count": 1},
        "vendor": {"items": [1, None, "unchanged"]},
    }
    record = run_stream(monkeypatch, [
        (21.0, b'data: {"choices":[{"text":"A"}]}\n'),
        (22.0, b'data: {"choices":[{"text":"BC"}]}\n'),
        (22.5, b'data: {"choices":[{"text":"",'
         b'"finish_reason":"length"}]}\n'),
        (23.0, completion_usage_event(usage)),
        (24.0, b'data: [DONE]\n'),
        (29.0, completion_usage_event({
            "prompt_tokens": 999, "completion_tokens": 999,
        })),
    ])
    assert record["status"] == "success"
    assert record["usage"] == usage
    assert json.loads(json.dumps(record))["usage"] == usage
    assert record["finish_reason"] == "length"
    assert cache_usage.parse_cache_usage(record["usage"]).state == state
    assert record["input_tokens"] == 4
    assert record["output_tokens"] == 3
    assert record["generated_text"] == "ABC"
    assert record["ttft_ms"] == 1000
    assert record["last_output_latency_ms"] == 2000
    assert record["itl_samples_ms"] == [1000]
    assert record["stream_event_count"] == 2
    assert record["e2el_ms"] == record["duration_ms"] == 4000
    assert online_serving.guidellm_token_latencies(record) == (
        2000 / 3, 500,
    )


@pytest.mark.parametrize("final_usage", [
    {"prompt_tokens": 8, "completion_tokens": 3},
    {"prompt_tokens": 8, "completion_tokens": 3,
     "prompt_tokens_details": None},
    {"prompt_tokens": 8, "completion_tokens": 3,
     "prompt_tokens_details": {"cached_tokens": 0}},
])
def test_completion_final_usage_replaces_instead_of_merging_or_summing(
    monkeypatch, final_usage
):
    record = run_stream(monkeypatch, [
        (21.0, b'data: {"choices":[{"text":"A"}]}\n'),
        (22.0, completion_usage_event({
            "prompt_tokens": 4, "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
            "earlier_only": True,
        })),
        (23.0, completion_usage_event(final_usage)),
        (23.5, completion_usage_event(final_usage)),
        (23.75, completion_usage_event(None)),
        (24.0, b'data: [DONE]\n'),
    ])
    assert record["status"] == "success"
    assert record["usage"] == final_usage
    assert record["input_tokens"] == 8
    assert record["output_tokens"] == 3
    assert record["stream_event_count"] == 1
    assert record["e2el_ms"] == 4000


@pytest.mark.parametrize("bad_usage", [
    {},
    {"prompt_tokens": True, "completion_tokens": 3},
    {"prompt_tokens": 4, "completion_tokens": -1},
    [],
])
def test_completion_invalid_final_usage_does_not_fall_back(
    monkeypatch, bad_usage
):
    record = run_stream(monkeypatch, [
        (21.0, b'data: {"choices":[{"text":"A"}]}\n'),
        (22.0, completion_usage_event({
            "prompt_tokens": 4, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 0},
        })),
        (23.0, completion_usage_event(bad_usage)),
        (24.0, b'data: [DONE]\n'),
    ])
    assert record["status"] == "failed"
    assert "usage" not in record
    assert "input_tokens" not in record


@pytest.mark.parametrize("tail, failure, error_type", [
    (b"", None, "RuntimeError"),
    (b'data: {"choices":[{"text":"","finish_reason":"stop"}]}\n',
     None, "RuntimeError"),
    (b'data: {"usage":', None, "JSONDecodeError"),
    (b"", httpx.ReadTimeout("before DONE"), "ReadTimeout"),
    (b'data: {"error":{"message":"failed"}}\ndata: [DONE]\n',
     None, "RuntimeError"),
])
def test_completion_incomplete_response_never_exposes_partial_usage(
    monkeypatch, tail, failure, error_type
):
    record = run_stream(monkeypatch, [
        (21.0, b'data: {"choices":[{"text":"A"}]}\n'),
        (22.0, completion_usage_event({
            "prompt_tokens": 4, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2},
        })),
        (23.0, tail),
    ], failure=failure)
    assert record["status"] == "failed"
    assert record["error"]["type"] == error_type
    assert "usage" not in record
    assert "ttft_ms" not in record
