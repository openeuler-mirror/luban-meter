"""Thinking transports retain evidence and keep scoring on final content."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from luban_meter.benchmarking.generation_performance import online_serving
from luban_meter.benchmarking.generation_performance.common.streaming import (
    collect_chat_stream,
)
from luban_meter.core.planning import build_plan
from luban_meter.core.run_coordinator import RunCoordinator


def sse(delta=None, **fields):
    choices = [{"delta": delta}] if delta is not None else []
    return (
        "data: " + json.dumps({"choices": choices, **fields}) + "\n\n"
    ).encode()


@pytest.mark.parametrize("alias", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize("content", ["", "答案：B"])
def test_stream_reasoning_timeline_and_token_denominator(alias, content):
    chunks = [sse({"role": "assistant", "content": ""})]
    chunks += [sse({alias: "思考"}), sse({alias: "继续"})]
    if content:
        chunks.append(sse({"content": content}))
    chunks += [
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n',
        sse(usage={
            "prompt_tokens": 10, "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 6},
        }),
        b'data: [DONE]\n',
    ]
    times = iter((11.0, 12.0, 14.0))
    observation = collect_chat_stream(
        io.BytesIO(b"".join(chunks)), clock=lambda: next(times)
    )
    record = online_serving._build_request_record(
        request_index=0, started=10.0, ended=15.0,
        benchmark_start=10.0, scheduled_time=10.0,
        observation=observation, error=None,
    )
    assert record["status"] == "success"
    assert record["ttft_ms"] == 1000
    assert record["reasoning_text"] == "思考继续"
    assert record["generated_text"] == content
    assert record["has_content"] is bool(content)
    assert record["first_content_latency_ms"] == (4000 if content else None)
    assert record["reasoning_event_offsets_ms"] == [1000, 2000]
    assert record["content_event_offsets_ms"] == ([4000] if content else [])
    assert record["output_tokens"] == 8
    assert record["reasoning_tokens"] == 6
    assert record["finish_reason"] == "length"
    assert online_serving.guidellm_token_latencies(record) == (
        (4000 if content else 2000) / 8,
        (3000 if content else 1000) / 7,
    )


def test_one_event_with_both_channels_and_duplicate_aliases_counts_once():
    observation = collect_chat_stream(io.BytesIO(
        sse({"reasoning": "a", "reasoning_content": "a", "content": "b"})
        + b'data: [DONE]\n'
    ), clock=lambda: 11.0)
    assert observation.event_times == (11.0,)
    assert observation.content_event_times == (11.0,)
    assert observation.reasoning_event_times == (11.0,)
    assert observation.reasoning_text == "a"
    assert observation.generated_text == "b"
    assert observation.reasoning_tokens is None
    assert observation.output_tokens is None


@pytest.mark.parametrize("reported", [
    {"queue_time_ms": 12.3},
    {"queue_time_ms": 0},
    {"queue_time_ms": 0.0004},
])
def test_stream_records_per_request_queue_time(reported):
    # vLLM sends metrics together with the final usage chunk.
    chunks = [
        sse({"content": "答案：B"}),
        sse(usage={"prompt_tokens": 10, "completion_tokens": 8},
            metrics=reported),
        b'data: [DONE]\n',
    ]
    observation = collect_chat_stream(io.BytesIO(b"".join(chunks)))
    record = online_serving._build_request_record(
        request_index=0, started=10.0, ended=15.0,
        benchmark_start=10.0, scheduled_time=10.0,
        observation=observation, error=None,
    )
    assert record["server_queue_time_ms"] == float(reported["queue_time_ms"])


@pytest.mark.parametrize("reported", [
    None,
    {"queue_time_ms": None},
    {"queue_time_ms": "12.3"},
    {"queue_time_ms": True},
    {"queue_time_ms": -1.0},
    {"other_field": 5},
    # Non-finite timings are unusable, not zero waits; the JSON number
    # 1e309 parses to float infinity.
    {"queue_time_ms": 1e309},
    {"queue_time_ms": float("inf")},
    {"queue_time_ms": float("nan")},
])
def test_stream_without_valid_queue_time_keeps_none(reported):
    chunks = [
        sse({"content": "答案：B"}),
        sse(usage={"prompt_tokens": 10, "completion_tokens": 8},
            metrics=reported),
        b'data: [DONE]\n',
    ]
    observation = collect_chat_stream(io.BytesIO(b"".join(chunks)))
    assert observation.server_queue_time_ms is None


@pytest.mark.parametrize("body, message", [
    (sse({"reasoning": "a", "reasoning_content": "b"}), "conflicting"),
    (sse({"reasoning": 5}), "reasoning must be"),
    (sse({"content": ""}), "first non-empty"),
    (sse({"reasoning_content": ""}), "first non-empty"),
    (sse({"reasoning": "x"}, usage={
        "prompt_tokens": 1, "completion_tokens": 1,
        "completion_tokens_details": {"reasoning_tokens": -1},
    }), "reasoning_tokens"),
])
def test_stream_invalid_and_empty_evidence(body, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        collect_chat_stream(io.BytesIO(body + b'data: [DONE]\n'))


@pytest.mark.parametrize("only_reasoning", [False, True])
@pytest.mark.parametrize("controls", [
    {},
    {"request_params": {"chat_template_kwargs": {"enable_thinking": True}}},
    {"request_params": {"chat_template_kwargs": {"enable_thinking": False}}},
    {"request_params": {"thinking": {"type": "disabled"}}},
    {"request_params": {"top_p": 0.9, "custom": {"items": [True, None, 1]}},
     "stop": ["END", "\n"]},
])
def test_real_subprocess_thinking_tasks(tmp_path, only_reasoning, controls):
    posts = []
    content = "" if only_reasoning else "答案：B"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(
                int(self.headers["Content-Length"])
            ))
            posts.append(body)
            usage = {
                "prompt_tokens": 10, "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 6},
            }
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(
                    sse({"reasoning": "答案：A"})
                    + sse({"content": content})
                    + b'data: {"choices":[{"delta":{},'
                    b'"finish_reason":"stop"}]}\n'
                    + sse(usage=usage) + b'data: [DONE]\n'
                )
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "choices": [{
                        "message": {
                            "reasoning": "答案：A", "content": content,
                        },
                        "finish_reason": "stop",
                    }], "usage": usage,
                }).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({
        "prompt": "hello", "question": "Q", "choices": ["a", "b", "c", "d"],
        "answer": "B",
    }) + '\n')
    common = {
        "model": "mock", "endpoint": "chat",
        "base_url": f"http://127.0.0.1:{server.server_port}",
        "dataset_path": str(dataset), "data": "jsonl",
        "num_requests": 1, "profile": "concurrent", "max_concurrency": 1,
        **controls,
    }
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"tasks": [
        {**common, "name": "perf", "scenario": "online_serving",
         "num_warmups": 1},
        {**common, "name": "quality", "scenario": "cmmlu", "few_shot": 0},
    ]}))
    try:
        result, directory = RunCoordinator().run(
            build_plan({}, task_file), tmp_path / "runs"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result["status"] == "success", result
    assert len(posts) == 3  # One warmup, one performance and one quality.
    for payload in posts:
        assert payload["stop"] == controls.get("stop", [])
        extra = controls.get("request_params", {})
        for key, value in extra.items():
            assert payload[key] == value
        for key in ("thinking", "chat_template_kwargs"):
            assert (key in payload) is (key in extra)
    assert [p["stream"] for p in posts] == [True, True, False]
    for task in ("perf", "quality"):
        raw = json.loads((
            directory / "tasks" / task / "raw/case_1.json"
        ).read_text())
        facts = raw["samples"][0]["facts"]
        assert facts["reasoning_text"] == "答案：A"
        assert facts["reasoning_tokens"] == 6
        assert facts["output_tokens"] == 8
        assert facts["has_content"] is not only_reasoning
        if task == "perf":
            assert facts["status"] == "success"
            missing_content_time = facts["first_content_latency_ms"] is None
            assert missing_content_time is only_reasoning
        else:
            assert facts["raw_output"] == content
            assert facts["prediction"] == (None if only_reasoning else "B")
            assert facts["status"] == (
                "parse_failed" if only_reasoning else "success"
            )
