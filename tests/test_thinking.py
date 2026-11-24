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


@pytest.mark.parametrize("delta", [
    {"content": "partial"},
    {"reasoning_content": "partial"},
    {"tool_calls": [{"function": {"name": "lookup"}}]},
])
def test_chat_error_after_output_is_rejected(delta):
    wire = (
        sse(delta)
        + sse(error={"message": "synthetic backend failure"})
        + b"data: [DONE]\n"
    )
    with pytest.raises(RuntimeError, match="stream reported an error"):
        collect_chat_stream(io.BytesIO(wire))


@pytest.mark.parametrize("finish_reason", [None, "stop"])
@pytest.mark.parametrize("delta", [
    {"content": "partial"},
    {"reasoning_content": "partial"},
    {"tool_calls": [{"function": {"name": "lookup"}}]},
])
def test_chat_eof_without_done_is_rejected(delta, finish_reason):
    wire = sse(delta)
    if finish_reason:
        wire += (
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
        )
    with pytest.raises(RuntimeError, match=r"stream ended before \[DONE\]"):
        collect_chat_stream(io.BytesIO(wire))


@pytest.mark.parametrize("failure", ["error", "eof"])
def test_incomplete_chat_streams_fail_single_and_multi_workers(
    tmp_path, monkeypatch, failure,
):
    monkeypatch.setenv("NO_PROXY", "*")
    posts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            payload = json.loads(self.rfile.read(
                int(self.headers["Content-Length"])
            ))
            posts.append(payload)
            prompt = payload["messages"][-1]["content"]
            wire = sse({"content": "private-response"})
            if prompt == "private-prompt-ok":
                wire += (
                    b'data: {"choices":[{"delta":{},'
                    b'"finish_reason":"stop"}]}\n\n'
                    + sse(usage={
                        "prompt_tokens": 2, "completion_tokens": 2,
                    })
                    + b"data: [DONE]\n\n"
                )
            elif failure == "error":
                wire += sse(error={"message": "private-error-body"})
                wire += b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    single = tmp_path / "single.jsonl"
    single.write_text("\n".join(json.dumps({"prompt": prompt}) for prompt in (
        "private-prompt-ok", "private-prompt-bad",
    )))
    multi = tmp_path / "multi.json"
    multi.write_text(json.dumps([{"conversations": [
        {"from": "human", "value": "private-prompt-bad"},
        {"from": "gpt", "value": "reference"},
        {"from": "human", "value": "must-not-be-sent"},
        {"from": "gpt", "value": "reference"},
    ]}]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    common = {
        "model": "mock", "base_url": f"http://127.0.0.1:{server.server_port}",
        "endpoint": "chat", "profile": "synchronous", "num_warmups": 0,
        "output_tokens": 2, "ignore_eos": False, "thinking_metrics": True,
        "api_key": "review-private-key", "request_timeout": 3,
        "task_timeout": 30, "goodput": ["ttft:1000000"],
    }
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"tasks": [
        {**common, "name": "single", "scenario": "online_serving",
         "dataset_path": str(single), "data": "jsonl", "num_requests": 2},
        {**common, "name": "multi", "scenario": "multi_turn_serving",
         "dataset_path": str(multi), "multi_num_sessions": 1,
         "multi_max_turns": 2},
    ]}))
    try:
        result, directory = RunCoordinator().run(
            build_plan({}, config), tmp_path / "runs",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    single_case, multi_case = [task["cases"][0] for task in result["tasks"]]
    assert single_case["status"] == "partial_failed"
    assert multi_case["status"] == "failed"
    values = single_case["metrics"]["streaming_performance"]["values"]
    service = values["service_view"]
    assert service["successful_requests"]["value"] == 1
    assert service["failed_requests"]["value"] == 1
    assert service["goodput"]["failed_count"]["value"] == 1
    assert service["goodput"]["slo_satisfied_rate"]["value"] == 0.5
    assert values["request_view"]["ttft"]["count"] == 1
    for name in ("single", "multi"):
        raw = json.loads((
            directory / "tasks" / name / "raw" / "case_1.json"
        ).read_text())
        response = raw["requests"][-1]["response"]
        assert response["status"] == "failed"
        assert "ttft_ms" not in response
        assert "content_event_offsets_ms" not in response
        assert response["phase_timing"]["status"] == "failed"
    session = raw["samples"][0]["facts"]
    assert session["abort_reason"] == "turn_failed"
    assert session["completed_turns"] == 0
    assert session["sent_failed_turns"] == 1
    assert session["not_started_turns"] == 1
    assert len(posts) == 3
    logs = list((directory / "log").glob("*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "Worker started:" in text
    assert "Case result saved" in text
    assert "task=single case=case_1" in text
    assert "task=multi case=case_1" in text
    assert "Streaming request failed:" in text
    assert "Traceback (most recent call last)" in text
    assert ("stream reported an error" if failure == "error" else
            "stream ended before [DONE]") in text
    for private in ("review-private-key", "private-prompt-ok",
                    "private-prompt-bad", "private-response",
                    "private-error-body"):
        assert private not in text


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
