"""Tool output affects overall timing without inventing content tokens."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from luban_meter.api import benchmarks
from luban_meter.benchmarking.generation_performance import online_serving
from luban_meter.benchmarking.generation_performance.common import (
    phase_timing,
)
from luban_meter.benchmarking.generation_performance.common.streaming import (
    collect_chat_stream,
)
from luban_meter.core.planning import build_plan
from luban_meter.core.run_coordinator import RunCoordinator


def event(delta=None, *, finish=None, usage=None):
    choice = {"delta": delta or {}}
    if finish is not None:
        choice["finish_reason"] = finish
    payload = {"choices": [choice] if delta is not None or finish else []}
    if usage is not None:
        payload["usage"] = usage
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def tool_delta(**function):
    return {"tool_calls": [{"index": 0, "function": function}]}


def record_for(*deltas, usage=None, finish="tool_calls"):
    wire = b"".join(event(delta) for delta in deltas)
    wire += event(finish=finish) + event(usage=usage) + b"data: [DONE]\n\n"
    times = iter((11.0, 12.0, 13.0))
    observation = collect_chat_stream(
        io.BytesIO(wire), clock=lambda: next(times),
    )
    return online_serving._build_request_record(
        request_index=0, started=10.0, ended=14.0,
        benchmark_start=10.0, scheduled_time=10.0,
        observation=observation, error=None,
    )


@pytest.mark.parametrize("field", ["name", "arguments"])
@pytest.mark.parametrize("text", ["lookup", "{", " ", "\n"])
def test_tool_function_output_is_timed_without_becoming_content(field, text):
    record = record_for(tool_delta(**{field: text}))
    assert record["status"] == "success"
    assert record["ttft_ms"] == 1000
    assert record["tool_call_event_offsets_ms"] == [1000]
    assert record["content_event_offsets_ms"] == []
    assert record["reasoning_event_offsets_ms"] == []
    assert record["generated_text"] == record["reasoning_text"] == ""
    assert record["has_content"] is False
    assert record["first_content_latency_ms"] is None
    assert record["input_tokens"] is record["output_tokens"] is None
    timing = phase_timing.request_phase_timing(record, thinking_metrics=True)
    assert timing["status"] == "other_finish"
    assert timing["content_event_count"] == 0
    assert timing["first_content_latency_ms"] is None


@pytest.mark.parametrize("calls", [
    None, [], {}, "tool", [None],
    [{"index": 0, "id": "call_0", "type": "function"}],
    [{"function": None}], [{"function": {}}],
    [{"function": {"name": "", "arguments": ""}}],
    [{"function": {"name": None, "arguments": None}}],
    [{"function": {"name": 1, "arguments": {"city": "Beijing"}}}],
])
def test_tool_metadata_does_not_preempt_first_content(calls):
    record = record_for({"tool_calls": calls}, {"content": "answer"})
    assert record["stream_event_count"] == 1
    assert record["tool_call_event_offsets_ms"] == []
    assert record["content_event_offsets_ms"] == [1000]
    with pytest.raises(RuntimeError, match="first non-empty"):
        record_for({"tool_calls": calls})


def test_tool_content_and_reasoning_in_one_chunk_share_one_timestamp():
    record = record_for({
        "tool_calls": [
            {"function": {"name": "lookup"}},
            {"function": {"arguments": "{"}},
        ],
        "content": "answer", "reasoning": "r",
    })
    assert record["stream_event_count"] == 1
    for key in ("output", "tool_call", "content", "reasoning"):
        assert record[key + "_event_offsets_ms"] == [1000]
    assert record["generated_text"] == "answer"
    assert record["reasoning_text"] == "r"


def test_tool_fragments_extend_overall_timing_and_keep_usage_counts():
    usage = {"prompt_tokens": 10, "completion_tokens": 8}
    record = record_for(
        tool_delta(name="lookup"), {"content": "answer"},
        {"tool_calls": [None, {"function": {"arguments": "}"}}]},
        usage=usage,
    )
    assert record["ttft_ms"] == 1000
    assert record["last_output_latency_ms"] == 3000
    assert record["first_content_latency_ms"] == 2000
    assert record["itl_samples_ms"] == [1000, 1000]
    assert record["tool_call_event_offsets_ms"] == [1000, 3000]
    assert record["content_event_offsets_ms"] == [2000]
    assert record["stream_event_count"] == 3
    assert record["output_tokens"] == 8
    assert record["usage"] == usage
    assert online_serving.guidellm_token_latencies(record) == (
        3000 / 8, 2000 / 7,
    )


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 10, "completion_tokens": 0},
    {"prompt_tokens": 10},
])
def test_tool_output_keeps_existing_usage_validation(usage):
    with pytest.raises((ValueError, RuntimeError), match="output|completion"):
        record_for(tool_delta(name="lookup"), usage=usage)


def test_real_tool_requests_workers_and_shared_log(tmp_path):
    posts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(
                int(self.headers["Content-Length"])
            ))
            posts.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if self.path == "/v1/completions":
                chunks = b'data: {"choices":[{"text":"C"}]}\n\n'
                chunks += b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
            else:
                mode = body["messages"][-1]["content"]
                chunks = event({"role": "assistant", "content": ""})
                chunks += event({"tool_calls": [{"id": "call_0"}]})
                if mode == "broken":
                    chunks += event(tool_delta(name="private-function"))
                    self.wfile.write(chunks + b"data: {invalid\n\n")
                    return
                if mode != "metadata":
                    delta = tool_delta(name="private-function")
                    if mode == "reasoning-tools":
                        delta["reasoning_content"] = "r"
                    chunks += event(delta)
                    chunks += event(tool_delta(arguments="private-arguments"))
                chunks += event(finish="tool_calls")
            chunks += event(usage={
                "prompt_tokens": 10, "completion_tokens": 8,
            }) + b"data: [DONE]\n\n"
            self.wfile.write(chunks)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    single = tmp_path / "single.jsonl"
    single.write_text("\n".join(json.dumps({"prompt": mode}) for mode in (
        "tools", "metadata", "broken",
    )))
    multi = tmp_path / "multi.json"
    multi.write_text(json.dumps([{"conversations": [
        {"from": "human", "value": "reasoning-tools"},
        {"from": "gpt", "value": "reference"},
        {"from": "human", "value": "next"},
        {"from": "gpt", "value": "reference"},
    ]}]))
    chat_endpoint = (
        benchmarks.get("multi_turn_serving").parameters["endpoint"].default
    )
    completion_endpoint = (
        benchmarks.get("online_serving").parameters["endpoint"].default
    )
    common = {
        "model": "mock", "base_url": f"http://127.0.0.1:{server.server_port}",
        "endpoint": chat_endpoint, "profile": "concurrent",
        "max_concurrency": 1, "num_warmups": 0, "api_key": "private-key",
        "thinking_metrics": True,
    }
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"tasks": [
        {**common, "name": "single", "scenario": "online_serving",
         "data": "jsonl", "dataset_path": str(single), "num_requests": 3},
        {**common, "name": "multi", "scenario": "multi_turn_serving",
         "dataset_path": str(multi), "multi_num_sessions": 1,
         "multi_max_turns": 2},
        {**common, "name": "completion", "scenario": "online_serving",
         "endpoint": completion_endpoint, "data": "jsonl",
         "dataset_path": str(single), "num_requests": 1},
    ]}))
    try:
        result, directory = RunCoordinator().run(
            build_plan({}, tasks), tmp_path / "runs",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    def raw(name):
        return json.loads((
            directory / "tasks" / name / "raw/case_1.json"
        ).read_text())

    single_raw = raw("single")
    success, metadata, broken = [
        request["response"] for request in single_raw["requests"]
    ]
    assert success["status"] == "success"
    assert success["stream_event_count"] == 2
    assert len(success["tool_call_event_offsets_ms"]) == 2
    assert success["content_event_offsets_ms"] == []
    assert success["first_content_latency_ms"] is None
    assert success["output_tokens"] == 8
    assert success["output_observation"] == "reasoning_content_and_tool_calls"
    assert success["phase_timing"]["status"] == "other_finish"
    for value in ("private-function", "private-arguments"):
        assert value not in json.dumps(success)
    assert metadata["status"] == broken["status"] == "failed"
    assert metadata["error"]["type"] == "RuntimeError"
    assert broken["error"]["type"] == "JSONDecodeError"
    assert "ttft_ms" not in metadata
    assert "ttft_ms" not in broken
    multi_raw = raw("multi")
    assert len(multi_raw["requests"]) == 1
    assert multi_raw["requests"][0]["response"]["status"] == "success"
    assert multi_raw["requests"][0]["response"]["reasoning_text"] == "r"
    session = multi_raw["samples"][0]["facts"]
    assert session["abort_reason"] == "tool_calls_only"
    assert session["sent_failed_turns"] == 1
    assert session["not_started_turns"] == 1
    completion = raw("completion")["requests"][0]["response"]
    assert completion["status"] == "success"
    assert completion["tool_call_event_offsets_ms"] == []
    assert completion["first_content_latency_ms"] == completion["ttft_ms"]
    view = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"]["values"]["request_view"]
    assert view["ttft"]["count"] == 1
    assert view["output_tokens"]["mean"] == 8
    assert len(posts) == 5
    logs = list((directory / "log").glob("luban-meter-*.log"))
    assert len(logs) == 1
    log = logs[0].read_text()
    assert "Tool-call output observed: requests=1 events=2" in log
    assert "task=single case=case_1 phase=metrics" in log
    assert "Streaming request failed: index=1" in log
    assert "Streaming request failed: index=2" in log
    assert "Traceback (most recent call last)" in log
    assert "result.json" in log
    for value in ("private-key", "private-function", "private-arguments"):
        assert value not in log
