"""Token ID supplements preserve overall timing and failures."""

import copy
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from luban_meter.benchmarking.generation_performance import online_serving
from luban_meter.benchmarking.generation_performance.common import (
    phase_timing,
    reasoning_tokens,
)
from luban_meter.benchmarking.generation_performance.common.streaming import (
    collect_chat_stream,
)
from luban_meter.core.planning import build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.reporting.html import write_html
from luban_meter.reporting.terminal import render_text
from luban_meter.utils.json_io import write_json_atomic
from tests.tokenizer_fixture import make_tokenizer

MISSING = object()


class Tokenizer:
    vocabulary = {
        "<think>": 100, "</think>": 101, "a": 11, "b": 12,
        "c": 13, "\n": 14, "Q": 16,
    }

    def get_vocab(self):
        return self.vocabulary

    def encode(self, text, **kwargs):
        return [self.vocabulary[text]]

    def decode(self, ids, **kwargs):
        inverse = {value: key for key, value in self.vocabulary.items()}
        return "".join(inverse[token] for token in ids)


def parameters(**overrides):
    return {
        "model": "mock", "endpoint": "/v1/chat/completions",
        "thinking_metrics": True, "tokenizer": "matching-model",
        "request_params": {"return_token_ids": True}, **overrides,
    }


@pytest.fixture
def counter():
    return reasoning_tokens.prepare_reasoning_counter(
        parameters(), Tokenizer(),
    )


def sse(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def wire(
    chunks=([11], [12], [101], [13]), prompt=(16, 100, 14),
    details=MISSING, output_tokens=4, input_tokens=3, finish="stop",
    conflicting_prompt=False,
):
    first = {"choices": [{"delta": {"role": "assistant"}}]}
    if prompt is not None:
        first["prompt_token_ids"] = list(prompt)
    events = [sse(first)]
    # The third event contains only the hidden </think> Token ID.
    for index, delta in enumerate([
        {"reasoning": "a"}, {"reasoning": "b"}, {}, {"content": "c"},
    ]):
        choice = {"delta": delta}
        if chunks is not None and index < len(chunks):
            choice["token_ids"] = chunks[index]
        events.append(sse({"choices": [choice]}))
    if conflicting_prompt:
        events.append(sse({"prompt_token_ids": [16], "choices": []}))
    events.append(sse({"choices": [{"delta": {}, "finish_reason": finish}]}))
    if output_tokens is not None:
        usage = {
            "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
        }
        if details is not MISSING:
            usage["completion_tokens_details"] = details
        events.append(sse({"choices": [], "usage": usage}))
    return b"".join(events) + b"data: [DONE]\n\n"


def execute(monkeypatch, data, counter, enabled=True):
    times = iter([10.0, 10.7])
    events = iter([10.1, 10.4, 10.6])
    monkeypatch.setattr(
        online_serving, "time", SimpleNamespace(perf_counter=times.__next__),
    )
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=httpx.ByteStream(data))
    )) as client:
        return online_serving.execute_request(
            client=client, request_index=3, payload={},
            endpoint="/v1/chat/completions", service_url="http://mock",
            api_key="", timeout=1, benchmark_start=10, scheduled_time=10,
            tracker=online_serving.ActiveRequestTracker(),
            stream_collector=lambda lines: collect_chat_stream(
                lines, clock=events.__next__,
            ),
            thinking_metrics=enabled, reasoning_token_counter=counter,
        )


def aggregate(records):
    return phase_timing.phase_timing_metric({
        "parameters": parameters(),
        "requests": [{"response": row} for row in records],
    })


def overall(record):
    return online_serving.process_case({
        "requests": [record], "benchmark_duration_seconds": 1,
        "maximum_request_concurrency": 1, "peak_concurrent_requests": 1,
    })


@pytest.mark.parametrize("details", [
    MISSING, None, {}, {"reasoning_tokens": None},
])
def test_missing_service_count_uses_ids_after_original_timer(
    monkeypatch, counter, details,
):
    baseline = execute(monkeypatch, wire(details=details), counter, False)
    row = execute(monkeypatch, wire(details=details), counter)
    assert row["status"] == "success"
    assert row["reasoning_tokens"] == 2
    assert row["reasoning_tokens_source"] == "token_ids"
    assert row["reasoning_tokens_reason"] is None
    assert row["reasoning_token_counting"]["starts_in_thinking"] is True
    assert row["usage"] == baseline["usage"]
    assert row["e2el_ms"] == 700
    assert row["output_event_offsets_ms"] == [100, 400, 600]
    assert row["stream_event_count"] == 3
    assert row["phase_timing"]["reasoning_tpot_ms"] == 200
    assert row["phase_timing"]["reasoning_itl_ms"] == 300
    assert overall(row) == overall(baseline)
    assert "reasoning_tokens_source" not in baseline
    metric = aggregate([row])
    assert metric["metadata"]["reasoning_token_sources"] == {"token_ids": 1}


@pytest.mark.parametrize("count", [0, 1, 4, 99])
def test_service_count_wins_over_unusable_ids(monkeypatch, counter, count):
    row = execute(monkeypatch, wire(
        details={"reasoning_tokens": count}, chunks=("bad-ids",),
    ), counter)
    assert row["status"] == "success"
    assert row["reasoning_tokens"] == count
    assert row["reasoning_tokens_source"] == "service_usage"
    assert row["reasoning_tokens_reason"] is None


@pytest.mark.parametrize("invalid_ids", [
    ["\ud800"], ["\udfff"], ["中文\ud800"], [{"\ud800": "\udfff"}],
])
@pytest.mark.parametrize("location", ["prompt", "output"])
@pytest.mark.parametrize("service_count", [None, 0, 2])
def test_unicode_id_diagnostics_preserve_utf8_facts_and_overall_metrics(
    monkeypatch, counter, tmp_path, invalid_ids, location, service_count,
):
    options = {"details": {"reasoning_tokens": service_count}}
    if location == "prompt":
        options["prompt"] = invalid_ids
    else:
        options["chunks"] = (invalid_ids,)
    data = wire(**options)
    baseline = execute(monkeypatch, data, counter, False)
    row = execute(monkeypatch, data, counter)
    assert row["status"] == "success"
    assert row["usage"] == baseline["usage"]
    assert overall(row) == overall(baseline)
    if service_count is None:
        assert row["reasoning_tokens_reason"] == (
            "invalid_prompt_token_ids" if location == "prompt"
            else "invalid_output_token_ids"
        )
    else:
        assert row["reasoning_tokens"] == service_count
        assert row["reasoning_tokens_source"] == "service_usage"
        assert row["reasoning_tokens_reason"] is None
    output = tmp_path / "record.json"
    write_json_atomic(output, row)
    saved = json.loads(output.read_text(encoding="utf-8"))
    observed = saved["token_id_observation"]
    diagnostic = (
        observed["prompt_token_ids"] if location == "prompt"
        else observed["output_token_id_chunks"][0]
    )
    assert isinstance(diagnostic, str) and diagnostic.isascii()
    assert json.loads(diagnostic) == invalid_ids


@pytest.mark.parametrize("details", [
    [], {"reasoning_tokens": -1}, {"reasoning_tokens": True},
    {"reasoning_tokens": 2.5}, {"reasoning_tokens": "2"},
])
def test_invalid_usage_still_fails_with_valid_ids(
    monkeypatch, counter, details,
):
    row = execute(monkeypatch, wire(details=details), counter)
    baseline = execute(monkeypatch, wire(details=details), counter, False)
    assert row["status"] == "failed"
    assert row["error"]["type"] == "ValueError"
    assert row["error"] == baseline["error"]
    assert overall(row) == overall(baseline)
    assert "token_id_observation" not in row
    assert row["reasoning_tokens_reason"] == "request_failed"


@pytest.mark.parametrize("options,reason", [
    ({"chunks": None}, "missing_output_token_ids"),
    ({"chunks": ("bad",)}, "invalid_output_token_ids"),
    ({"chunks": ([True],)}, "invalid_output_token_ids"),
    ({"chunks": ([-1],)}, "invalid_output_token_ids"),
    ({"chunks": ([999],)}, "invalid_output_token_ids"),
    ({"chunks": ([11.0],)}, "invalid_output_token_ids"),
    ({"chunks": ([float("nan")],)}, "invalid_output_token_ids"),
    ({"chunks": ([float("inf")],)}, "invalid_output_token_ids"),
    ({"output_tokens": 5}, "output_token_ids_count_mismatch"),
    ({"output_tokens": None}, "missing_completion_tokens"),
    ({"input_tokens": 4}, "prompt_token_ids_count_mismatch"),
    ({"prompt": (True, 100, 14)}, "invalid_prompt_token_ids"),
    ({"conflicting_prompt": True}, "conflicting_prompt_token_ids"),
    ({"prompt": None}, "thinking_start_unavailable"),
    ({"prompt": (100, 16, 14)}, "thinking_start_unavailable"),
    ({"chunks": ([11], [12], [], [13]), "output_tokens": 3},
     "incomplete_thinking_block"),
    ({"chunks": ([100], [12], [101], [13])},
     "multiple_or_nested_thinking_blocks"),
    ({"chunks": ([11], [101], [101], [13])}, "unexpected_thinking_end"),
])
def test_unreliable_ids_only_disable_supplement(
    monkeypatch, counter, options, reason,
):
    row = execute(monkeypatch, wire(**options), counter)
    assert row["status"] == "success"
    assert row["reasoning_tokens"] is None
    assert row["reasoning_tokens_reason"] == reason
    json.dumps(row, allow_nan=False)
    view = aggregate([row])["values"]["request_view"]
    assert view["reasoning_ttft"]["count"] == 1
    assert view["reasoning_tpot"]["mean"] is None
    assert view["reasoning_itl"]["mean"] is None


@pytest.mark.parametrize("prompt", [None, (16, 101, 16)])
def test_generated_start_and_multi_token_chunk(monkeypatch, counter, prompt):
    row = execute(monkeypatch, wire(
        chunks=([100, 11, 12, 101, 13], [], [], []), prompt=prompt,
        output_tokens=5,
    ), counter)
    assert row["reasoning_tokens"] == 2
    assert row["reasoning_token_counting"]["starts_in_thinking"] is False
    assert row["output_tokens"] == 5
    assert row["stream_event_count"] == 3


@pytest.mark.parametrize("count", [0, 1, 2])
def test_empty_single_token_and_same_event_channels(
    monkeypatch, counter, count,
):
    ids = [100, *([11] * count), 101, 13]
    delta = {"content": "c"}
    if count:
        delta["reasoning"] = "a" * count
    data = b"".join([
        sse({"choices": [{"delta": delta, "token_ids": ids}]}),
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        sse({"choices": [], "usage": {
            "prompt_tokens": 3, "completion_tokens": len(ids),
        }}),
        b"data: [DONE]\n\n",
    ])
    row = execute(monkeypatch, data, counter)
    assert row["reasoning_tokens"] == count
    assert row["reasoning_tokens_source"] == "token_ids"
    assert row["stream_event_count"] == 1
    timing = row["phase_timing"]
    assert timing["reasoning_itl_ms"] == (0 if count > 1 else None)
    assert timing["reasoning_stream_event_itl_samples_ms"] == []
    if count:
        assert timing["reasoning_ttft_ms"] == timing[
            "first_content_latency_ms"
        ] == 100


def test_counting_exception_preserves_success(monkeypatch, counter, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(counter.tokenizer, "decode", fail)
    row = execute(monkeypatch, wire(), counter)
    assert row["status"] == "success"
    assert row["reasoning_tokens_reason"] == "tokenizer_counting_failed"
    assert "Thinking Token ID counting failed: index=3" in caplog.text


def test_mixed_sources_weighted_means_and_exclusions(monkeypatch, counter):
    ids = execute(monkeypatch, wire(), counter)
    service = execute(
        monkeypatch, wire(details={"reasoning_tokens": 4}), counter,
    )
    truncated = execute(monkeypatch, wire(finish="length"), counter)
    missing = execute(monkeypatch, wire(chunks=None), counter)
    records = [ids, service, truncated, missing]
    before = copy.deepcopy(records)
    metric = aggregate(records)
    assert records == before
    view = metric["values"]["request_view"]
    assert view["reasoning_tpot"]["mean"] == 133.333
    assert view["reasoning_tpot"]["weight_sum"] == 6
    assert view["reasoning_itl"]["mean"] == 150
    assert view["reasoning_itl"]["weight_sum"] == 4
    assert view["reasoning_itl"]["count"] == 2
    assert "p95" not in view["reasoning_itl"]
    assert metric["values"]["phase_counts"]["excluded_requests"]["value"] == 1
    assert metric["metadata"]["reasoning_token_sources"] == {
        "token_ids": 1, "service_usage": 1, "unavailable": 1,
    }
    assert metric["metadata"]["reasoning_token_missing_reasons"] == {
        "missing_output_token_ids": 1,
    }


@pytest.mark.parametrize("overrides,reason", [
    ({"thinking_metrics": False}, None),
    ({"endpoint": "/v1/completions"}, None),
    ({"request_params": {}}, "token_ids_not_requested"),
    ({"request_params": {"return_token_ids": False}},
     "token_ids_not_requested"),
    ({"tokenizer": ""}, "tokenizer_not_configured"),
])
def test_counter_setup_is_explicit_and_optional(
    monkeypatch, overrides, reason,
):
    def unexpected(*args):
        pytest.fail("must not load a tokenizer")

    monkeypatch.setattr(reasoning_tokens, "load_tokenizer", unexpected)
    result = reasoning_tokens.prepare_reasoning_counter(
        parameters(**overrides),
    )
    assert (result.unavailable_reason if result else None) == reason


def test_unavailable_tokenizer_is_a_supplemental_error(monkeypatch, caplog):
    def fail(*args):
        raise OSError("missing local tokenizer")

    monkeypatch.setattr(reasoning_tokens, "load_tokenizer", fail)
    result = reasoning_tokens.prepare_reasoning_counter(parameters())
    assert result.unavailable_reason == "tokenizer_unavailable"
    assert "Thinking tokenizer preparation failed" in caplog.text
    assert "Traceback" in caplog.text


def test_unsupported_markers_and_disabled_fallback(monkeypatch, counter):
    tokenizer = Tokenizer()
    tokenizer.vocabulary = {"a": 11}
    result = reasoning_tokens.prepare_reasoning_counter(
        parameters(), tokenizer,
    )
    assert result.unavailable_reason == "unsupported_thinking_markers"
    row = execute(monkeypatch, wire(), replace(
        counter, unavailable_reason="tokenizer_not_configured",
    ))
    assert row["status"] == "success"
    assert row["reasoning_tokens_reason"] == "tokenizer_not_configured"


def test_real_sse_single_multi_workers_and_saved_reports(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = make_tokenizer(tokenizer_path)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<think>", "</think>"],
    })
    tokenizer.save_pretrained(tokenizer_path)
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prompt = encode("Q<think>\n")
    chunks = tuple(encode(text) for text in ("a", "b", "</think>", "c"))
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
            posts.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(wire(chunks=chunks, prompt=prompt))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    single = tmp_path / "single.jsonl"
    single.write_text('{"prompt":"private-prompt"}\n')
    multi = tmp_path / "multi.json"
    multi.write_text(json.dumps([{"conversations": [
        {"from": "human", "value": "private-prompt"},
        {"from": "gpt", "value": "reference"},
        {"from": "human", "value": "next"},
        {"from": "gpt", "value": "reference"},
    ]}]))
    common = {
        "model": "mock", "base_url": f"http://127.0.0.1:{server.server_port}",
        "endpoint": "/v1/chat/completions", "profile": "concurrent",
        "max_concurrency": 1, "num_warmups": 0, "api_key": "private-key",
        "thinking_metrics": True, "tokenizer": str(tokenizer_path),
        "request_params": {"return_token_ids": True},
    }
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps({"tasks": [
        {**common, "name": "single", "scenario": "online_serving",
         "data": "jsonl", "dataset_path": str(single), "num_requests": 1},
        {**common, "name": "multi", "scenario": "multi_turn_serving",
         "dataset_path": str(multi), "multi_num_sessions": 1,
         "multi_max_turns": 2},
        {**common, "name": "disabled", "scenario": "online_serving",
         "data": "jsonl", "dataset_path": str(single), "num_requests": 1,
         "thinking_metrics": False},
    ]}))
    try:
        result, directory = RunCoordinator().run(
            build_plan({}, task_file), tmp_path / "runs",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(posts) == 4  # No preflight generation or fallback retry.
    assert all(post["return_token_ids"] is True for post in posts)
    for task, expected in zip(result["tasks"], (1, 2, 1)):
        case = task["cases"][0]
        raw = json.loads((
            directory / "tasks" / task["name"] / "raw/case_1.json"
        ).read_text())
        rows = [request["response"] for request in raw["requests"]]
        assert len(rows) == expected
        assert all(row["status"] == "success" for row in rows)
        for row in rows:
            assert "completion_tokens_details" not in row["usage"]
            assert row["output_tokens"] == 4
            if task["name"] == "disabled":
                assert row["reasoning_tokens"] is None
                assert "reasoning_tokens_source" not in row
                assert "phase_timing" not in row
            else:
                assert row["reasoning_tokens_source"] == "token_ids"
                assert row["reasoning_tokens"] == 2
                assert (
                    row["token_id_observation"]["prompt_token_ids"] == prompt
                )
        if task["name"] == "disabled":
            assert "streaming_phase_timing" not in case["metrics"]
        else:
            phase = case["metrics"]["streaming_phase_timing"]
            view = phase["values"]["request_view"]
            assert view["reasoning_tpot"]["count"] == expected
            assert view["reasoning_tpot"]["weight_sum"] == 2 * expected
            assert view["reasoning_itl"]["weight_sum"] == expected
            assert view["reasoning_itl"]["mean"] is not None
            assert "p95" not in view["reasoning_itl"]
            assert phase["metadata"]["reasoning_token_sources"] == {
                "token_ids": expected,
            }
    text = render_text(result, width=120)
    assert "thinking TPOT: Mean=" in text
    html = write_html(
        result, directory / "token-ids.html",
        include_requests=True, raw_root=directory,
    )
    assert "Token ID补充" in html.read_text()
    embedded = json.loads(html.read_text().split(
        'id="report-data">', 1,
    )[1].split("</script>", 1)[0])
    assert embedded["result"] == result
    assert all(
        cases[0]["status"] == "available"
        for cases in embedded["request_details"]
    )
    logs = list((directory / "log").glob("luban-meter-*.log"))
    assert len(logs) == 1
    log = logs[0].read_text()
    assert "Thinking Token ID fallback prepared:" in log
    assert "token_sources={'token_ids': 2}" in log
    assert "thinking_token_summary=mean_only" in log
    assert "task=multi case=case_1 phase=metrics" in log
    assert "result.json" in log
    assert "private-key" not in log and "private-prompt" not in log


def test_unicode_ids_keep_single_multi_workers_and_full_reports_successful(
    tmp_path,
):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer = make_tokenizer(tokenizer_path)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<think>", "</think>"],
    })
    tokenizer.save_pretrained(tokenizer_path)
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
            posts.append(body)
            count = 2 if body["model"].endswith("2") else None
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(wire(
                chunks=(["\ud800", "\udfff"],),
                details={"reasoning_tokens": count},
            ))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    single = tmp_path / "single.jsonl"
    single.write_text('{"prompt":"private-prompt"}\n')
    multi = tmp_path / "multi.json"
    multi.write_text(json.dumps([{"conversations": [
        {"from": "human", "value": "private-prompt"},
        {"from": "gpt", "value": "reference"},
        {"from": "human", "value": "next"},
        {"from": "gpt", "value": "reference"},
    ]}]))
    tasks = []
    for is_multi in (False, True):
        for enabled in (False, True):
            for count in (None, 2):
                task = {
                    "name": (
                        f"{'multi' if is_multi else 'single'}_"
                        f"{'on' if enabled else 'off'}_R{count}"
                    ),
                    "scenario": (
                        "multi_turn_serving" if is_multi else "online_serving"
                    ),
                    "model": f"mock-{count}",
                    "base_url": f"http://127.0.0.1:{server.server_port}",
                    "endpoint": "/v1/chat/completions",
                    "data": "sharegpt" if is_multi else "jsonl",
                    "dataset_path": str(multi if is_multi else single),
                    "profile": "concurrent", "max_concurrency": 1,
                    "num_warmups": 0, "output_tokens": 8,
                    "ignore_eos": False, "thinking_metrics": enabled,
                    "request_params": {"return_token_ids": True},
                    "tokenizer": str(tokenizer_path),
                }
                if is_multi:
                    task.update(multi_num_sessions=1, multi_max_turns=2)
                else:
                    task["num_requests"] = 1
                tasks.append(task)
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"tasks": tasks}))
    try:
        result, directory = RunCoordinator().run(
            build_plan({}, config), tmp_path / "runs",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result["status"] == "success"
    assert len(posts) == 12
    for task in result["tasks"]:
        case = task["cases"][0]
        assert case["status"] == "success"
        assert case["coverage"]["complete"] == 1
        parameters = case["parameters"]
        enabled = parameters["thinking_metrics"]
        native = 2 if parameters["model"].endswith("2") else None
        expected = 2 if task["scenario"] == "multi_turn_serving" else 1
        metric = case["metrics"]["streaming_performance"]
        assert metric["status"] == "success"
        service = metric["values"]["service_view"]
        assert service["successful_requests"]["value"] == expected
        assert service["failed_requests"]["value"] == 0
        raw = json.loads((
            directory / "tasks" / task["name"] / "raw/case_1.json"
        ).read_text(encoding="utf-8"))
        assert len(raw["requests"]) == expected
        for request in raw["requests"]:
            response = request["response"]
            assert response["status"] == "success"
            assert response["reasoning_tokens"] == native
            if enabled:
                diagnostic = response["token_id_observation"][
                    "output_token_id_chunks"][0]
                assert json.loads(diagnostic) == ["\ud800", "\udfff"]
                assert response["reasoning_tokens_reason"] == (
                    "invalid_output_token_ids" if native is None else None
                )
                assert response["reasoning_tokens_source"] == (
                    None if native is None else "service_usage"
                )
            else:
                assert "token_id_observation" not in response
                assert "phase_timing" not in response
        if enabled:
            phase = case["metrics"]["streaming_phase_timing"]["values"]
            view = phase["request_view"]
            assert view["reasoning_tpot"]["count"] == (
                0 if native is None else expected
            )
        else:
            assert "streaming_phase_timing" not in case["metrics"]
    html = write_html(
        result, directory / "unicode-ids.html",
        include_requests=True, raw_root=directory,
    )
    embedded = json.loads(html.read_text().split(
        'id="report-data">', 1,
    )[1].split("</script>", 1)[0])
    assert embedded["result"] == result
    assert all(cases[0]["status"] == "available"
               for cases in embedded["request_details"])
    logs = list((directory / "log").glob("*.log"))
    assert len(logs) == 1
    log = logs[0].read_text(encoding="utf-8")
    assert "UnicodeEncodeError" not in log
    assert "invalid_output_token_ids" in log
    for task in result["tasks"]:
        assert f"task={task['name']} case=case_1 phase=metrics" in log
    assert "private-prompt" not in log
