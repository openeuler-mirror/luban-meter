"""Phase spans must remain reproducible and expose excluded populations."""

import copy
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from luban_meter.benchmarking.benchmark_metrics import StreamingPhaseTiming
from luban_meter.benchmarking.generation_performance import online_serving
from luban_meter.benchmarking.generation_performance.common import (
    phase_timing,
)
from luban_meter.benchmarking.generation_performance.common.streaming import (
    collect_chat_stream,
)
from luban_meter.core.planning import build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.log import configure_logging
from luban_meter.reporting.html import write_html
from luban_meter.reporting.terminal import render_text


def response(reasoning=(400, 3000), content=(3200, 4200), finish="stop"):
    return {
        "status": "success", "finish_reason": finish,
        "reasoning_event_offsets_ms": list(reasoning),
        "content_event_offsets_ms": list(content),
    }


def thinking_timing(record):
    return phase_timing.request_phase_timing(record, thinking_metrics=True)


def compute(
    records, endpoint="chat", *, thinking_metrics=True,
):
    facts = {
        "parameters": {
            "endpoint": endpoint, "thinking_metrics": thinking_metrics,
        },
        "requests": [{"response": row} for row in records],
    }
    before = copy.deepcopy(facts)
    result = StreamingPhaseTiming().compute(facts)
    assert facts == before
    return result


def event(delta=None, finish=None):
    choice = {"delta": delta or {}}
    if finish is not None:
        choice["finish_reason"] = finish
    return ("data: " + json.dumps({"choices": [choice]}) + "\n\n").encode()


def test_content_only_derivation_matches_full_phase_formulas():
    record = {**response(), "reasoning_tokens": 4}
    timing = phase_timing.request_phase_timing(record)
    assert timing == {
        "status": "eligible", "content_duration_ms": 1000,
        "first_content_latency_ms": 3200, "content_event_count": 2,
    }
    enabled = compute([record])
    disabled = compute([record], thinking_metrics=False)
    view = disabled["values"]["request_view"]
    assert set(view) == {"content_duration", "first_content_latency"}
    for key, value in view.items():
        assert value == enabled["values"]["request_view"][key]
    counts = disabled["values"]["phase_counts"]
    assert "interleaved_requests" not in counts
    assert not any("reasoning" in key for key in counts)
    for key, value in counts.items():
        assert value == enabled["values"]["phase_counts"][key]


@pytest.mark.parametrize("enabled", [None, False])
def test_disabled_switch_skips_phase_derivation(monkeypatch, enabled):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled phase timing must not run")

    monkeypatch.setattr(phase_timing, "request_phase_timing", forbidden)
    monkeypatch.setattr(
        online_serving.reasoning_tokens, "supplement_reasoning_tokens",
        forbidden,
    )
    wire = (event({"reasoning": "r", "content": "c"})
            + event(finish="length") + b"data: [DONE]\n\n")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=httpx.ByteStream(wire))
    )
    arguments = {} if enabled is None else {"thinking_metrics": enabled}
    with httpx.Client(transport=transport) as client:
        record = online_serving.execute_request(
            client=client, request_index=0, payload={},
            endpoint="chat", service_url="http://test",
            api_key="", timeout=1, benchmark_start=0, scheduled_time=0,
            tracker=online_serving.ActiveRequestTracker(),
            stream_collector=collect_chat_stream, **arguments,
        )
    assert record["status"] == "success"
    assert record["finish_reason"] == "length"
    assert record["content_event_offsets_ms"]
    assert record["reasoning_event_offsets_ms"]
    assert "phase_timing" not in record


@pytest.mark.parametrize("width", [60, 120])
@pytest.mark.parametrize("verbose", [False, True])
def test_thinking_token_means_have_no_terminal_percentile_placeholders(
    width, verbose,
):
    metric = compute([{**response(), "reasoning_tokens": 4}])
    view = metric["values"]["request_view"]
    metric["values"]["request_view"] = {
        name: view[name] for name in ("reasoning_tpot", "reasoning_itl")
    }
    result = {
        "kind": "run", "id": "means", "status": "success",
        "tasks": [{
            "name": "means", "category": "generation_performance",
            "status": "success", "cases": [{
                "id": "case_1", "status": "success", "parameters": {},
                "metrics": {"streaming_phase_timing": metric},
            }],
        }],
    }
    text = render_text(result, width=width, verbose=verbose)
    assert "Mean=750 ms/token" in text
    assert "Mean=866.667 ms/token" in text
    assert "P50=" not in text and "P95=" not in text
    assert "| P50" not in text and "| P95" not in text
    assert "weighted_cdf" not in text
    assert "weight_sum=4" in text or "权重总数=4" in text


def test_fixed_stream_timeline_matches_agreed_example():
    wire = b"".join([
        event({"reasoning": "a"}), event({"reasoning": "b"}),
        event({"content": "c"}), event({"content": "d"}),
        event(finish="stop"), b"data: [DONE]\n\n",
    ])
    clock = iter([10.4, 13.0, 13.2, 14.2])
    observation = collect_chat_stream(io.BytesIO(wire), lambda: next(clock))
    record = online_serving._build_request_record(
        request_index=0, started=10, ended=14.3, benchmark_start=10,
        scheduled_time=10, observation=observation, error=None,
    )
    timing = thinking_timing(record)
    assert timing == {
        "status": "eligible", "reasoning_duration_ms": 2600,
        "content_duration_ms": 1000, "first_content_latency_ms": 3200,
        "reasoning_event_count": 2, "content_event_count": 2,
        "interleaved": False,
        "reasoning_ttft_ms": 400, "reasoning_tpot_ms": None,
        "reasoning_itl_ms": None,
        "reasoning_stream_event_itl_samples_ms": [2600],
    }
    assert record["ttft_ms"] == 400
    assert record["e2el_ms"] == 4300


def test_thinking_latency_formulas_use_request_start_and_channel_tokens():
    record = {**response(), "reasoning_tokens": 4}
    timing = thinking_timing(record)
    assert timing["reasoning_ttft_ms"] == 400
    assert timing["reasoning_tpot_ms"] == 750
    assert timing["reasoning_itl_ms"] == pytest.approx(2600 / 3)
    assert timing["reasoning_stream_event_itl_samples_ms"] == [2600]
    view = compute([record])["values"]["request_view"]
    assert view["reasoning_itl"]["mean"] == 866.667
    assert view["reasoning_itl"]["weight_sum"] == 3
    assert view["reasoning_tpot"]["weight_sum"] == 4


def test_thinking_weighted_population_and_event_intervals():
    first = {**response([100, 500], [600]), "reasoning_tokens": 2}
    second = {**response([200, 300, 800], [900]), "reasoning_tokens": 4}
    excluded = {**response(finish="length"), "reasoning_tokens": 100}
    view = compute([first, second, excluded])["values"]["request_view"]
    assert view["reasoning_ttft"]["mean"] == 150
    assert view["reasoning_ttft"]["p50"] == 150
    tpot, itl = view["reasoning_tpot"], view["reasoning_itl"]
    assert (tpot["count"], tpot["weight_sum"], tpot["mean"]) == (2, 6, 216.667)
    assert (itl["count"], itl["weight_sum"], itl["mean"]) == (2, 4, 250)
    for item in (tpot, itl):
        assert set(item) == {
            "mean", "count", "weight_sum", "unit", "aggregation",
            "sample_unit", "weight_unit", "condition",
        }
        assert item["aggregation"] == "token_weighted"
    events = view["reasoning_stream_event_itl"]
    assert (events["count"], events["mean"], events["p95"]) == (
        3, 333.333, 490,
    )
    assert events["sample_unit"] == "event_interval"


@pytest.mark.parametrize("tokens", [None, 0])
def test_absent_or_zero_reasoning_tokens_only_disable_token_metrics(tokens):
    view = compute([{**response(), "reasoning_tokens": tokens}])[
        "values"]["request_view"]
    assert view["reasoning_ttft"]["count"] == 1
    assert view["reasoning_duration"]["mean"] == 2600
    assert view["reasoning_stream_event_itl"]["count"] == 1
    for name in ("reasoning_tpot", "reasoning_itl"):
        assert view[name]["count"] == 0
        assert view[name]["weight_sum"] == 0
        assert view[name]["mean"] is None


@pytest.mark.parametrize("details,weights", [
    pytest.param(None, (0, 0), id="missing-details"),
    pytest.param({}, (0, 0), id="missing-count"),
    pytest.param({"reasoning_tokens": None}, (0, 0), id="null"),
    pytest.param({"reasoning_tokens": 0}, (0, 0), id="zero"),
    pytest.param({"reasoning_tokens": 1}, (1, 0), id="one"),
    pytest.param({"reasoning_tokens": 4}, (4, 3), id="multiple"),
    pytest.param({"reasoning_tokens": -1}, None, id="negative"),
    pytest.param({"reasoning_tokens": True}, None, id="boolean"),
    pytest.param({"reasoning_tokens": 2.5}, None, id="fraction"),
    pytest.param({"reasoning_tokens": "4"}, None, id="string"),
    pytest.param([], None, id="invalid-details"),
])
@pytest.mark.parametrize("thinking_metrics", [False, True])
def test_http_sse_reasoning_counts_preserve_request_outcome(
    details, weights, tmp_path, thinking_metrics,
):
    usage = {"prompt_tokens": 10, "completion_tokens": 6}
    if details is not None:
        usage["completion_tokens_details"] = details
    wire = b"".join([
        event({"reasoning": "private-thought"}),
        event({"reasoning": "r2"}), event({"content": "private-answer"}),
        event(finish="stop"),
        ("data: " + json.dumps({"choices": [], "usage": usage})
         + "\n\n").encode(),
        b"data: [DONE]\n\n",
    ])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(wire)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with configure_logging(tmp_path, task="reasoning-count") as log_path:
            with httpx.Client() as client:
                started = time.perf_counter()
                row = online_serving.execute_request(
                    client=client, request_index=7,
                    payload={"prompt": "private-prompt"},
                    endpoint="chat",
                    service_url=f"http://127.0.0.1:{server.server_port}",
                    api_key="private-key", timeout=2,
                    benchmark_start=started, scheduled_time=started,
                    tracker=online_serving.ActiveRequestTracker(),
                    stream_collector=collect_chat_stream,
                    thinking_metrics=thinking_metrics,
                )
            case = {
                "requests": [row], "benchmark_duration_seconds": 1,
                "maximum_request_concurrency": 1,
                "peak_concurrent_requests": 1,
            }
            overall, success, failed = online_serving.process_case(case)
            supplement = compute([row]) if thinking_metrics else None
            assert online_serving.process_case(case) == (
                overall, success, failed,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    if thinking_metrics:
        view = supplement["values"]["request_view"]
        counts = supplement["values"]["phase_counts"]
    log = log_path.read_text()
    if weights is None:
        assert row["status"] == "failed"
        assert row["error"]["type"] == "ValueError"
        assert "reasoning_tokens" in row["error"]["message"] or (
            "completion_tokens_details" in row["error"]["message"]
        )
        if thinking_metrics:
            assert row["phase_timing"]["status"] == "failed"
            assert all(node["count"] == 0 and node["mean"] is None
                       for node in view.values())
            assert counts["excluded_requests"]["value"] == 1
            assert counts["failed_requests"]["value"] == 1
        assert (success, failed) == (0, 1)
        assert "Streaming request failed: index=7" in log
        assert "Traceback (most recent call last)" in log
    else:
        assert row["status"] == "success"
        assert (success, failed) == (1, 0)
        if thinking_metrics:
            assert row["phase_timing"]["status"] == "eligible"
            assert counts["excluded_requests"]["value"] == 0
            for name in ("reasoning_ttft", "reasoning_duration",
                         "reasoning_stream_event_itl"):
                assert view[name]["count"] == 1
            for name, weight in zip(
                ("reasoning_tpot", "reasoning_itl"), weights,
            ):
                assert view[name]["weight_sum"] == weight
                assert view[name]["count"] == int(weight > 0)
        assert overall["request_view"]["tpot"]["weight_sum"] == 6
        assert overall["request_view"]["itl"]["weight_sum"] == 5
    for name in ("ttft", "tpot", "itl", "e2el"):
        assert overall["request_view"][name]["count"] == success
    assert overall["service_view"]["total_requests"]["value"] == 1
    assert overall["service_view"]["failed_requests"]["value"] == failed
    if not thinking_metrics:
        assert "phase_timing" not in row
        assert "Phase timing computed:" not in log
    else:
        assert "Phase timing computed:" in log
    for secret in ("private-key", "private-prompt", "private-thought",
                   "private-answer"):
        assert secret not in log


@pytest.mark.parametrize("tokens,expected_itl", [(1, None), (4, 0)])
def test_one_thinking_event_does_not_fabricate_event_intervals(
    tokens, expected_itl,
):
    row = {**response([400], [400]), "reasoning_tokens": tokens}
    timing = thinking_timing(row)
    assert timing["reasoning_tpot_ms"] == 400 / tokens
    assert timing["reasoning_itl_ms"] == expected_itl
    view = compute([row])["values"]["request_view"]
    assert view["reasoning_stream_event_itl"]["count"] == 0
    assert view["reasoning_itl"]["count"] == (0 if tokens == 1 else 1)


def test_no_thinking_channel_has_no_thinking_metrics_even_with_tokens():
    view = compute([{**response([], [500]), "reasoning_tokens": 10}])[
        "values"]["request_view"]
    for name in ("reasoning_ttft", "reasoning_tpot", "reasoning_itl",
                 "reasoning_stream_event_itl"):
        assert view[name]["count"] == 0
        assert view[name]["mean"] is None


def test_thinking_supplement_preserves_overall_metrics_and_population():
    wire = b"".join([
        event({"reasoning": "a"}), event({"reasoning": "b"}),
        event({"content": "c"}), event({"content": "d"}),
        event(finish="stop"),
        b'data: {"choices":[],"usage":{"prompt_tokens":10,'
        b'"completion_tokens":6,"completion_tokens_details":'
        b'{"reasoning_tokens":4}}}\n\n',
    ])
    clock = iter([10.4, 13, 13.2, 14.2])
    observation = collect_chat_stream(io.BytesIO(wire), lambda: next(clock))
    record = online_serving._build_request_record(
        request_index=0, started=10, ended=14.3, benchmark_start=10,
        scheduled_time=10, observation=observation, error=None,
    )
    records = [record, {**record, "finish_reason": "length"}]
    case = {
        "requests": records, "benchmark_duration_seconds": 10,
        "maximum_request_concurrency": 2, "peak_concurrent_requests": 2,
    }
    before = online_serving.process_case(case)
    supplement = compute(records)["values"]["request_view"]
    assert online_serving.process_case(case) == before
    overall = before[0]["request_view"]
    assert before[1:] == (2, 0)
    assert overall["ttft"]["count"] == 2
    assert overall["ttft"]["mean"] == 400
    assert overall["e2el"]["mean"] == 4300
    assert overall["tpot"]["mean"] == 700
    assert overall["tpot"]["weight_sum"] == 12
    assert overall["itl"]["mean"] == 760
    assert overall["itl"]["weight_sum"] == 10
    for name, expected in (("tpot", 700), ("itl", 760)):
        assert overall[name]["p50"] == expected
        assert overall[name]["p95"] == expected
        assert overall[name]["percentile_method"] == "weighted_cdf"
    assert supplement["reasoning_ttft"]["count"] == 1
    assert supplement["reasoning_tpot"]["mean"] == 750
    assert supplement["reasoning_tpot"]["weight_sum"] == 4


@pytest.mark.parametrize("reasoning,content,expected,interleaved", [
    ([100], [100], (0, 0, 100), False),
    ([], [200, 600], (None, 400, 200), False),
    ([100, 500], [], (400, None, None), False),
    ([100, 500], [200, 600], (400, 400, 200), True),
    ([300], [100], (0, 0, 100), True),
])
def test_single_missing_and_interleaved_channels(
    reasoning, content, expected, interleaved,
):
    timing = thinking_timing(response(reasoning, content))
    assert tuple(timing[name] for name in (
        "reasoning_duration_ms", "content_duration_ms",
        "first_content_latency_ms",
    )) == expected
    assert timing["interleaved"] is interleaved


@pytest.mark.parametrize("finish,status", [
    ("length", "truncated"), (None, "missing_finish"),
    ("tool_calls", "other_finish"), ("content_filter", "other_finish"),
])
def test_incomplete_outputs_retain_spans_but_exclude_entire_request(
    finish, status,
):
    record = response(finish=finish)
    timing = thinking_timing(record)
    assert timing["status"] == status
    assert timing["reasoning_duration_ms"] == 2600
    assert timing["content_duration_ms"] == 1000
    metric = compute([record])
    assert metric["status"] == "unavailable"
    assert all(node["count"] == 0 and node["mean"] is None
               for node in metric["values"]["request_view"].values())
    assert metric["values"]["phase_counts"][status + "_requests"]["value"] == 1


def test_separate_denominators_and_exclusion_counts():
    failed = {**response(), "status": "failed"}
    metric = compute([
        response(), response([100], [100]), response([], [200, 600]),
        response([100, 500], []), response([100, 500], [200, 600]),
        response(finish="length"), response(finish=None),
        response(finish="tool_calls"), failed,
    ])
    view = metric["values"]["request_view"]
    assert view["reasoning_duration"]["count"] == 4
    assert view["reasoning_duration"]["mean"] == 850
    assert view["reasoning_duration"]["p95"] == 2270
    assert view["content_duration"]["count"] == 4
    assert view["content_duration"]["mean"] == 450
    assert view["first_content_latency"]["mean"] == 925
    counts = {k: v["value"] for k, v in
              metric["values"]["phase_counts"].items()}
    assert counts == {
        "total_requests": 9, "eligible_requests": 5,
        "excluded_requests": 4, "failed_requests": 1,
        "truncated_requests": 1, "missing_finish_requests": 1,
        "other_finish_requests": 1,
        "interleaved_requests": 1, "reasoning_single_event_requests": 1,
        "content_single_event_requests": 1, "missing_reasoning_requests": 1,
        "missing_content_requests": 1,
    }
    assert thinking_timing(failed)[
        "reasoning_duration_ms"] is None


def test_no_usage_required_and_empty_or_non_chat_results_unavailable():
    assert compute([response()])["status"] == "success"
    assert compute([])["status"] == "unavailable"
    assert compute([response([], [])])["status"] == "unavailable"
    assert compute([response()], "completion")["status"] == "unavailable"


@pytest.mark.parametrize("field", [
    "reasoning_event_offsets_ms", "content_event_offsets_ms",
])
def test_successful_records_require_both_channel_timestamp_fields(field):
    record = response()
    del record[field]
    with pytest.raises(KeyError, match=field):
        compute([record])


def test_embedded_think_tags_do_not_create_a_reasoning_channel():
    observation = collect_chat_stream(io.BytesIO(
        event({"content": "<think>thought</think>answer"})
        + event(finish="stop")
    ), clock=lambda: 2.0)
    assert observation.reasoning_event_times == ()
    assert observation.content_event_times == (2.0,)


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectTimeout])
def test_timeout_has_no_phase_values(failure):
    def timeout(request):
        raise failure("simulated timeout", request=request)

    tracker = online_serving.ActiveRequestTracker()
    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        record = online_serving.execute_request(
            client=client, request_index=9, payload={},
            endpoint="chat", service_url="http://test",
            api_key="", timeout=1, benchmark_start=0, scheduled_time=0,
            tracker=tracker, stream_collector=collect_chat_stream,
            thinking_metrics=True,
        )
    assert record["status"] == "failed"
    assert record["error"]["type"] == failure.__name__
    assert record["duration_ms"] >= 0
    assert all(record["phase_timing"][name] is None for name in (
        "reasoning_duration_ms", "content_duration_ms",
        "first_content_latency_ms",
    ))
    metric = compute([record])
    assert metric["values"]["phase_counts"]["failed_requests"]["value"] == 1


def test_real_workers_reports_and_shared_log(tmp_path):
    posts = []
    modes = ["normal", "single", "content_only", "reasoning_only",
             "length", "missing", "tool_calls", "broken"]

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
            mode = body["messages"][-1]["content"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if mode == "broken":
                self.wfile.write(event({"reasoning": "private-output"})
                                 + b"data: {invalid\n\n")
                return
            if mode == "single":
                chunks = [event({"reasoning": "r", "content": "c"})]
            else:
                chunks = []
                if mode != "content_only":
                    chunks += [event({"reasoning": "private-output"}),
                               event({"reasoning": "r2"})]
                if mode != "reasoning_only":
                    chunks += [event({"content": "c1"}),
                               event({"content": "c2"})]
            finish = mode if mode in ("length", "tool_calls") else "stop"
            if mode != "missing":
                chunks.append(event(finish=finish))
            if mode != "reasoning_only":
                tokens = {"single": 1, "content_only": 0}.get(mode, 4)
                chunks.append(("data: " + json.dumps({
                    "choices": [], "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": tokens + 2,
                        "completion_tokens_details": {
                            "reasoning_tokens": tokens,
                        },
                    },
                }) + "\n\n").encode())
            self.wfile.write(b"".join(chunks) + b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dataset = tmp_path / "single.jsonl"
    dataset.write_text("\n".join(json.dumps({"prompt": m}) for m in modes))
    multi = tmp_path / "multi.json"
    multi.write_text(json.dumps([{"conversations": [
        {"from": "human", "value": "private-prompt"},
        {"from": "gpt", "value": "reference"},
        {"from": "human", "value": "next"},
        {"from": "gpt", "value": "reference"},
    ]}]))
    common = {
        "model": "mock", "base_url": f"http://127.0.0.1:{server.server_port}",
        "endpoint": "chat", "profile": "concurrent",
        "max_concurrency": 1, "num_warmups": 0, "api_key": "private-key",
        "thinking_metrics": True,
    }
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"tasks": [
        {**common, "name": "single", "scenario": "online_serving",
         "data": "jsonl", "dataset_path": str(dataset), "num_requests": 8},
        {**common, "name": "multi", "scenario": "multi_turn_serving",
         "dataset_path": str(multi), "multi_num_sessions": 1,
         "multi_max_turns": 2},
        {**common, "name": "disabled", "scenario": "online_serving",
         "data": "jsonl", "dataset_path": str(dataset), "num_requests": 1,
         "thinking_metrics": False},
        {**common, "name": "disabled_multi", "scenario": "multi_turn_serving",
         "dataset_path": str(multi), "multi_num_sessions": 1,
         "multi_max_turns": 2, "thinking_metrics": False},
        {**{key: value for key, value in common.items()
            if key != "thinking_metrics"},
         "name": "default", "scenario": "online_serving", "data": "jsonl",
         "dataset_path": str(dataset), "num_requests": 1},
    ]}))
    try:
        result, directory = RunCoordinator().run(
            build_plan({}, tasks), tmp_path / "runs"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(posts) == 14
    single, multi_result = (
        task["cases"][0] for task in result["tasks"][:2]
    )
    for task in result["tasks"][2:]:
        disabled = task["cases"][0]
        assert disabled["parameters"]["metrics"] == ["streaming_performance"]
        assert "streaming_phase_timing" not in disabled["metrics"]
        disabled_raw = json.loads((
            directory / "tasks" / task["name"] / "raw/case_1.json"
        ).read_text())
        for request in disabled_raw["requests"]:
            disabled_response = request["response"]
            assert disabled_response["reasoning_tokens"] == 4
            assert len(disabled_response["reasoning_event_offsets_ms"]) == 2
            assert "phase_timing" not in disabled_response
        disabled_text = render_text({**result, "tasks": [task]}, width=120)
        for label in ("streaming_phase_timing", "thinking TTFT",
                      "正式回答输出时长", "首个正式回答延迟", "分阶段样本"):
            assert label not in disabled_text
    single_metric = single["metrics"]["streaming_phase_timing"]
    counts = single_metric["values"]["phase_counts"]
    assert counts["eligible_requests"]["value"] == 4
    assert counts["excluded_requests"]["value"] == 4
    assert counts["failed_requests"]["value"] == 1
    assert single_metric["values"]["request_view"][
        "reasoning_duration"]["count"] == 3
    thinking = single_metric["values"]["request_view"]
    assert thinking["reasoning_ttft"]["count"] == 3
    assert thinking["reasoning_tpot"]["count"] == 2
    assert thinking["reasoning_tpot"]["weight_sum"] == 5
    assert thinking["reasoning_itl"]["count"] == 1
    assert thinking["reasoning_itl"]["weight_sum"] == 3
    assert thinking["reasoning_stream_event_itl"]["count"] == 2
    multi_metric = multi_result["metrics"]["streaming_phase_timing"]
    assert multi_metric["values"]["request_view"][
        "content_duration"]["count"] == 2
    assert multi_metric["values"]["request_view"][
        "reasoning_tpot"]["weight_sum"] == 8
    raw = json.loads((directory / "tasks/single/raw/case_1.json").read_text())
    responses = [r["response"] for r in raw["requests"]]
    assert [r["phase_timing"]["status"] for r in responses] == [
        "eligible", "eligible", "eligible", "eligible", "truncated",
        "missing_finish", "other_finish", "failed",
    ]
    assert responses[1]["phase_timing"]["reasoning_duration_ms"] == 0
    assert responses[-1]["phase_timing"]["content_duration_ms"] is None
    assert responses[-1]["error"]["type"] == "JSONDecodeError"
    text = render_text(result, width=120)
    assert "thinking 输出时长: Mean=" in text
    assert "P95=" in text and "N=3" in text and "排除=4" in text
    assert "thinking TTFT: Mean=" in text
    assert "thinking TPOT: Mean=" in text
    assert "N=2 权重总数=5 Token" in text
    assert "N=1 权重总数=3 Token间隔" in text
    assert "N=2（事件间隔）" in text
    for line in text.splitlines():
        if "thinking TPOT: Mean=" in line or "thinking ITL: Mean=" in line:
            assert "P95" not in line
    (directory / "summary.txt").write_text(text)
    html = write_html(result, directory / "phases.html",
                      include_requests=True, raw_root=directory)
    embedded = json.loads(html.read_text().split(
        'id="report-data">', 1
    )[1].split("</script>", 1)[0])
    assert embedded["result"] == result
    logs = list((directory / "log").glob("luban-meter-*.log"))
    assert len(logs) == 1
    log = logs[0].read_text()
    assert "Phase timing computed: eligible=4 excluded=4" in log
    assert "Phase timing computed: eligible=2 excluded=0" in log
    for name in ("disabled", "disabled_multi", "default"):
        assert any(
            f"task={name} case=case_1" in line
            and "Phase timing configured: enabled=False" in line
            for line in log.splitlines()
        )
    assert "Phase timing configured: enabled=True" in log
    assert "thinking_samples=" in log
    assert "thinking_weights={'reasoning_tpot': 5, 'reasoning_itl': 3}" in log
    assert "thinking_token_summary=mean_only" in log
    assert "Streaming request failed: index=7" in log
    assert "Traceback (most recent call last)" in log
    assert "task=single case=case_1 phase=metrics" in log
    assert "Task" in log and "result.json" in log
    for secret in ("private-key", "private-output", "private-prompt"):
        assert secret not in log


def test_empty_phase_distributions_show_unavailable():
    result = {
        "kind": "run", "id": "test", "status": "success", "tasks": [{
            "name": "test", "category": "generation_performance",
            "status": "success", "cases": [{
                "id": "case_1", "status": "success",
                "parameters": {"endpoint": "chat"},
                "metrics": {"streaming_phase_timing": compute([
                    response(finish="length"),
                ])},
            }],
        }],
    }
    text = render_text(result)
    assert "thinking 输出时长: 不可用 (N=0)" in text
    assert "thinking TPOT: 不可用 (N=0) 权重总数=0 Token" in text
    assert "thinking ITL: 不可用 (N=0) 权重总数=0 Token间隔" in text
