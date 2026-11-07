"""PR54 regressions use injected HTTP faults and the real logging scope."""

import io
import json

import httpx
import pytest

from luban_meter.log import configure_logging
from luban_meter.reporting.html import report_data
from luban_meter.reporting.terminal import render_text
from tests.test_prefix_cache_warmup import (
    execute, service as _service, task,
)


@pytest.fixture
def service(tmp_path):
    yield from _service.__wrapped__(tmp_path)


def inject(monkeypatch, fail_at, mode):
    from luban_meter.benchmarking.generation_performance import (
        generation_benchmarks,
    )

    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) - 1 == fail_at:
            echo = request.content.decode() + " PRIVATE-RESPONSE-BODY"
            if mode == "timeout":
                raise httpx.ReadTimeout(echo, request=request)
            if mode == "http":
                return httpx.Response(
                    503, stream=httpx.ByteStream(echo.encode()),
                )
            if mode == "badjson":
                return httpx.Response(200, stream=httpx.ByteStream(
                    ("data: not-json " + echo + "\n\n").encode(),
                ))
            if mode == "断流":
                return httpx.Response(200, stream=httpx.ByteStream(
                    b"data: {}\n\n",
                ))
        usage = {
            "prompt_tokens": len(body["prompt"]),
            "completion_tokens": body["max_tokens"],
            "prompt_tokens_details": {
                "cached_tokens": 0 if len(bodies) == 1 else 32,
            },
        }
        events = [
            {"choices": [{"text": "x", "finish_reason": "length"}]},
            {"choices": [], "usage": usage},
        ]
        return httpx.Response(
            200, stream=httpx.ByteStream((
                "".join("data: " + json.dumps(e) + "\n\n" for e in events)
                + "data: [DONE]\n\n"
            ).encode()),
        )

    monkeypatch.setattr(
        generation_benchmarks, "create_http_client",
        lambda *a, **k: httpx.Client(transport=httpx.MockTransport(respond)),
    )
    return bodies


@pytest.mark.parametrize("policy", ["strict", "observe"])
@pytest.mark.parametrize("mode", ["http", "timeout", "badjson", "断流"])
def test_setup_failure_is_traceable_without_body(
    service, tmp_path, monkeypatch, policy, mode,
):
    bodies = inject(monkeypatch, 0, mode)
    console = io.StringIO()
    with configure_logging(tmp_path / "logs", stream=console) as log:
        result, _ = execute(tmp_path, task(
            service, prefix_count=1, prefix_cache_warmup_policy=policy,
        ))
    assert result["status"] == "discarded"
    assert len(bodies) == 1
    text = log.read_text()
    assert "stage=seed prefix_id=0" in text
    assert "request=0" in text and "_send_cache_request" in text
    assert "Traceback (most recent call last)" in text
    expected = {
        "http": "HTTPStatusError HTTP 503", "timeout": "ReadTimeout",
        "badjson": "JSONDecodeError", "断流": "RuntimeError",
    }[mode]
    assert expected in text
    for output in (text, console.getvalue()):
        assert bodies[0]["cache_salt"] not in output
        assert json.dumps(bodies[0]["prompt"]) not in output
        assert "PRIVATE-RESPONSE-BODY" not in output


@pytest.mark.parametrize("mode", ["http", "timeout"])
def test_formal_exception_is_redacted_before_logging_and_case_is_partial(
    service, tmp_path, monkeypatch, mode,
):
    bodies = inject(monkeypatch, 3, mode)
    console = io.StringIO()
    with configure_logging(tmp_path / "logs", stream=console) as log:
        result, directory = execute(tmp_path, task(
            service, prefix_count=1, prefix_cache_warmup_policy="strict",
        ))
    metric = result["metrics"]["prefix_cache_hit_rate"]
    v = {k: n["value"] for k, n in metric["values"].items()}
    assert len(bodies) == 10  # Three setup and seven formal requests.
    assert v["submitted_requests"] == 7
    assert v["successful_requests"] == 6
    assert v["actual_hit_rate"] is None
    assert v["observed_subset_hit_rate"] == 0.5
    assert v["total_cached_tokens"] == 192
    assert v["measured_prompt_tokens"] == 384
    assert v["request_coverage"] == pytest.approx(6 / 7)
    assert v["successful_request_coverage"] == 1
    assert v["token_coverage"] is None
    assert metric["measurement"]["status"] == "partial"
    assert "formal_observation_incomplete" in metric["measurement"]["reasons"]
    assert metric["target_attainment"]["status"] == "unknown"
    assert result["metrics"]["streaming_performance"]["values"][
        "request_view"
    ]["ttft"]["count"] == 6
    text = log.read_text() + console.getvalue()
    assert "Streaming request failed: index=0" in text
    assert "ReadTimeout" in text if mode == "timeout" else "HTTP 503" in text
    for body in bodies:
        assert body["cache_salt"] not in text
        assert json.dumps(body["prompt"]) not in text
    assert "PRIVATE-RESPONSE-BODY" not in text
    payload = {
        "kind": "task", "name": "review-mock", "scenario": "prefix_cache",
        "category": "generation_performance", "status": "partial_failed",
        "cases": [result],
    }
    terminal = render_text(payload, all_cases=True)
    assert "实际命中率: 不可测" in terminal
    assert "已测子集命中率（非完整 Case）: 50%" in terminal
    assert "(6/7)" in terminal
    projected = report_data(payload)["result"]["cases"][0]["metrics"][
        "prefix_cache_hit_rate"
    ]
    assert projected["values"]["actual_hit_rate"]["value"] is None
    assert projected["values"]["request_coverage"]["value"] < 1
    facts = json.loads((directory / "raw/case_1.json").read_text())
    assert all(body["cache_salt"] not in json.dumps(facts) for body in bodies)


def test_default_observe_keeps_low_hits_without_refill(service, tmp_path):
    from tests.test_prefix_cache_warmup import Handler, PrefixHandler

    PrefixHandler.mode, PrefixHandler.fail_at = "zero", 3
    result, _ = execute(tmp_path, task(service))
    metric = result["metrics"]["prefix_cache_hit_rate"]
    assert metric["warmup"]["warmup_policy"] == "observe"
    assert metric["warmup"]["warmup_fully_matched"] is False
    assert metric["measurement"]["status"] == "valid"
    assert metric["values"]["successful_requests"]["value"] == 7
    assert len(Handler.posts) == 16


def test_real_worker_shares_redacted_instance_log(service, tmp_path):
    from luban_meter.core.run_coordinator import RunCoordinator
    from tests.test_prefix_cache_warmup import Handler, PrefixHandler

    PrefixHandler.mode, PrefixHandler.fail_at = "http", 9
    definition = task(service)
    result, directory = RunCoordinator().run(
        {"tasks": [definition], "case_count": 1}, tmp_path / "runs",
    )
    logs = list((directory / "log").glob("*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "Worker started" in text and "Worker initializing" in text
    assert "Streaming request failed: index=0" in text
    assert "HTTPStatusError HTTP 500" in text
    assert "Traceback (most recent call last)" in text
    assert result["tasks"][0]["cases"][0]["status"] == "partial_failed"
    for _, body in Handler.posts:
        assert body["cache_salt"] not in text
        assert json.dumps(body["prompt"]) not in text
