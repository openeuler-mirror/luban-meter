"""CPU protocol fixtures for cache setup, isolation and recovery."""

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from luban_meter.core.planning import build_plan
from luban_meter.benchmarking.generation_performance import prefix_cache
from luban_meter.runtime.evaluation_session import run_case
from luban_meter.runtime.case_recovery import recover_cases
from tests.test_unified_runtime import Handler
from tests.tokenizer_fixture import make_tokenizer


class PrefixHandler(Handler):
    mode = None
    fail_at = 3
    on_request = None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.posts.append((self.path, body))
        number = len(Handler.posts) - 1
        ids = body["prompt"]
        salt = body["cache_salt"]
        limit = ((len(ids) - 1) // 16) * 16
        cached = max(
            (length for length in range(16, limit + 1, 16)
             if (salt, tuple(ids[:length])) in self.blocks), default=0,
        )
        for length in range(16, len(ids) + 1, 16):
            self.blocks.add((salt, tuple(ids[:length])))
        mode = self.mode if number == self.fail_at else None
        usage = {
            "prompt_tokens": len(ids), "completion_tokens": body["max_tokens"],
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        if mode == "missing":
            usage.pop("prompt_tokens_details")
        if mode in ("zero", "invalid", "short", "over"):
            usage["prompt_tokens_details"]["cached_tokens"] = {
                "zero": 0, "invalid": True, "short": max(0, cached - 16),
                "over": min(len(ids), cached + 16),
            }[mode]
        if mode == "length":
            usage["prompt_tokens"] += 1
        if self.on_request:
            self.on_request(number)
        self.send_response(500 if mode == "http" else 200)
        self.end_headers()
        if mode == "http":
            self.wfile.write(json.dumps(body).encode())
            return
        text = salt if mode == "echo" else (
            "" if body["max_tokens"] == 1 else "x"
        )
        events = [
            {"choices": [{"text": text, "finish_reason": "length"}]},
            {"choices": [], "usage": usage},
        ]
        if mode == "error":
            events.append({"error": {"message": salt}})
        for event in events:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        if mode != "断流":
            self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def service(tmp_path):
    Handler.posts = []
    Handler.tokenizer_path = str(tmp_path / "tokenizer")
    make_tokenizer(Handler.tokenizer_path)
    PrefixHandler.blocks = set()
    PrefixHandler.mode = None
    PrefixHandler.fail_at = 3
    PrefixHandler.on_request = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), PrefixHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def task(service, **updates):
    return build_plan({
        "scenario": "prefix_cache", "model": "mock", "base_url": service,
        "tokenizer": Handler.tokenizer_path, "prefix_cache_target": 0.5,
        "input_tokens": 64, "output_tokens": 2, "prefix_count": 3,
        "num_requests": 7, "kv_cache_block_size": 16,
        "profile": "synchronous", **updates,
    })["tasks"][0]


def execute(tmp_path, definition):
    directory = tmp_path / "task"
    (directory / "raw").mkdir(parents=True)
    result = run_case(
        prefix_cache.PrefixCacheBenchmark, definition["cases"][0],
        directory, tmp_path / "stop",
    )
    return result, directory


def assert_private(directory):
    text = "\n".join(
        path.read_text() for path in directory.rglob("*") if path.is_file()
    )
    assert all(body["cache_salt"] not in text for _, body in Handler.posts)


def test_order_salts_suffixes_and_formal_window(service, tmp_path):
    result, directory = execute(tmp_path, task(service, num_warmups=1))
    assert result["status"] == "success", result
    bodies = [body for _, body in Handler.posts]
    assert len(bodies) == 17  # Ordinary + seed/probe/verify + seven formal.
    assert bodies[0]["cache_salt"] != bodies[1]["cache_salt"]
    assert len({b["cache_salt"] for b in bodies[1:]}) == 1
    assert [len(b["prompt"]) for b in bodies[1:4]] == [32] * 3
    for i in range(3):
        assert bodies[1+i]["prompt"] == bodies[4+i]["prompt"][:32]
        assert bodies[1+i]["prompt"] == bodies[7+i]["prompt"][:32]
    heads = [tuple(b["prompt"][32:48]) for b in bodies[4:]]
    assert len(set(heads)) == len(heads)
    raw = json.loads(Path(result["raw"]).read_text())
    assert len(raw["requests"]) == len(raw["samples"]) == 7
    assert raw["window"]["admitted_requests"] == 7
    view = result["metrics"]["streaming_performance"]["values"]["request_view"]
    assert view["ttft"]["count"] == view["e2el"]["count"] == 7
    assert raw["metadata"]["cache_warmup"]["verify_completed"] == 3
    events = [json.loads(line) for line in
              (directory / "raw/events.jsonl").read_text().splitlines()]
    setup = [e for e in events if e["phase"] == "cache_warmup"]
    formal = [e for e in events if e["phase"] == "formal"]
    assert max(e["time"] for e in setup) <= raw["window"]["started_at"]
    assert min(e["time"] for e in formal) >= raw["window"]["started_at"]
    assert_private(directory)


@pytest.mark.parametrize("mode, reason", [
    ("missing", "missing"), ("zero", "cache_mismatch"),
    ("invalid", "invalid"), ("short", "cache_mismatch"),
    ("length", "input_length"), ("http", "HTTPStatusError"),
    ("error", "RuntimeError"), ("断流", "RuntimeError"),
])
@pytest.mark.parametrize("at, phase", [(3, "probe"), (6, "verify")])
def test_failure_prevents_formal_and_survives_recovery(
    service, tmp_path, mode, reason, at, phase,
):
    PrefixHandler.mode, PrefixHandler.fail_at = mode, at
    definition = task(service, prefix_cache_warmup_policy="strict")
    result, directory = execute(tmp_path, definition)
    assert result["status"] == "discarded"
    diagnostic = result["setup_diagnostic"]
    assert diagnostic["phase"] == phase
    assert diagnostic["prefix_id"] == 0
    assert diagnostic["expected_cached_tokens"] == 32
    assert diagnostic["reason"] == reason
    assert len(Handler.posts) == at + 1
    assert not (directory / "raw/case_1.json").exists()
    outcome = {"cases": [result]}
    recover_cases(definition, directory, outcome, "interrupted")
    assert outcome["cases"][0] == result
    assert_private(directory)


def test_zero_target_uses_separate_capability_salt(service, tmp_path):
    result, directory = execute(tmp_path, task(service, prefix_cache_target=0))
    assert result["status"] == "success"
    assert len(Handler.posts) == 8
    bodies = [body for _, body in Handler.posts]
    assert bodies[0]["cache_salt"] != bodies[1]["cache_salt"]
    raw = json.loads(Path(result["raw"]).read_text())
    diagnostic = raw["metadata"]["cache_warmup"]
    assert diagnostic["status"] == "not_applicable"
    assert diagnostic["cached_tokens"] == 0
    assert diagnostic["seed_completed"] == diagnostic["verify_completed"] == 0
    assert_private(directory)


def test_cancel_during_validation(service, tmp_path):
    PrefixHandler.on_request = staticmethod(
        lambda index: (tmp_path / "stop").touch() if index == 6 else None
    )
    result, directory = execute(tmp_path, task(service))
    assert result["status"] == "discarded"
    assert result["setup_diagnostic"]["reason"] == "interrupted"
    assert len(Handler.posts) == 7
    assert_private(directory)


def test_formal_echo_of_salt_is_redacted(service, tmp_path):
    PrefixHandler.mode, PrefixHandler.fail_at = "echo", 9
    result, directory = execute(tmp_path, task(service))
    assert result["status"] == "success"
    from luban_meter.reporting.terminal import render_text

    report = render_text({
        "kind": "task", "name": "cache", "category": "generation_performance",
        "status": "success", "cases": [result],
    })
    assert all(body["cache_salt"] not in report for _, body in Handler.posts)
    assert_private(directory)


def test_timeout_is_sanitized_before_scheduler_events(
    service, tmp_path, monkeypatch,
):
    import httpx
    from luban_meter.benchmarking.generation_performance import (
        generation_benchmarks,
    )

    def timeout(request):
        raise httpx.ReadTimeout(request.content.decode())

    monkeypatch.setattr(
        generation_benchmarks, "create_http_client",
        lambda *args, **kwargs: httpx.Client(
            transport=httpx.MockTransport(timeout),
        ),
    )
    result, directory = execute(tmp_path, task(service))
    assert result["setup_diagnostic"]["reason"] == "ReadTimeout"
    assert result["setup_diagnostic"]["phase"] == "seed"
    assert not Handler.posts
    assert_private(directory)


def test_recovery_keeps_only_safe_setup_checkpoint(service, tmp_path):
    from luban_meter.runtime.case_recovery import save_setup_diagnostic

    definition = task(service)
    directory = tmp_path / "crashed"
    (directory / "raw").mkdir(parents=True)
    save_setup_diagnostic(directory, "case_1", {
        "status": "running", "phase": "verify", "prefix_id": 2,
        "expected_cached_tokens": 32, "probe_completed": 3,
        "prompt": [123], "cache_salt": "DO-NOT-PERSIST",
        "reason": "unsafe-error",
    })
    outcome = {"cases": []}
    recover_cases(definition, directory, outcome, "timeout")
    diagnostic = outcome["cases"][0]["setup_diagnostic"]
    assert diagnostic["status"] == "failed"
    assert diagnostic["reason"] == "interrupted"
    assert diagnostic["phase"] == "verify"
    text = json.dumps(outcome)
    assert "DO-NOT-PERSIST" not in text
    assert "unsafe-error" not in text
    assert not (directory / "raw/case_1.setup.json").exists()


@pytest.mark.parametrize("target", [0, 0.5])
def test_missing_seed_or_capability_evidence_fails(service, tmp_path, target):
    PrefixHandler.mode, PrefixHandler.fail_at = "missing", 0
    result, _ = execute(tmp_path, task(service, prefix_cache_target=target))
    assert result["setup_diagnostic"]["reason"] == "missing"
    assert len(Handler.posts) == 1
    assert result["status"] == "discarded"
