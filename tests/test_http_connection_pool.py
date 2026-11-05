"""Real HTTP/1.1 sockets verify pooling, lifecycle and bounded load."""

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from luban_meter.benchmarking.generation_performance.common.streaming import (
    collect_completion_stream,
)
from luban_meter.benchmarking.generation_performance.online_serving import (
    ActiveRequestTracker,
    execute_request,
)
from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
    ServiceError,
)
from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import build_plan
from luban_meter.log import configure_logging, log_context
from luban_meter.runtime.evaluation_session import run_case
from luban_meter.runtime.http_client import create_http_client


class PoolServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), PoolHandler)
        self.condition = threading.Condition()
        self.accepted = 0
        self.closed = 0
        self.requests = []
        self.mode = "ok"
        self.chunk_delay = 0.0
        self.release = threading.Event()
        self.release.set()
        self.stalled = threading.Event()

    def get_request(self):
        connection, address = super().get_request()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self.condition:
            self.accepted += 1
        return connection, address

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"


class PoolHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def finish(self):
        try:
            super().finish()
        finally:
            with self.server.condition:
                self.server.closed += 1
                self.server.condition.notify_all()

    def do_GET(self):
        self.respond({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.respond(body)

    def respond(self, body):
        server = self.server
        mode = server.mode
        with server.condition:
            server.requests.append((self.path, body, dict(self.headers)))
            server.condition.notify_all()
        try:
            if self.path == "/redirect":
                self.send_response(int(mode))
                self.send_header("Location", "/v1/models")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if mode == "http_error":
                self.send_json({"error": "rejected"}, status=503)
                return
            if not body.get("stream"):
                result = {"data": [{"id": "mock"}]}
                if "completions" in self.path:
                    result = {
                        "choices": [{"text": "42",
                                     "message": {"content": "42"}}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    }
                elif self.path == "/tokenize":
                    result = {"tokens": [1, 2]}
                self.send_json(result)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Set-Cookie", "session=unwanted; Path=/")
            self.end_headers()
            if not server.release.wait(timeout=10):
                return
            if mode == "before_timeout":
                server.stalled.wait(timeout=2)
                return
            if mode == "malformed":
                self.send_chunk(b"data: not-json\n\n")
            else:
                self.send_chunk(b'data: {"choices":[{"text":"a"}]}\n\n')
                self.send_chunk(b'data: {"choices":[{"text":"b"}]}\n\n')
                self.send_chunk(
                    b'data: {"choices":[],"usage":'
                    b'{"prompt_tokens":2,"completion_tokens":2}}\n\n'
                )
            self.send_chunk(b"data: [DONE]\n\n")
            if mode == "tail_timeout":
                server.stalled.wait(timeout=2)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_chunk(self, body):
        self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n")
        self.wfile.flush()
        if self.server.chunk_delay:
            time.sleep(self.server.chunk_delay)

    def send_json(self, body, status=200):
        content = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Set-Cookie", "session=unwanted; Path=/")
        self.end_headers()
        self.wfile.write(content)
        self.wfile.flush()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    instance = PoolServer()
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.release.set()
        instance.stalled.set()
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def request(http, server, *, index=0, timeout=3.0, tracker=None):
    return execute_request(
        client=http, request_index=index, payload={"stream": True},
        endpoint="/v1/completions", service_url=server.url,
        api_key="test-key", timeout=timeout, benchmark_start=0,
        scheduled_time=0, tracker=tracker or ActiveRequestTracker(),
        stream_collector=collect_completion_stream,
    )


def assert_closed(server):
    with server.condition:
        assert server.condition.wait_for(
            lambda: server.closed == server.accepted, timeout=5
        )


def test_stream_connections_reused_and_cookies_not_persisted(server):
    with create_http_client(3, max_connections=1) as http:
        for _ in range(3):
            record = request(http, server)
            assert record["status"] == "success"
            assert record["http_version"] == "HTTP/1.1"
        assert server.accepted == 1
    assert_closed(server)
    for _, body, headers in server.requests:
        assert body == {"stream": True}
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Accept-Encoding"] == "identity"
        assert "Cookie" not in headers


def test_idle_expiry_reopens_connection(server):
    with create_http_client(
        3, max_connections=1, keepalive_expiry=0.05
    ) as http:
        assert request(http, server)["status"] == "success"
        time.sleep(0.1)
        assert request(http, server)["status"] == "success"
        assert server.accepted == 2
    assert_closed(server)


def test_read_timeout_is_not_a_total_request_deadline(server):
    server.chunk_delay = 0.08
    with create_http_client(0.2, max_connections=1) as http:
        record = request(http, server, timeout=0.2)
        assert record["status"] == "success"
        assert record["duration_ms"] > 200
        assert "connection_cleanup_error" not in record
    assert_closed(server)


@pytest.mark.parametrize(
    "mode, status",
    [("tail_timeout", "success"), ("before_timeout", "failed"),
     ("malformed", "failed"), ("http_error", "failed")],
)
def test_failures_do_not_leak_pool_slot_or_retry(
    server, mode, status, tmp_path
):
    server.mode = mode
    with create_http_client(0.05, max_connections=1) as http:
        with configure_logging(tmp_path, filename="task.log", task="http"):
            with log_context(case="case_1", request="request-0"):
                first = request(http, server, timeout=0.05)
        assert first["status"] == status
        assert len(server.requests) == 1
        if mode == "tail_timeout":
            assert first["connection_cleanup_error"]["type"] == "ReadTimeout"
            assert first["output_tokens"] == 2
        elif mode == "http_error":
            assert "503" in first["error"]["message"]
            assert "rejected" in first["error"]["message"]
        server.mode = "ok"
        second = request(http, server)
        assert second["status"] == "success"
        assert len(server.requests) == 2
    server.stalled.set()
    assert_closed(server)
    text = (tmp_path / "task.log").read_text()
    assert "Traceback (most recent call last)" in text
    assert "task=http case=case_1" in text and "request=request-0" in text
    if mode == "tail_timeout":
        assert "HTTP connection cleanup failed" in text
    else:
        assert "Streaming request failed" in text


def test_quality_requests_reuse_connection_and_preserve_errors(
    server, tmp_path
):
    service = OpenAIClient(server.url, max_connections=1)
    try:
        assert service.discover_served_model_name() == "mock"
        assert service.tokenize("hello") == [1, 2]
        assert service.complete("hello")["text"] == "42"
        chat = service.chat([{"role": "user", "content": "hi"}])
        assert chat["text"] == "42"
        assert server.accepted == 1
        server.mode = "http_error"
        with configure_logging(tmp_path, filename="task.log"):
            with pytest.raises(ServiceError, match="503.*rejected"):
                service.complete("error")
        assert len(server.requests) == 5
        assert server.accepted == 1
        assert all("Cookie" not in row[2] for row in server.requests)
    finally:
        service.close()
    assert_closed(server)
    text = (tmp_path / "task.log").read_text()
    assert "Quality HTTP request rejected" in text
    assert "Traceback (most recent call last)" in text
    assert "HTTPStatusError" in text


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_post_redirect_behavior(server, status):
    server.mode = str(status)
    service = OpenAIClient(server.url, max_connections=1)
    try:
        if status in (307, 308):
            with pytest.raises(ServiceError, match=str(status)):
                service._post_direct("/redirect", {})
            assert len(server.requests) == 1
        else:
            body, _ = service._post_direct("/redirect", {})
            assert body["data"][0]["id"] == "mock"
            assert len(server.requests) == 2
    finally:
        service.close()
    assert_closed(server)


def test_128_simultaneous_requests_complete_and_release_connections(
    server, tmp_path
):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"prompt":"hello"}\n')
    case = build_plan({
        "scenario": "online_serving", "model": "mock",
        "base_url": server.url, "profile": "concurrent",
        "data": "jsonl", "dataset_path": str(dataset),
        "repeat_data": True, "num_requests": 128, "num_warmups": 0,
        "max_concurrency": 128, "request_timeout": 10,
    })["tasks"][0]["cases"][0]
    (tmp_path / "raw").mkdir()
    server.release.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            run_case, benchmarks.get("online_serving"), case,
            tmp_path, tmp_path / "stop",
        )
        try:
            with server.condition:
                assert server.condition.wait_for(
                    lambda: len(server.requests) == 128, timeout=10
                )
            assert server.accepted == 128
        finally:
            server.release.set()
        result = pending.result(timeout=10)
    assert result["status"] == "success", result
    assert result["window"]["request_capacity"] == 128
    assert result["window"]["peak_requests"] == 128
    facts = json.loads(Path(result["raw"]).read_text())
    records = [row["facts"] for row in facts["samples"]]
    assert all(record["status"] == "success" for record in records)
    assert {row["request_index"] for row in records} == set(range(128))
    assert_closed(server)


def test_warmup_and_formal_share_pool_but_cases_do_not(server, tmp_path):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"prompt":"hello"}\n')
    task = build_plan({
        "scenario": "online_serving", "model": "mock",
        "base_url": server.url, "profile": "synchronous",
        "data": "jsonl", "dataset_path": str(dataset),
        "repeat_data": True, "num_requests": 2, "num_warmups": 1,
        "cases": {"items": [{"http_keepalive_expiry": 5},
                            {"http_keepalive_expiry": 10}]},
    })["tasks"][0]
    (tmp_path / "raw").mkdir()
    for expected_connections, case in enumerate(task["cases"], start=1):
        result = run_case(
            benchmarks.get("online_serving"), case, tmp_path, tmp_path / "stop"
        )
        assert result["status"] == "success", result
        facts = json.loads(Path(result["raw"]).read_text())
        assert len(facts["requests"]) == 2
        assert facts["window"]["admitted_requests"] == 2
        settings = facts["metadata"]["http_client"]
        assert settings["max_connections"] == 1
        assert settings["keepalive_expiry"] == 5 * expected_connections
        assert server.accepted == expected_connections
        assert len(server.requests) == 3 * expected_connections
        assert_closed(server)


@pytest.mark.parametrize("scenario", ["online_serving", "gsm8k"])
def test_failed_warmup_closes_client(server, tmp_path, scenario):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({
        "prompt": "hello", "question": "1 + 1?", "answer": "#### 2",
    }) + "\n")
    case = build_plan({
        "scenario": scenario, "model": "mock", "base_url": server.url,
        "profile": "synchronous", "data": "jsonl",
        "dataset_path": str(dataset), "num_warmups": 1,
    })["tasks"][0]["cases"][0]
    (tmp_path / "raw").mkdir()
    server.mode = "http_error"
    result = run_case(
        benchmarks.get(scenario), case, tmp_path, tmp_path / "stop"
    )
    assert result["status"] == "discarded", result
    assert len(server.requests) == 1
    assert server.accepted == 1
    assert_closed(server)
