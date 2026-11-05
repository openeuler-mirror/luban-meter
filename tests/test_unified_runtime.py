"""Public extension and real subprocess tests using a synthetic service."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from luban_meter.api import (
    GenerationPerformanceBenchmark,
    Metric,
    ModelServiceQualityBenchmark,
    Parameter,
    benchmarks,
    metrics,
)
from luban_meter.cli import main
from luban_meter.core.planning import bootstrap, build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.reporting.html import load_result, write_html
from luban_meter.runtime.evaluation_session import run_case
from luban_meter.runtime.load_profiles import ConcurrentProfile, RateProfile
from luban_meter.runtime.request_scheduler import (
    AdmissionStopped,
    RequestScheduler,
)


class Handler(BaseHTTPRequestHandler):
    posts = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path == "/metrics":
            self.wfile.write(
                b"# TYPE vllm:num_requests_running gauge\n"
                b"vllm:num_requests_running 1\n"
            )
        else:
            self.wfile.write(b'{"data":[{"id":"mock"}]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.posts.append((self.path, body))
        if self.path == "/tokenize":
            response = {"tokens": [1, 2, 3], "count": 3, "max_model_len": 4096}
        elif body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            output_count = max(1, body["max_tokens"] - self.output_shortfall)
            for i in range(output_count):
                choice = (
                    {"delta": {"content": "x"}}
                    if "messages" in body
                    else {"text": "x"}
                )
                payload = {"choices": [choice]}
                self.wfile.write(
                    ("data: " + json.dumps(payload) + "\n\n").encode()
                )
                self.wfile.flush()
                time.sleep(0.002)
            usage = {
                "choices": [],
                "usage": {
                    "prompt_tokens": len(
                        body.get("prompt") or body["messages"][0]["content"]
                    )
                    + self.input_delta,
                    "completion_tokens": output_count,
                },
            }
            if self.omit_usage:
                usage = {"choices": []}
            self.wfile.write(
                (
                    "data: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n"
                ).encode()
            )
            return
        else:
            response = {
                "choices": [{"text": "A", "message": {"content": "A"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


@pytest.fixture
def service(tmp_path):
    from tests.tokenizer_fixture import make_tokenizer

    Handler.tokenizer_path = str(tmp_path / "tokenizer")
    make_tokenizer(Handler.tokenizer_path)
    Handler.posts = []
    Handler.input_delta = 0
    Handler.output_shortfall = 0
    Handler.omit_usage = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def serving(base_url):
    return {
        "scenario": "online_serving",
        "model": "mock",
        "tokenizer": Handler.tokenizer_path,
        "base_url": base_url,
        "input_tokens": 8,
        "output_tokens": 3,
        "num_requests": 4,
        "profile": "concurrent",
        "max_concurrency": 2,
    }


def test_mixed_tasks_and_html(service, tmp_path):
    dataset = tmp_path / "choice.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "1",
                "question": "Q",
                "choices": ["a", "b", "c", "d"],
                "answer": "A",
                "subject": "test",
            }
        )
        + "\n"
    )
    task_file = tmp_path / "tasks.json"
    task_file.write_text(
        json.dumps(
            {
                "tasks": [
                    {"name": "perf", **serving(service)},
                    {
                        "name": "quality",
                        "scenario": "ceval",
                        "dataset_path": str(dataset),
                        "few_shot": 0,
                        "metrics": ["choice_accuracy", "sample_accuracy"],
                    },
                ]
            }
        )
    )
    plan = build_plan({"model": "mock", "base_url": service}, task_file)
    assert "input_tokens" not in plan["tasks"][1]["cases"][0]["parameters"]
    result, directory = RunCoordinator().run(plan, tmp_path / "runs")
    assert result["status"] == "success", result
    assert [t["name"] for t in result["tasks"]] == ["perf", "quality"]
    case = result["tasks"][0]["cases"][0]
    assert case["window"]["admitted_requests"] == 4
    assert case["window"]["peak_requests"] <= 2
    assert case["coverage"]["complete"] == 4
    assert "schema_version" not in json.dumps(result)
    assert load_result(directory / "result.json") == result
    html = write_html(result, directory / "report.html").read_text()
    assert "<svg" not in html  # SVG is generated from embedded data offline.
    assert "report-data" in html and "cdn." not in html
    assert (directory / "tasks/perf/raw/events.jsonl").is_file()


def test_case_matrix_and_cli_override(tmp_path):
    config = tmp_path / "tasks.json"
    config.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "first",
                        "input_tokens": 4,
                        "cases": {
                            "matrix": {
                                "output_tokens": [2, 4],
                                "request_rate": [1, 2],
                            }
                        },
                    },
                    {"name": "second"},
                ]
            }
        )
    )
    plan = build_plan(
        {"scenario": "online_serving", "model": "mock", "input_tokens": 8},
        config,
    )
    assert len(plan["tasks"][0]["cases"]) == 4
    assert plan["tasks"][1]["cases"][0]["parameters"]["input_tokens"] == 8
    config.write_text(
        json.dumps({"tasks": [{"scenario": "online_serving", "oops": 1}]})
    )
    with pytest.raises(ValueError, match="oops"):
        build_plan({"model": "mock"}, config)


def test_flat_cli_plan(capsys):
    assert (
        main(
            [
                "plan",
                "--scenario",
                "online_serving",
                "--model",
                "mock",
                "--input-tokens",
                "128",
                "--request-rate",
                "inf",
                "--http-max-connections",
                "128",
                "--http-keepalive-expiry",
                "7.5",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["tasks"][0]["cases"][0]["parameters"]["input_tokens"] == 128
    parameters = output["tasks"][0]["cases"][0]["parameters"]
    assert parameters["http_max_connections"] == 128
    assert parameters["http_keepalive_expiry"] == 7.5
    with pytest.raises(SystemExit) as error:
        main(["plan", "--module", "generate"])
    assert error.value.code == 2


def test_metric_requires_facts():
    with pytest.raises(ValueError, match="requires"):
        build_plan(
            {
                "scenario": "online_serving",
                "model": "mock",
                "metrics": ["code_pass_rate"],
            }
        )


@pytest.mark.parametrize(
    "base", [GenerationPerformanceBenchmark, ModelServiceQualityBenchmark]
)
def test_public_inheritance_and_custom_metric(base, tmp_path):
    bootstrap()
    name = "example_" + base.category
    metric_name = name + "_metric"

    class CustomMetric(Metric):
        requires = frozenset({"custom_score"})

        def compute(self, facts):
            rows = [r for r in facts["samples"] if r["state"] == "complete"]
            return {
                "score": {
                    "value": sum(r["facts"]["score"] for r in rows),
                    "unit": "point",
                    "sample_count": len(rows),
                }
            }

    class CustomBenchmark(base):
        parameters = {"amount": Parameter(int, 3, minimum=1)}
        capabilities = frozenset({"custom_score"})
        default_metrics = (metric_name,)

        def samples(self, context):
            return [{"n": n} for n in range(context.parameters["amount"])]

        def evaluate(self, sample, context):
            return {"score": sample["n"]}

    benchmarks.register(name)(CustomBenchmark)
    metrics.register(metric_name)(CustomMetric)
    plan = build_plan({"scenario": name, "amount": 4})
    task = plan["tasks"][0]
    directory = tmp_path / name
    (directory / "raw").mkdir(parents=True)
    result = run_case(
        CustomBenchmark, task["cases"][0], directory, directory / "stop"
    )
    assert result["status"] == "success", result
    assert result["metrics"][metric_name]["score"]["value"] == 6
    assert result["coverage"]["complete"] == 4
    with pytest.raises(ValueError, match="duplicate"):
        benchmarks.register(name)(CustomBenchmark)


def test_scheduler_request_budget_and_seed():
    p = {
        "request_rate": 1000,
        "arrival_process": "gamma",
        "burstiness": 0.5,
        "seed": 9,
        "max_concurrency": 2,
        "num_requests": 2,
    }
    first, second = (RateProfile(p).create_strategy() for _ in range(2))
    assert [first.plan({}).offset_s for _ in range(5)] == [
        second.plan({}).offset_s for _ in range(5)
    ]
    scheduler = RequestScheduler(
        ConcurrentProfile(p), p, lambda: False, lambda *args: None
    )
    assert scheduler.run(lambda t: t.index, sample_id="x") == 0
    assert scheduler.run(lambda t: t.index, sample_id="x") == 1
    with pytest.raises(AdmissionStopped):
        scheduler.run(lambda t: t.index, sample_id="x")


def test_old_result_rejected_and_text_escaped(tmp_path):
    old = tmp_path / "old.json"
    old.write_text('{"schema_version":"luban-meter.result/v2"}')
    with pytest.raises(ValueError):
        load_result(old)
    value = {
        "kind": "run",
        "id": "</script><script>alert(1)</script>",
        "status": "success",
        "tasks": [],
    }
    html = write_html(value, tmp_path / "report.html").read_text()
    assert value["id"] not in html
    assert "\\u003c/script>" in html


def test_offline_batches_keep_actual_dimensions(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from luban_meter.benchmarking.generation_performance import (
        generation_benchmarks as scenarios,
    )
    from luban_meter.benchmarking.generation_performance import (
        vllm_engine as sampling,
    )

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            SamplingParams=lambda **kw: kw,
        ),
    )
    from tests.tokenizer_fixture import make_tokenizer

    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    engine = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    monkeypatch.setattr(sampling, "build_engine", lambda *args: engine)
    monkeypatch.setattr(sampling, "engine_max_model_len", lambda e: 1024)
    monkeypatch.setattr(
        sampling,
        "kv_cache_environment",
        lambda e: {
            "num_gpu_blocks": 8,
            "block_size": 16,
            "kv_cache_size_tokens": 128,
            "kv_cache_max_concurrency": 4,
        },
    )
    batches = []

    def generate(engine, params, prompts):
        size = len(prompts)
        batches.append(size)
        return [
            {
                "request_index": i,
                "actual_prompt_tokens": 4,
                "actual_output_tokens": 3,
                "internal_ttft_seconds": 0.1,
                "scheduled_ts": 10.0,
                "first_token_ts": 10.1,
                "last_token_ts": 10.3,
            }
            for i in range(size)
        ]

    monkeypatch.setattr(sampling, "generate_batch", generate)
    plan = build_plan(
        {
            "scenario": "offline_vllm_engine",
            "model": "mock",
            "input_tokens": 4,
            "output_tokens": 3,
            "num_requests": 5,
            "batch_size": 2,
        }
    )
    (tmp_path / "raw").mkdir()
    result = run_case(
        scenarios.OfflineEngineBenchmark,
        plan["tasks"][0]["cases"][0],
        tmp_path,
        tmp_path / "stop",
    )
    assert result["status"] == "success", result
    assert batches == [2, 2, 1]
    assert result["coverage"]["complete"] == 5
    assert result["window"]["admitted_requests"] == 5


def test_observer_and_case_circuit_breaker(service, tmp_path):
    config = tmp_path / "tasks.json"
    config.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        **serving(service),
                        "num_requests": 12,
                        "circuit_breaker": 0.001,
                        "cases": {
                            "items": [{"input_tokens": 8}, {"input_tokens": 9}]
                        },
                    }
                ]
            }
        )
    )
    plan = build_plan({}, config, observe=True)
    result, _ = RunCoordinator().run(
        plan,
        tmp_path / "runs",
        observation={"url": service + "/metrics", "interval": 0.01},
    )
    first, second = result["tasks"][0]["cases"]
    assert first["status"] == "success", first
    assert first["observation"]["valid_samples"] >= 2
    assert second["status"] == "skipped"
    assert second["reason"] == "circuit_breaker_exceeded"


def test_invalid_data_source_is_rejected_before_execution():
    with pytest.raises(ValueError, match="data source"):
        build_plan({**serving("http://127.0.0.1:1"), "data": "missing"})


@pytest.mark.parametrize("process", ["constant", "poisson", "gamma"])
def test_profile_arrivals_are_positive_reproducible(process):
    p = {
        "scenario": "online_serving",
        "model": "mock",
        "arrival_process": process,
        "request_rate": 10,
        "burstiness": 0.5,
    }
    values = build_plan(p)["tasks"][0]["cases"][0]["parameters"]
    first, second = (RateProfile(values).create_strategy() for _ in range(2))
    offsets = [first.plan({}).offset_s for _ in range(101)]
    intervals = [b - a for a, b in zip(offsets, offsets[1:])]
    assert all(value > 0 for value in intervals)
    assert offsets == [second.plan({}).offset_s for _ in range(101)]
    if process == "constant":
        assert intervals == pytest.approx([0.1] * 100)


def test_concurrent_has_no_artificial_arrival_backlog():
    p = {"max_concurrency": 1}
    scheduler = RequestScheduler(
        ConcurrentProfile(p),
        p,
        lambda: False,
        lambda *args: None,
    )
    first = scheduler.run(lambda ticket: ticket, sample_id="1")
    time.sleep(0.01)
    second = scheduler.run(lambda ticket: ticket, sample_id="2")
    assert first.started == first.scheduled
    assert second.started == second.scheduled
    assert second.scheduled > first.scheduled


def test_new_profile_works_without_core_scenario_branch():
    from luban_meter.api import profiles

    @profiles.register("test_constant")
    class CustomProfile(RateProfile):
        parameters = {"tick": Parameter(float, 0.1, minimum=0.001)}

        def create_strategy(self):
            from luban_meter.runtime.scheduling_strategies import RateStrategy

            return RateStrategy({
                **self.parameters,
                "request_rate": 1 / self.parameters["tick"],
            })

    plan = build_plan(
        {
            "scenario": "online_serving",
            "model": "mock",
            "profile": "test_constant",
            "tick": 0.2,
        }
    )
    p = plan["tasks"][0]["cases"][0]["parameters"]
    assert p["max_concurrency"] == 32
    strategy = CustomProfile(p).create_strategy()
    assert strategy.plan({}).offset_s == 0
    assert strategy.plan({}).offset_s == 0.2


def test_duration_stops_admission_and_drains(service, tmp_path):
    p = serving(service)
    p.update(profile="rate", request_rate=0.5, max_duration=0.05)
    result, _ = RunCoordinator().run(build_plan(p), tmp_path)
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    assert case["window"]["admitted_requests"] == 1
    assert case["window"]["stop_reason"] == "max_duration"
    assert case["coverage"]["complete"] == 1


def test_observation_failure_does_not_fail_inference(service, tmp_path):
    result, directory = RunCoordinator().run(
        build_plan(serving(service)),
        tmp_path,
        observation={"url": "http://127.0.0.1:1/metrics", "interval": 0.01},
    )
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    assert case["observation"]["status"] == "failed"
    assert case["observation"]["failed_samples"] >= 2
    text = next((directory / "log").glob("luban-meter-*.log")).read_text()
    failures = case["observation"]["failed_samples"]
    assert text.count("Prometheus sampling failed") == failures
    assert "Traceback (most recent call last)" in text
    assert "task=task_1 case=case_1" in text
    assert "Prometheus observation stopped" in text


def test_html_bounds_observation_data_without_mutating_result():
    from luban_meter.reporting.html import report_data

    snapshots = [
        {"finished_at": i, "metrics": {}, "error": "unavailable"}
        for i in range(1000)
    ]
    payload = {
        "kind": "task",
        "status": "success",
        "name": "demo",
        "category": "generation_performance",
        "cases": [
            {
                "id": "case_1",
                "status": "success",
                "observation": {
                    "snapshots": snapshots,
                    "failed_samples": 1000,
                },
            },
        ],
    }
    rendered = report_data(payload)["result"]["cases"][0]["observation"]
    assert "snapshots" not in rendered
    assert len(rendered["failure_examples"]) == 20
    assert rendered["failed_samples"] == 1000
    assert len(payload["cases"][0]["observation"]["snapshots"]) == 1000
