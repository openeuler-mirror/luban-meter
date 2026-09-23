"""Paired trace records, transport-neutral scheduling and Engine lifecycle."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from luban_meter.benchmarking.generation_performance.async_engine import (
    AsyncEngineBackend,
)
from luban_meter.core.planning import build_plan
from luban_meter.data import DatasetLoader
from luban_meter.data.trace_adapter import select_trace
from luban_meter.runtime.load_profiles import ReplayProfile
from luban_meter.runtime.request_scheduler import (
    AdmissionStopped,
    RequestScheduler,
)
from tests.test_unified_runtime import service as _service


@pytest.fixture
def service(tmp_path):
    yield from _service.__wrapped__(tmp_path)



def trace_file(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text(
        "timestamp,input_token,output_token\n"
        "102,30,7\n100,10,5\n100,20,6\n105,40,8\n"
    )
    return path


def trace_parameters(**changes):
    return {
        "profile": "replay",
        "replay_speed": 2.0,
        "num_requests": 4,
        "max_concurrency": 1,
        "shuffle": False,
        "seed": 42,
        **changes,
    }


def test_trace_sort_scale_and_pairing(tmp_path):
    records = DatasetLoader(trace_file(tmp_path), format="burstgpt").load()
    selected = select_trace(records, trace_parameters())
    assert [r["input_tokens"] for r in selected] == [10, 20, 30, 40]
    assert [r["output_tokens"] for r in selected] == [5, 6, 7, 8]
    strategy = ReplayProfile(trace_parameters()).create_strategy()
    assert [strategy.plan(r["scheduling"]).offset_s for r in selected] == [
        0,
        0,
        1,
        2.5,
    ]
    assert selected[0]["timestamp"] == "100"
    assert selected[0]["source_line"] == 3
    with pytest.raises(ValueError, match="shuffle"):
        select_trace(records, trace_parameters(shuffle=True))
    with pytest.raises(ValueError, match="usable"):
        select_trace(records, trace_parameters(num_requests=5))


def test_trace_skip_keeps_original_gaps_and_diagnostics(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,input_token,output_token\n10,4,2\n11,2.5,2\n14,8,3\n"
    )
    with pytest.raises(ValueError, match="input_token") as failure:
        DatasetLoader(path, format="burstgpt").load()
    assert str(path) in str(failure.value) and "2.5" in str(failure.value)
    errors = []
    records = DatasetLoader(
        path, format="burstgpt", skip_invalid=True, diagnostics=errors
    ).load()
    assert errors[0]["line"] == 3
    assert [r["offset_s"] for r in records] == [10, 14]
    assert (
        select_trace(records, trace_parameters(num_requests=2))[1][
            "scheduling"
        ]["offset_s"]
        == 4
    )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "", "True"])
def test_invalid_trace_lengths_are_not_coerced(tmp_path, value):
    path = tmp_path / "invalid.csv"
    path.write_text(f"timestamp,input_token,output_token\n0,4,{value}\n")
    with pytest.raises(ValueError, match="output_token"):
        DatasetLoader(path, format="burstgpt").load()


class FakeAsyncEngine:
    def __init__(self):
        self.model_config = SimpleNamespace(max_model_len=4096)
        self.closed = False
        self.aborted = []
        self.submissions = []

    def get_tokenizer(self):
        return object()

    async def generate(self, prompt, settings, request_id):
        self.submissions.append((request_id, prompt))
        await asyncio.sleep(settings.delay)
        yield SimpleNamespace(
            prompt_token_ids=prompt["prompt_token_ids"],
            outputs=[SimpleNamespace(token_ids=list(range(settings.budget)))],
            metrics=None,
        )

    async def abort(self, request_id):
        self.aborted.append(request_id)

    def shutdown(self):
        self.closed = True


def run_replay(backend):
    p = trace_parameters(num_requests=3)
    events = []
    scheduler = RequestScheduler(
        ReplayProfile(p),
        p,
        lambda: False,
        lambda kind, row: events.append((kind, dict(row))),
    )
    rows = [(0, 4, 2), (0.002, 6, 3), (0.004, 8, 4)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(
                scheduler.run,
                lambda ticket, row=row: backend(row),
                sample_id=str(i),
                metadata={"trace_index": i, "offset_s": row[0]},
            )
            for i, row in enumerate(rows)
        ]
        outputs = [job.result(timeout=5) for job in jobs]
    finished = [r for k, r in events if k == "request_finished"]
    return outputs, finished


def test_http_callable_and_async_engine_share_replay_schedule():
    def http(row):
        time.sleep(0.02)
        return {"input_tokens": row[1], "output_tokens": row[2]}

    fake = FakeAsyncEngine()
    engine = AsyncEngineBackend({}, factory=lambda: fake)
    try:

        def submit(row):
            return engine.generate(
                {"prompt_token_ids": list(range(row[1]))},
                SimpleNamespace(delay=0.02, budget=row[2]),
                1,
            )

        http_rows, http_events = run_replay(http)
        engine_rows, engine_events = run_replay(submit)
        for rows in (http_rows, engine_rows):
            assert [(r["input_tokens"], r["output_tokens"]) for r in rows] == [
                (4, 2),
                (6, 3),
                (8, 4),
            ]
        for events in (http_events, engine_events):
            assert [r["scheduled_offset_s"] for r in events] == pytest.approx(
                [0, 0.001, 0.002], abs=1e-7
            )
            assert events[1]["scheduling_delay_s"] > 0.01
            assert [r["sample_id"] for r in events] == ["0", "1", "2"]
        assert all(not r["engine_timestamps_available"] for r in engine_rows)
        assert all(r["submitted_at_monotonic_s"] > 0 for r in engine_rows)
    finally:
        engine.close()
    assert fake.closed and not engine.thread.is_alive()


def test_async_timeout_aborts_only_current_request():
    fake = FakeAsyncEngine()
    engine = AsyncEngineBackend({}, factory=lambda: fake)
    try:
        with pytest.raises(TimeoutError):
            engine.generate(
                {"prompt_token_ids": [1]},
                SimpleNamespace(delay=1, budget=1),
                0.01,
            )
        assert len(fake.aborted) == 1
    finally:
        engine.close()


def test_replay_duration_stops_waiting_requests():
    p = trace_parameters(max_duration=0.03)
    scheduler = RequestScheduler(
        ReplayProfile(p), p, lambda: False, lambda *args: None
    )
    scheduler.run(
        lambda ticket: time.sleep(0.04),
        sample_id="0",
        metadata={"trace_index": 0, "offset_s": 0},
    )
    with pytest.raises(AdmissionStopped, match="max_duration"):
        scheduler.run(
            lambda ticket: pytest.fail("must not submit"),
            sample_id="1",
            metadata={"trace_index": 1, "offset_s": 0.001},
        )


def test_sync_batch_rejects_replay_before_engine_start(tmp_path):
    path = trace_file(tmp_path)
    with pytest.raises(ValueError, match="incompatible profile"):
        build_plan(
            {
                "scenario": "offline_vllm_engine",
                "model": "unused",
                "data": "burstgpt",
                "dataset_path": str(path),
                "profile": "replay",
            }
        )
    plan = build_plan(
        {
            "scenario": "offline_vllm_async",
            "model": "unused",
            "data": "burstgpt",
            "dataset_path": str(path),
            "profile": "replay",
            "num_requests": 4,
        }
    )
    json.dumps(plan)


def test_real_http_trace_replay_preserves_paired_lengths(
    service, tmp_path
):
    from luban_meter.core.run_coordinator import RunCoordinator
    from tests.test_unified_runtime import serving

    path = trace_file(tmp_path)
    parameters = {
        **serving(service),
        "data": "burstgpt",
        "dataset_path": str(path),
        "profile": "replay",
        "replay_speed": 1000,
        "num_requests": 4,
        "max_concurrency": 1,
    }
    result, _ = RunCoordinator().run(build_plan(parameters), tmp_path / "run")
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    from pathlib import Path

    facts = json.loads(Path(case["raw"]).read_text())
    assert [
        (
            r["source"]["target_input_tokens"],
            r["source"]["target_output_tokens"],
        )
        for r in facts["samples"]
    ] == [(10, 5), (20, 6), (30, 7), (40, 8)]
    assert [
        r["scheduled_offset_s"] for r in facts["requests"]
    ] == pytest.approx([0, 0, 0.002, 0.005], abs=1e-7)


def test_async_engine_scenario_runs_same_trace(tmp_path, monkeypatch):
    import sys

    from luban_meter.benchmarking.generation_performance import (
        async_engine,
        generation_benchmarks,
    )
    from luban_meter.runtime.evaluation_session import run_case
    from tests.tokenizer_fixture import make_tokenizer

    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    fake = FakeAsyncEngine()
    fake.get_tokenizer = lambda: tokenizer
    original = AsyncEngineBackend
    monkeypatch.setattr(
        async_engine,
        "AsyncEngineBackend",
        lambda p: original(p, factory=lambda: fake),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            SamplingParams=lambda **p: SimpleNamespace(
                delay=0.005, budget=p["max_tokens"]
            ),
        ),
    )
    plan = build_plan(
        {
            "scenario": "offline_vllm_async",
            "model": "mock",
            "data": "burstgpt",
            "dataset_path": str(trace_file(tmp_path)),
            "num_requests": 4,
            "num_warmups": 1,
            "profile": "replay",
            "replay_speed": 1000,
            "max_concurrency": 1,
        }
    )
    (tmp_path / "raw").mkdir()
    case = run_case(
        generation_benchmarks.AsyncEngineBenchmark,
        plan["tasks"][0]["cases"][0],
        tmp_path,
        tmp_path / ".stop",
    )
    assert case["status"] == "success", case
    facts = json.loads((tmp_path / "raw/case_1.json").read_text())
    assert [
        r["scheduled_offset_s"] for r in facts["requests"]
    ] == pytest.approx([0, 0, 0.002, 0.005], abs=1e-7)
    assert [
        (r["facts"]["input_tokens"], r["facts"]["output_tokens"])
        for r in facts["samples"]
    ] == [(10, 5), (20, 6), (30, 7), (40, 8)]
    assert case["metrics"]["async_engine_performance"]["ttft"]["count"] == 4
    assert fake.closed
