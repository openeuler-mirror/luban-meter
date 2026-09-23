"""Explicit strategy contracts and shared admission behavior."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from luban_meter.core.planning import build_plan
from luban_meter.runtime.load_profiles import Profile, profiles
from luban_meter.runtime.request_scheduler import (
    AdmissionStopped,
    RequestScheduler,
)
from luban_meter.runtime.scheduling_strategies import (
    AsyncConstantStrategy,
    AsyncGammaStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    RateStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
    TraceReplayStrategy,
)


def resolve(profile, **options):
    parameters = build_plan({
        "scenario": "online_serving", "model": "mock",
        "profile": profile, **options,
    })["tasks"][0]["cases"][0]["parameters"]
    return profiles.get(profile)(parameters), parameters


@pytest.mark.parametrize("name, options, implementation", [
    ("constant", {}, AsyncConstantStrategy),
    ("poisson", {}, AsyncPoissonStrategy),
    ("concurrent", {}, ConcurrentStrategy),
    ("throughput", {}, ThroughputStrategy),
    ("synchronous", {}, SynchronousStrategy),
    ("rate", {"arrival_process": "constant"}, AsyncConstantStrategy),
    ("rate", {"arrival_process": "poisson"}, AsyncPoissonStrategy),
    ("rate", {"arrival_process": "gamma"}, AsyncGammaStrategy),
])
def test_public_entrypoints_resolve_concrete_strategies(
    name, options, implementation
):
    profile, _ = resolve(name, **options)
    assert type(profile.create_strategy()) is implementation


def test_replay_and_batch_keep_their_contracts():
    replay = profiles.get("replay")({"replay_speed": 2})
    strategy = replay.create_strategy()
    assert isinstance(strategy, TraceReplayStrategy)
    plan = strategy.plan({"offset_s": 4, "trace_index": 0})
    assert plan.offset_s == 2
    assert not strategy.ready(plan, 1)
    assert strategy.ready(plan, 2)
    batch = profiles.get("batch")({"batch_size": 8})
    assert batch.execution_mode == "batch"
    assert batch.workers == 1
    assert batch.create_strategy().plan({}).offset_s is None


@pytest.mark.parametrize("name", ["constant", "poisson"])
@pytest.mark.parametrize("value", [0, -1, "inf", "nan", True])
def test_explicit_arrival_rate_must_be_positive_finite(name, value):
    with pytest.raises(ValueError):
        resolve(name, request_rate=value)


@pytest.mark.parametrize("name", [
    "constant", "poisson", "concurrent", "throughput", "synchronous",
])
@pytest.mark.parametrize("value", [0, -1, True])
def test_capacity_validation(name, value):
    with pytest.raises(ValueError):
        resolve(name, max_concurrency=value)


def test_synchronous_rejects_conflicting_concurrency():
    with pytest.raises(ValueError, match="max_concurrency"):
        resolve("synchronous", max_concurrency=2)


@pytest.mark.parametrize("name", ["constant", "poisson", "throughput"])
def test_explicit_profiles_reject_unrelated_options(name):
    with pytest.raises(ValueError, match="unknown parameters"):
        resolve(name, arrival_process="gamma")


@pytest.mark.parametrize("name", [
    "constant", "poisson", "concurrent", "throughput", "synchronous",
])
def test_request_profiles_reject_synchronous_engine(name):
    with pytest.raises(ValueError, match="incompatible profile"):
        build_plan({
            "scenario": "offline_vllm_engine", "model": "mock",
            "profile": name,
        })


@pytest.mark.parametrize("process", ["constant", "poisson", "gamma"])
def test_legacy_rate_adapter_and_warmup_preserve_sequence(process):
    profile, parameters = resolve(
        "rate", arrival_process=process, request_rate=10,
        burstiness=0.5, seed=17,
    )
    for _ in range(20):
        profile.create_warmup_strategy().plan({})
    strategy = profile.create_strategy()
    legacy = RateStrategy(parameters)
    offsets = [strategy.plan({}).offset_s for _ in range(200)]
    assert offsets[0] == 0
    assert offsets == [legacy.plan({}).offset_s for _ in range(200)]
    assert all(b > a for a, b in zip(offsets, offsets[1:]))
    if process == "constant":
        assert offsets == pytest.approx([index / 10 for index in range(200)])


def test_poisson_seed_and_mean_interval():
    def offsets(seed):
        profile, _ = resolve("poisson", request_rate=10, seed=seed)
        strategy = profile.create_strategy()
        return [strategy.plan({}).offset_s for _ in range(10001)]

    first = offsets(17)
    assert first == offsets(17)
    assert first != offsets(18)
    assert first[-1] / 10000 == pytest.approx(0.1, rel=0.04)


@pytest.mark.parametrize("name, capacity", [
    ("synchronous", 1), ("concurrent", 3), ("throughput", 3),
])
def test_capacity_refill_and_failure_release(name, capacity):
    profile, parameters = resolve(name, max_concurrency=capacity)
    records = []
    scheduler = RequestScheduler(
        profile, parameters, lambda: False,
        lambda event, row: records.append((event, row)),
    )
    release = threading.Event()
    filled = threading.Event()
    lock = threading.Lock()
    active = peak = entered = 0

    def operation(ticket):
        nonlocal active, peak, entered
        with lock:
            active += 1
            entered += 1
            peak = max(peak, active)
            if active == capacity:
                filled.set()
        try:
            assert release.wait(5)
            if ticket.index == 0:
                raise TimeoutError("synthetic timeout")
            return ticket
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(scheduler.run, operation, sample_id=str(index))
            for index in range(8)
        ]
        try:
            assert filled.wait(5)
            with lock:
                assert entered == capacity
        finally:
            release.set()
        failures = 0
        for future in futures:
            try:
                ticket = future.result(timeout=5)
                assert ticket.scheduled == ticket.started
            except TimeoutError:
                failures += 1
    assert failures == 1
    assert entered == 8
    assert peak == capacity
    assert scheduler.active == 0
    finished = [row for event, row in records if event == "request_finished"]
    assert len(finished) == 8
    assert sum(row["status"] == "failed" for row in finished) == 1


def test_strategy_capacity_is_enforced_for_custom_profile():
    class WideProfile(Profile):
        def create_strategy(self):
            return SynchronousStrategy()

    parameters = {"max_concurrency": 8}
    scheduler = RequestScheduler(
        WideProfile(parameters), parameters, lambda: False, lambda *a: None
    )
    assert scheduler.capacity == 1


@pytest.mark.parametrize("name", [
    "constant", "poisson", "concurrent", "throughput", "synchronous",
    "replay",
])
def test_request_budget_stops_all_explicit_strategies(name):
    parameters = {
        "request_rate": 10, "max_concurrency": 1,
        "replay_speed": 1, "num_requests": 1,
    }
    scheduler = RequestScheduler(
        profiles.get(name)(parameters), parameters,
        lambda: False, lambda *a: None,
    )
    calls = []
    scheduler.run(
        lambda ticket: calls.append(ticket.index), sample_id="0",
        metadata={"offset_s": 0, "trace_index": 0},
    )
    with pytest.raises(AdmissionStopped):
        scheduler.run(
            lambda ticket: calls.append(ticket.index), sample_id="1",
            metadata={"offset_s": 100, "trace_index": 1},
        )
    assert calls == [0]
    assert scheduler.active == 0
    assert scheduler.reason == "num_requests"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_strategy_capacity_is_rejected(limit):
    class InvalidProfile(Profile):
        def create_strategy(self):
            return ConcurrentStrategy(limit)

    with pytest.raises(ValueError, match="concurrency_limit"):
        RequestScheduler(
            InvalidProfile({}), {}, lambda: False, lambda *a: None
        )
