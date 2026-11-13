"""Saved interface values are rejected before any scheduled model call."""

import copy
import json
import time
from pathlib import Path

import pytest

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import build_plan
from luban_meter.core.schedule_commands import register, update
from luban_meter.core.schedule_store import ScheduleStore
from luban_meter.core.scheduled_runs import ScheduleService
from tests.test_endpoint_types import task_parameters
from tests.test_unified_runtime import Handler, service as mock_service


@pytest.mark.parametrize("scenario", [
    "ceval", "cmmlu", "gsm8k", "lcsts", "squad",
])
@pytest.mark.parametrize("endpoint", [
    "/v1/chat/completions", "/v1/completions", "unknown",
])
def test_quality_domain_rejects_old_or_unknown_saved_interface(
    tmp_path, scenario, endpoint,
):
    plan = build_plan(task_parameters(tmp_path, scenario))
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    parameters["endpoint"] = endpoint
    with pytest.raises(ValueError, match="endpoint must be one of"):
        benchmarks.get(scenario).domain_parameters(parameters)


@pytest.fixture
def service(tmp_path):
    yield from mock_service.__wrapped__(tmp_path)


@pytest.mark.parametrize("scenario", ["wikitext", "humaneval"])
def test_quality_domain_keeps_completion_only_restriction(tmp_path, scenario):
    plan = build_plan(task_parameters(tmp_path, scenario))
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    parameters["endpoint"] = "chat"
    with pytest.raises(ValueError, match="endpoint must be one of"):
        benchmarks.get(scenario).domain_parameters(parameters)


@pytest.mark.parametrize("task_count", [1, 2])
@pytest.mark.parametrize("endpoint", [
    "/v1/chat/completions", "/v1/completions",
])
def test_old_saved_schedule_fails_without_requests_then_can_be_updated(
    tmp_path, service, endpoint, task_count,
):
    parameters = task_parameters(
        tmp_path, "cmmlu", base_url=service, endpoint="chat",
        profile="synchronous", num_requests=1, api_key="saved-test-secret",
    )
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"tasks": [parameters] * task_count}))
    store = ScheduleStore(tmp_path)
    now = time.time()
    plan = register(
        store, config, cron="* * * * *", zone="UTC", now=now,
    )
    # Reproduce the persisted resolved values written by the old version.
    with store.transaction() as records:
        saved = copy.deepcopy(records.get_plan(plan["id"]))
        saved["resolved"]["tasks"][-1]["cases"][0][
            "parameters"
        ]["endpoint"] = endpoint
        saved = records.save_plan(saved, now)
    original = copy.deepcopy(saved["resolved"])
    # The source JSON already uses the new name; saved snapshots are separate.
    assert json.loads(config.read_text())["tasks"][0]["endpoint"] == "chat"
    dispatcher = ScheduleService(store)
    try:
        due = now + 60
        dispatcher.tick(due)
        assert dispatcher.process is not None
        code = dispatcher.process.wait(timeout=20)
        dispatcher.tick(due)
        assert Handler.posts == []
        assert code == 2
        with store.transaction() as records:
            failed, = records.executions.values()
            assert failed["status"] == "failed"
            assert records.revisions[saved["revision"]][
                "resolved"
            ] == original
        log_path = Path(failed["log_path"])
        log = log_path.read_text()
        assert "endpoint must be one of ('completion', 'chat')" in log
        assert "Saved Case parameter validation failed" in log
        assert "update or recreate the schedule with a valid task JSON" in log
        assert f"task=task_{task_count} case=case_1 phase=validate" in log
        assert "Traceback (most recent call last)" in log
        assert "saved-test-secret" not in log
        assert not list(log_path.parent.parent.glob("tasks/*"))

        # Explicit update replaces only the future snapshot and can run again.
        changed = update(store, plan["id"], config=config, now=due)
        assert changed["resolved"]["tasks"][0]["cases"][0][
            "parameters"
        ]["endpoint"] == "chat"
        dispatcher.tick(due + 60)
        assert dispatcher.process is not None
        assert dispatcher.process.wait(timeout=20) == 0
        dispatcher.tick(due + 60)
        assert len(Handler.posts) == task_count
        for path, body in Handler.posts:
            assert path == "/v1/chat/completions"
            assert "messages" in body and "prompt" not in body
        with store.transaction() as records:
            states = [r["status"] for r in records.executions.values()]
            assert sorted(states) == ["failed", "success"]
    finally:
        if dispatcher.process is not None:
            store.request_stop(dispatcher.execution_id)
            dispatcher.process.wait(timeout=20)
