"""Calendar boundaries, daily persistence and durable scheduling decisions."""

import csv
import json
from datetime import datetime, timezone

import pytest

from luban_meter.core import schedule_commands as commands
from luban_meter.core.schedule_store import ScheduleStore
from luban_meter.core.schedule_time import stamp, timestamp, timing
from luban_meter.core.scheduled_runs import ScheduleService

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "snapshot",
                        lambda path: {"tasks": [], "case_count": 0})
    return ScheduleStore(tmp_path)


def register(store, offset=60, **options):
    return commands.register(
        store, "task.json", at=stamp(NOW + offset), zone="UTC", now=NOW,
        **options,
    )


@pytest.mark.parametrize("at,cron,zone", [
    (stamp(NOW - 1), None, "UTC"),
    (stamp(NOW), None, "UTC"),
    (None, None, "UTC"),
    (stamp(NOW + 60), "* * * * *", "UTC"),
    (None, "* * * * * *", "UTC"),
    (None, "99 * * * *", "UTC"),
    (None, "0 0 31 2 *", "UTC"),
    (stamp(NOW + 60), None, "Not/A_Timezone"),
    ("2030-03-10 02:30", None, "America/New_York"),
    ("2030-11-03 01:30", None, "America/New_York"),
])
def test_reject_invalid_calendar(at, cron, zone):
    with pytest.raises(ValueError):
        timing(at, cron, zone, NOW)


def test_cron_calendar_and_fixed_zone():
    plan = timing(None, "30 2 * * *", "Asia/Shanghai", NOW)
    assert timestamp(plan["next_at"]) == NOW + 18.5 * 3600
    # Both DOM and DOW follow cron's OR semantics, not AND.
    plan = timing(None, "0 0 2 * mon", "UTC", NOW)
    assert timestamp(plan["next_at"]) == NOW + 86400


def test_daily_json_csv_history_and_clock_rollback(store):
    plan = register(store)
    with store.transaction() as records:
        row = records.new_execution(plan, plan["next_at"], "queued", NOW)
        row = records.save_execution(
            dict(row, status="success", summary='a,b\n"quoted"'),
            NOW + 86400,
        )
        # A later record remains authoritative after a wall-clock rollback.
        records.save_execution(dict(row, status="cancelled"), NOW - 86400)
    with store.transaction() as records:
        assert records.executions[row["execution_id"]]["status"] == "cancelled"
        assert records.revisions[plan["revision"]]["resolved"]["tasks"] == []
        assert 'a,b\n"quoted"' in commands.render_list(records)
    directories = sorted(store.root.glob("????-??-??"))
    assert len(directories) == 3
    assert all((d / "plans.json").exists() for d in directories)
    assert all((d / "executions.csv").exists() for d in directories)
    for day in directories:
        saved = json.loads((day / "plans.json").read_text())
        assert saved[0]["id"] == plan["id"]
    with (directories[2] / "executions.csv").open(newline="") as stream:
        assert next(csv.DictReader(stream))["summary"] == 'a,b\n"quoted"'


def test_fifo_same_plan_no_backlog_and_no_duplicate_admission(store):
    first = register(store, 30)
    second = register(store, 20)
    periodic = commands.register(
        store, "task.json", cron="* * * * *", zone="UTC", now=NOW,
    )
    service = ScheduleService(store)
    with store.transaction() as records:
        service.admit(records, NOW + 60)
        service.admit(records, NOW + 60)
        assert [r["plan_id"] for r in records.queue()] == [
            second["id"], first["id"], periodic["id"],
        ]
        service.admit(records, NOW + 120)
        assert len(records.queue()) == 3
        skipped = [r for r in records.executions.values()
                   if r["status"] == "skipped"]
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "previous occurrence unfinished"


def test_modify_queued_and_cancel_are_durable(store):
    plan = register(store)
    service = ScheduleService(store)
    with store.transaction() as records:
        service.admit(records, NOW + 60)
    changed = commands.update(store, plan["id"], config="other.json",
                              now=NOW + 61)
    with store.transaction() as records:
        queue = records.queue()
        assert len(queue) == 1
        assert queue[0]["plan_revision"] == changed["revision"]
        assert queue[0]["scheduled_at"] == plan["next_at"]
        records.save_execution(dict(queue[0], status="starting"), NOW + 62)
    with pytest.raises(ValueError, match="starting or running"):
        commands.update(store, plan["id"], at=stamp(NOW + 500), now=NOW + 63)
    commands.cancel(store, plan["id"], NOW + 64)
    commands.cancel(store, plan["id"], NOW + 65)
    service.recover(NOW + 66)
    with store.transaction() as records:
        assert not records.plans[plan["id"]]["enabled"]
        assert not records.queue()
        assert any(r["status"] == "cancelled"
                   for r in records.executions.values())


def test_restart_preserves_queue_but_misses_unadmitted_occurrence(store):
    queued = register(store, 20)
    missed = register(store, 40)
    service = ScheduleService(store)
    with store.transaction() as records:
        service.admit(records, NOW + 20)
    service.recover(NOW + 100)
    service.recover(NOW + 101)
    with store.transaction() as records:
        assert [r["plan_id"] for r in records.queue()] == [queued["id"]]
        assert len(records.executions) == 2
        row = next(r for r in records.executions.values()
                   if r["plan_id"] == missed["id"])
        assert row["status"] == "missed"
        text = commands.render_list(records)
        assert "queued" in text and "missed" in text and "UTC" in text


def test_empty_directory_list(store):
    with store.transaction() as records:
        assert commands.render_list(records) == "暂无计划"


def test_time_modification_reorders_queue_and_retains_old_snapshot(store):
    first = register(store, 20)
    second = register(store, 30)
    service = ScheduleService(store)
    with store.transaction() as records:
        service.admit(records, NOW + 30)
    changed = commands.update(store, first["id"], at=stamp(NOW + 100),
                              now=NOW + 31)
    with store.transaction() as records:
        assert [r["plan_id"] for r in records.queue()] == [second["id"]]
        service.admit(records, NOW + 100)
        assert [r["plan_id"] for r in records.queue()] == [
            second["id"], first["id"],
        ]
        assert records.revisions[first["revision"]]["at"] == first["at"]
        assert records.plans[first["id"]]["at"] == changed["at"]


def test_corrupt_history_is_reported_not_silently_ignored(store):
    plan = register(store)
    with store.transaction() as records:
        records.new_execution(plan, plan["next_at"], "queued", NOW)
    (path,) = store.root.glob("*/executions.csv")
    with path.open("a") as stream:
        stream.write("truncated,row\n")
    with pytest.raises(ValueError, match="incomplete CSV"):
        with store.transaction():
            pass


def test_interrupted_parameter_update_does_not_lose_queued_task(
    store, monkeypatch,
):
    plan = register(store)
    service = ScheduleService(store)
    with store.transaction() as records:
        service.admit(records, NOW + 60)
        changed = records.get_plan(plan["id"])
        changed["source"] = "new-parameters.json"
        changed = records.save_plan(changed, NOW + 61)
        # Simulate a command exiting after JSON commit, before CSV update.
        assert records.queue()[0]["plan_revision"] != changed["revision"]
    monkeypatch.setattr(
        "luban_meter.core.scheduled_runs.subprocess.Popen",
        lambda *args, **kwargs: object(),
    )
    service.tick(NOW + 62)
    with store.transaction() as records:
        (row,) = records.executions.values()
        assert row["status"] == "starting"
        assert row["plan_revision"] == changed["revision"]


def test_interrupted_time_update_does_not_skip_new_due_time(store):
    plan = register(store)
    service = ScheduleService(store)
    with store.transaction() as records:
        service.admit(records, NOW + 60)
        changed = records.get_plan(plan["id"])
        changed.update(timing(stamp(NOW + 100), None, "UTC", NOW + 61))
        changed["calendar_id"] = "new-calendar"
        records.save_plan(changed, NOW + 61)
    with store.transaction() as records:
        service.admit(records, NOW + 100)
        new = [r for r in records.queue()
               if r["scheduled_at"] == stamp(NOW + 100)]
        assert len(new) == 1
