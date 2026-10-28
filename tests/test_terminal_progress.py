"""Progress reads saved facts without changing execution or scoring."""

import io
import json
from pathlib import Path

import pytest

from luban_meter import cli
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.reporting.progress import EventTail, TerminalProgress


class Terminal(io.StringIO):
    def isatty(self):
        return True


def plan_fixture():
    return {"tasks": [{
        "name": "task", "cases": [
            {"id": "case_1", "parameters": {"num_requests": 8}},
            {"id": "case_2", "parameters": {"num_requests": None}},
        ],
    }], "case_count": 2}


def append_events(directory, events):
    raw = directory / "raw"
    raw.mkdir(exist_ok=True)
    with (raw / "events.jsonl").open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def event(kind, phase="formal", case_id="case_1", **fields):
    return {"kind": kind, "phase": phase, "case_id": case_id, **fields}


def start(tmp_path, stream=None):
    plan = plan_fixture()
    clock = [0.0]
    output = stream if stream is not None else io.StringIO()
    progress = TerminalProgress(plan, stream=output, clock=lambda: clock[0])
    progress("run_started")
    progress("task_started", task=plan["tasks"][0], directory=tmp_path)
    return progress, output, clock


def test_partial_unicode_jsonl_and_replacement(tmp_path):
    target = tmp_path / "events.jsonl"
    data = json.dumps({"kind": "数据"}, ensure_ascii=False).encode()
    target.write_bytes(data[:-2])
    tail = EventTail()
    assert tail.read(target) == (True, [])
    with target.open("ab") as stream:
        stream.write(data[-2:] + b"\ninvalid\n")
    assert tail.read(target) == (False, [{"kind": "数据"}])
    replacement = tmp_path / "replacement"
    replacement.write_text('{"kind": "new"}\n')
    replacement.replace(target)
    assert tail.read(target) == (True, [{"kind": "new"}])


def test_tail_bounds_reads_and_skips_oversized_lines(tmp_path):
    target = tmp_path / "events.jsonl"
    target.write_bytes(b"x" * 100 + b'\n{"ok":1}\n')
    tail = EventTail()
    tail.read_limit = 16
    tail.line_limit = 32
    found = []
    for _ in range(10):
        _, events = tail.read(target)
        found.extend(events)
        assert len(tail.pending) <= tail.line_limit
    assert found == [{"ok": 1}]


def test_warmup_weighted_samples_and_requests_are_distinct(tmp_path):
    progress, output, clock = start(tmp_path)
    append_events(tmp_path, [
        event("case_ready", total_samples=10),
        event("request_finished", phase="warmup", status="success"),
        event("sample_finished", phase="warmup", state="complete"),
        event("request_finished", status="success", logical_requests=3),
        event("request_finished", status="failed", logical_requests=2),
        event("sample_finished", state="complete", weight=2),
        event("sample_finished", state="incomplete", weight=1),
    ])
    progress("poll")
    assert "样本 2/10" in progress.lines()[1]
    assert "请求 5/8" in progress.lines()[1]
    assert "请求失败 2" in progress.lines()[1]
    assert "未完成 1" in progress.lines()[1]
    assert "\x1b" not in output.getvalue()
    assert "\r" not in output.getvalue()
    before = output.getvalue()
    append_events(tmp_path, [event("sample_finished", state="failed")])
    clock[0] += 1
    progress("poll")
    assert "样本 3/10" in progress.lines()[1]
    assert output.getvalue() == before  # Redirected logs do not spam requests.


def test_unknown_total_and_warmup_phase(tmp_path):
    progress, _, clock = start(tmp_path)
    append_events(tmp_path, [
        event("case_ready", case_id="case_2", total_samples=None),
        event("request_started", case_id="case_2", phase="warmup"),
    ])
    progress("poll")
    assert "预热" in progress.lines()[1]
    assert "0/未知" in progress.lines()[1]
    append_events(tmp_path, [event(
        "request_finished", case_id="case_2", status="success"
    )])
    clock[0] += 1
    progress("poll")
    assert "请求 1/未知" in progress.lines()[1]
    assert "%" not in progress.lines()[1]


def test_result_commit_wins_over_delayed_events(tmp_path):
    progress, _, clock = start(tmp_path)
    append_events(tmp_path, [event("case_ready", total_samples=3)])
    result = {"cases": [{
        "id": "case_1", "status": "partial_failed", "coverage": {
            "selected": 3, "complete": 2, "failed": 1,
        },
    }]}
    (tmp_path / "result.json").write_text(json.dumps(result))
    progress("poll")
    assert "1/2 已结束" in progress.lines()[0]
    assert "失败 1" in progress.lines()[0]
    append_events(tmp_path, [event("sample_finished", state="complete")])
    clock[0] += 1
    progress("poll")
    assert "样本 3/3" in progress.lines()[1]


def test_skips_discard_and_interruption_do_not_claim_success(tmp_path):
    progress, output, _ = start(tmp_path)
    task = progress.plan["tasks"][0]
    result = {"status": "interrupted", "cases": [
        {"id": "case_1", "status": "discarded"},
        {"id": "case_2", "status": "skipped"},
    ]}
    progress("task_finished", task=task, result=result)
    progress("run_finished", result={"status": "interrupted"})
    text = output.getvalue()
    assert "2/2 已结束" in text
    assert "成功 0" in text
    assert "跳过 1 中断/丢弃 1" in text
    assert "运行结束：interrupted" in text


def test_tty_clears_before_logs_and_finishes_on_newline(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    progress, output, _ = start(tmp_path, Terminal())
    assert progress.active
    progress("log")
    assert not progress.active
    assert output.getvalue().endswith("\r\x1b[2K\x1b[1A\r\x1b[2K")
    progress("aborted")
    assert not progress.active
    assert output.getvalue().endswith("\n")
    assert "运行中断" in output.getvalue()


def test_missing_events_and_broken_display_are_best_effort(tmp_path):
    progress, _, _ = start(tmp_path)
    progress("poll")
    assert not progress.disabled

    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("closed pipe")

    display = TerminalProgress(plan_fixture(), stream=Broken())
    display("run_started")
    assert display.disabled
    coordinator = RunCoordinator()
    coordinator.progress = lambda *a, **k: 1 / 0
    coordinator._notify("poll")
    assert coordinator.progress is None


def test_control_characters_do_not_inject_terminal_commands(tmp_path):
    progress, output, _ = start(tmp_path)
    progress.task["name"] = "task\x1b[2J\nattack"
    progress("aborted")
    assert "\x1b" not in output.getvalue()
    assert "task[2Jattack" in output.getvalue()


@pytest.mark.parametrize("disabled", [False, True])
def test_cli_progress_flag_is_not_a_benchmark_parameter(
    tmp_path, monkeypatch, capsys, disabled,
):
    plan = plan_fixture()
    plan["tasks"][0].update(scenario="online_serving")
    monkeypatch.setattr(cli, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(cli, "case_summary", lambda *a, **k: {})
    received = []

    def run(self, plan, output, **kwargs):
        received.append(kwargs["progress"])
        return {"kind": "run", "status": "success", "tasks": []}, tmp_path

    monkeypatch.setattr(cli.RunCoordinator, "run", run)
    args = ["run", "--no-summary"]
    if disabled:
        args.append("--no-progress")
    assert cli.main(args) == 0
    assert (received[0] is None) is disabled
    assert "no_progress" not in plan["tasks"][0]["cases"][0]["parameters"]
    assert Path(tmp_path / "report.html").exists()
