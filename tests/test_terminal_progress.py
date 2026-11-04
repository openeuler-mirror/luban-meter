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


def test_missing_events_and_broken_display_are_best_effort(tmp_path):
    progress, _, _ = start(tmp_path)
    progress("poll")
    assert not progress.disabled

    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("closed pipe")

    display = TerminalProgress(plan_fixture(), stream=Broken())
    display("task_started", task=display.plan["tasks"][0],
            directory=tmp_path)
    assert display.disabled
    coordinator = RunCoordinator()
    coordinator.progress = lambda *a, **k: 1 / 0
    coordinator._notify("poll")
    assert coordinator.progress is None


@pytest.mark.parametrize("disabled", [False, True])
def test_cli_progress_flag_is_not_a_benchmark_parameter(
    tmp_path, monkeypatch, capsys, disabled,
):
    plan = plan_fixture()
    plan["tasks"][0].update(scenario="online_serving")
    monkeypatch.setattr(cli, "build_plan", lambda *a, **k: plan)
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


def test_full_case_row_is_retained(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    progress, output, clock = start(tmp_path, Terminal())
    assert output.getvalue() == "当前任务：task\n"
    append_events(tmp_path, [
        event("case_ready", total_samples=20),
        event("request_finished", phase="warmup", status="success"),
        event("sample_finished", phase="warmup", state="complete"),
        event("request_finished", status="success", logical_requests=3),
        event("request_finished", status="failed"),
        event("sample_finished", state="complete", weight=3),
        event("sample_finished", state="failed"),
    ])
    progress("poll")
    row = progress.lines()[0]
    assert "task case1/2" in row
    assert "样本 4/20 请求 4/8 请求失败 1 样本失败 1" in row
    assert len(progress.lines()) == 1
    before = len(output.getvalue())
    progress.clear()
    assert output.getvalue()[before:] == "\r\x1b[2K"
    result = {"cases": [{
        "id": "case_1", "status": "partial_failed",
        "coverage": {"selected": 20, "complete": 3, "failed": 1,
                     "incomplete": 2, "not_started": 14},
    }]}
    (tmp_path / "result.json").write_text(json.dumps(result))
    clock[0] += 1
    progress("poll")
    text = output.getvalue()
    assert text.endswith("未完成 2 未开始 14 状态 partial_failed\n")
    assert "样本 4/20 请求 4/8" in text
    assert progress.lines() == ()
    progress._record(0, result)
    assert output.getvalue() == text
    append_events(tmp_path, [
        event("sample_finished", state="complete"),
        event("case_ready", case_id="case_2", total_samples=20),
        event("request_finished", case_id="case_2", status="success"),
    ])
    clock[0] += 1
    progress("poll")
    assert output.getvalue()[len(text):].startswith("task case2/2")
    assert "请求 1/未知" in progress.lines()[0]


@pytest.mark.parametrize("status", [
    "success", "failed", "partial_failed", "skipped",
    "interrupted", "discarded",
])
def test_final_rows_deduplicate_and_correct_status(tmp_path, status):
    progress, output, _ = start(tmp_path)
    result = {"status": status, "cases": [
        {"id": "case_1", "status": status},
        {"id": "case_2", "status": "skipped"},
    ]}
    for _ in range(2):
        progress("task_finished", task=progress.task, result=result)
    text = output.getvalue()
    assert text.count("task case1/2") == 1
    assert text.count("task case2/2") == 1
    assert f"状态 {status}" in text
    assert "样本 未知" in text
    result["cases"][0]["status"] = "discarded"
    progress("task_finished", task=progress.task, result=result)
    if status != "discarded":
        assert "更正：task case1/2" in output.getvalue()
    assert len(progress.results) == 2


def test_skipped_task_has_own_header(tmp_path):
    progress, output, _ = start(tmp_path)
    task = {"name": "next", "cases": [
        {"id": "case_1", "parameters": {}},
    ]}
    progress.plan["tasks"].append(task)
    progress("task_finished", task=task, result={
        "status": "skipped", "cases": [],
    })
    assert "当前任务：next\nnext case1/1" in output.getvalue()
    assert output.getvalue().endswith("状态 skipped\n")


def test_unknown_counts_and_safe_text(tmp_path):
    progress, output, _ = start(tmp_path)
    progress._event(event(
        "case_ready", case_id="case_2", total_samples=None,
    ))
    assert "样本 0/未知 请求 0/未知" in progress.lines()[0]
    progress.task["name"] = "task\x1b[2J\nattack"
    progress._record(0, {"cases": [
        {"id": "case_2", "status": "failed"},
    ]})
    assert "\x1b" not in output.getvalue()
    assert "task[2Jattack" in output.getvalue()
