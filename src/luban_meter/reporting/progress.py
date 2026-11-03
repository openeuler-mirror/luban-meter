"""Read existing run evidence for optional, best-effort terminal progress."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path


class EventTail:
    """Bound each read and retain incomplete UTF-8 JSONL records."""

    read_limit = 1024 * 1024
    line_limit = 8 * 1024 * 1024

    def __init__(self):
        self.identity = None
        self.offset = 0
        self.pending = b""
        self.dropping = False

    def read(self, path):
        try:
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                reset = self.identity != identity or stat.st_size < self.offset
                if reset:
                    self.identity = identity
                    self.offset = 0
                    self.pending = b""
                    self.dropping = False
                stream.seek(self.offset)
                chunk = stream.read(self.read_limit)
                self.offset = stream.tell()
        except OSError:
            return False, []
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        if self.dropping and lines:
            lines.pop(0)
            self.dropping = False
        events = []
        for line in lines:
            if len(line) > self.line_limit:
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        if len(self.pending) > self.line_limit:
            self.pending = b""
            self.dropping = True
        return reset, events


def _count(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _safe(text):
    return "".join(c for c in str(text) if c.isprintable())


def _fit(text, width):
    result = ""
    used = 0
    for char in _safe(text):
        size = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used + size > width:
            break
        result += char
        used += size
    return result


def _bar(done, total, width=12):
    if not total:
        return "[" + "-" * width + "]"
    filled = min(width, int(done / total * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _amount(done, total):
    return f"{done}/{total}" if total is not None else f"{done}/未知"


class TerminalProgress:
    """Parent-side renderer; no changes to collection or metric semantics."""

    refresh_interval = 0.25

    def __init__(self, plan, *, stream=None, clock=time.monotonic):
        self.plan = plan
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.tty = self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self.last_poll = float("-inf")
        self.last_key = None
        self.active = False
        self.active_lines = 0
        self.disabled = False
        self.task_index = None
        self.task = None
        self.directory = None
        self.tail = EventTail()
        self.live = {}
        self.current = None
        self.results = {}
        self.finished_tasks = set()
        self.printed_cases = {}
        self.case_results = {}
        self.titled_tasks = set()
        self.result_stamp = None
        self.phase = "等待任务"

    def __call__(self, kind, **facts):
        if self.disabled:
            return
        try:
            self._update(kind, **facts)
        except Exception:
            # Optional display must never affect execution or its exit code.
            self.disabled = True
            try:
                self.clear()
            except (OSError, UnicodeError):
                pass

    def _update(self, kind, **facts):
        if kind == "log":
            self.clear()
            return
        if kind == "task_started":
            self.clear()
            self.task = facts["task"]
            self.task_index = self.plan["tasks"].index(self.task)
            self.directory = Path(facts["directory"])
            self.tail = EventTail()
            self.live = {}
            self.current = None
            self.result_stamp = None
            self.phase = "准备任务"
            self._task_title(self.task_index)
        elif kind == "poll":
            if self.clock() - self.last_poll < self.refresh_interval:
                return
            self.last_poll = self.clock()
            self._poll()
        elif kind == "task_finished":
            task = facts["task"]
            if task is self.task:
                self._poll()
            else:
                self.task = task
                self.task_index = self.plan["tasks"].index(task)
                self.directory = None
                self.live = {}
                self.current = None
            index = self.plan["tasks"].index(task)
            self._record(index, facts["result"])
            present = {c["id"] for c in facts["result"].get("cases", [])}
            for case in task["cases"]:
                if case["id"] not in present:
                    self.results[index, case["id"]] = "skipped"
            self._print_completions()
            self.finished_tasks.add(index)
            self.phase = "任务结束：" + facts["result"]["status"]
        elif kind == "run_finished":
            self.phase = "运行结束：" + facts["result"]["status"]
        elif kind == "aborted":
            self.phase = "运行中断"
        self._render()
        if kind in ("run_finished", "aborted"):
            if self.active:
                self.stream.write("\n")
                self.stream.flush()
                self.active = False

    def _poll(self):
        if self.directory is None:
            return
        reset, events = self.tail.read(self.directory / "raw" / "events.jsonl")
        if reset:
            self.live = {}
            self.current = None
        for event in events:
            self._event(event)
        target = self.directory / "result.json"
        try:
            stat = target.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp != self.result_stamp:
                result = json.loads(target.read_text(encoding="utf-8"))
                self._record(self.task_index, result)
                self.result_stamp = stamp
        except (OSError, ValueError):
            pass
        if self.current and self.current not in {
            key[1] for key in self.results if key[0] == self.task_index
        }:
            # A committed collection precedes optional metric calculation.
            saved = self.directory / "raw" / f"{self.current}.result.json"
            if saved.exists():
                self.phase = "计算指标 / 收尾"

    def _event(self, event):
        case_id = event.get("case_id")
        planned = {case["id"]: case for case in self.task["cases"]}
        if case_id not in planned or (
            self.task_index, case_id
        ) in self.results:
            return
        if case_id not in self.live:
            self.live[case_id] = {
                "samples": Counter(), "requests": Counter(),
                "sample_total": None,
                "request_total": _count(
                    planned[case_id]["parameters"].get("num_requests")
                ),
            }
        state = self.live[case_id]
        self.current = case_id
        kind = event.get("kind", "")
        phase = event.get("phase")
        if kind == "case_started":
            self.phase = "准备 Case"
        if kind == "case_ready":
            state["sample_total"] = _count(event.get("total_samples"))
            self.phase = "准备完成"
        if phase == "warmup" and kind.startswith(("sample_", "request_")):
            self.phase = "预热（不计入正式进度）"
        if phase != "formal":
            return
        self.phase = "正式评测"
        if kind == "sample_finished":
            weight = _count(event.get("weight", 1))
            if weight is not None:
                state["samples"][event.get("state", "unknown")] += weight
        elif kind == "request_finished":
            cost = _count(event.get("logical_requests", 1))
            if cost is not None:
                state["requests"][event.get("status", "unknown")] += cost

    def _record(self, index, result):
        planned = {case["id"] for case in self.plan["tasks"][index]["cases"]}
        terminal = {
            "success", "failed", "partial_failed", "skipped",
            "interrupted", "discarded",
        }
        for case in result.get("cases", []):
            if case["id"] not in planned or case["status"] not in terminal:
                continue
            self.results[index, case["id"]] = case["status"]
            self.case_results[index, case["id"]] = case
            if index == self.task_index and case["id"] == self.current:
                state = self.live.get(self.current)
                coverage = case.get("coverage")
                if state is not None and coverage:
                    state["samples"] = Counter({
                        key: coverage[key] for key in
                        ("complete", "failed", "incomplete", "not_started")
                        if _count(coverage.get(key)) is not None
                    })
                    state["sample_total"] = _count(coverage.get("selected"))
                self.phase = "Case 结束：" + case["status"]

        self._print_completions()

    def _history(self, text):
        """Append a permanent line above the replaceable progress block."""
        self.clear()
        self.stream.write(_safe(text) + "\n")
        self.stream.flush()

    def _task_title(self, index):
        if index not in self.titled_tasks:
            self._history(f"当前任务：{self.plan['tasks'][index]['name']}")
            self.titled_tasks.add(index)

    def _case_line(self, index, case_id, status=None):
        task = self.plan["tasks"][index]
        ids = [case["id"] for case in task["cases"]]
        position = ids.index(case_id) + 1
        state = self.live.get(case_id, {}) if index == self.task_index else {}
        samples = state.get("samples", Counter())
        requests = state.get("requests", Counter())
        total = state.get("sample_total")
        request_total = state.get("request_total")
        saved = self.case_results.get((index, case_id), {})
        coverage = saved.get("coverage")
        if coverage:
            samples = Counter({
                key: value for key, value in coverage.items()
                if _count(value) is not None
            })
            total = _count(coverage.get("selected"))
        done = samples["complete"] + samples["failed"]
        known = bool(state or coverage)
        sample_amount = _amount(done, total) if known else "未知"
        request_amount = (
            _amount(sum(requests.values()), request_total) if state else "未知"
        )

        def sample_count(key):
            return samples[key] if known else "未知"

        line = (
            f"{task['name']} case{position}/{len(ids)}："
            f"{_bar(done, total)} 样本 {sample_amount} "
            f"请求 {request_amount} "
            f"请求失败 {requests['failed'] if state else '未知'} "
            f"样本失败 {sample_count('failed')} "
            f"未完成 {sample_count('incomplete')} "
            f"未开始 {sample_count('not_started')}"
        )
        if status is not None:
            return line + f" 状态 {status}"
        return line + f" · {self.phase}"

    def _print_completions(self):
        for key, status in self.results.items():
            if self.printed_cases.get(key) == status:
                continue
            index, case_id = key
            self._task_title(index)
            correction = "更正：" if key in self.printed_cases else ""
            self._history(correction + self._case_line(index, case_id, status))
            self.printed_cases[key] = status

    def lines(self):
        if self.task is None or self.current is None:
            return ()
        key = self.task_index, self.current
        if key in self.printed_cases:
            return ()
        return (self._case_line(*key),)

    def clear(self):
        if self.active:
            self.stream.write(
                "\r\x1b[2K"
                + "\x1b[1A\r\x1b[2K" * (self.active_lines - 1)
            )
            self.stream.flush()
            self.active = False

    def _render(self):
        lines = self.lines()
        if not lines:
            self.clear()
            return
        if self.tty:
            width = max(10, shutil.get_terminal_size((100, 24)).columns - 1)
            self.clear()
            self.stream.write("\n".join(_fit(line, width) for line in lines))
            self.active = True
            self.active_lines = len(lines)
        else:
            key = (self.task_index, self.current, self.phase,
                   tuple(sorted(self.results.items())),
                   len(self.finished_tasks))
            if key == self.last_key:
                return
            self.last_key = key
            self.stream.write("进度 | " + " | ".join(map(_safe, lines)) + "\n")
        self.stream.flush()
