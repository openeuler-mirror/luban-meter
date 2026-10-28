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


def _elapsed(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


class TerminalProgress:
    """Parent-side renderer; no changes to collection or metric semantics."""

    refresh_interval = 0.25

    def __init__(self, plan, *, stream=None, clock=time.monotonic):
        self.plan = plan
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.tty = self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self.started = clock()
        self.last_poll = float("-inf")
        self.last_key = None
        self.active = False
        self.disabled = False
        self.task_index = None
        self.task = None
        self.directory = None
        self.tail = EventTail()
        self.live = {}
        self.current = None
        self.results = {}
        self.finished_tasks = set()
        self.result_stamp = None
        self.phase = "等待任务"
        self.total = sum(len(task["cases"]) for task in plan["tasks"])

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
        elif kind == "poll":
            if self.clock() - self.last_poll < self.refresh_interval:
                return
            self.last_poll = self.clock()
            self._poll()
        elif kind == "task_finished":
            task = facts["task"]
            if task is self.task:
                self._poll()
            index = self.plan["tasks"].index(task)
            self._record(index, facts["result"])
            present = {c["id"] for c in facts["result"].get("cases", [])}
            for case in task["cases"]:
                if case["id"] not in present:
                    self.results[index, case["id"]] = "skipped"
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
        for case in result.get("cases", []):
            self.results[index, case["id"]] = case["status"]
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

    def lines(self):
        counts = Counter(self.results.values())
        failed = sum(counts[s] for s in ("failed", "partial_failed"))
        skipped = counts["skipped"]
        interrupted = counts["interrupted"] + counts["discarded"]
        overall = (
            f"Case {_bar(len(self.results), self.total)} "
            f"{len(self.results)}/{self.total} 已结束 "
            f"任务 {len(self.finished_tasks)}/{len(self.plan['tasks'])} "
            f"成功 {counts['success']} 失败 {failed} "
            f"跳过 {skipped} 中断/丢弃 {interrupted} "
            f"已用 {_elapsed(self.clock() - self.started)}"
        )
        if self.task is None:
            return overall, self.phase
        prefix = f"{self.task['name']}/{self.current or '-'}"
        state = self.live.get(self.current)
        if state is None or self.phase.startswith(("任务结束", "运行")):
            return overall, f"{prefix} · {self.phase}"
        samples, requests = state["samples"], state["requests"]
        sample_done = samples["complete"] + samples["failed"]
        request_done = sum(requests.values())
        sample_total, request_total = (
            state["sample_total"], state["request_total"]
        )
        bar = (
            _bar(sample_done, sample_total) if sample_total is not None
            else _bar(request_done, request_total)
        )
        current = (
            f"{prefix} · {self.phase} {bar} "
            f"样本 {_amount(sample_done, sample_total)} "
            f"请求 {_amount(request_done, request_total)} "
            f"请求失败 {requests['failed']} 样本失败 {samples['failed']} "
            f"未完成 {samples['incomplete']} 未开始 {samples['not_started']}"
        )
        return overall, current

    def clear(self):
        if self.active:
            self.stream.write("\r\x1b[2K\x1b[1A\r\x1b[2K")
            self.stream.flush()
            self.active = False

    def _render(self):
        lines = self.lines()
        if self.tty:
            width = max(10, shutil.get_terminal_size((100, 24)).columns - 1)
            self.clear()
            self.stream.write("\n".join(_fit(line, width) for line in lines))
            self.active = True
        else:
            key = (self.task_index, self.current, self.phase,
                   len(self.results), len(self.finished_tasks))
            if key == self.last_key:
                return
            self.last_key = key
            self.stream.write("进度 | " + " | ".join(map(_safe, lines)) + "\n")
        self.stream.flush()
