"""Bounded terminal views of saved requests, without inference or scoring."""

from __future__ import annotations

import heapq
import json
import shlex
import shutil
from collections import Counter
from pathlib import Path

from .data import validate_result
from .request_details import finite, load_details
from .terminal import (
    MAX_REPORT_WIDTH,
    boxed,
    number,
    safe_text,
    select_cases,
    table_rows,
    wrapped,
)

DEFAULT_LIMIT = 10
BODY_LIMIT = 2000
TEXT_LIMIT = 500
LINK_LIMIT = 5
# Only explicit stored fields with known units; never derive latencies from
# scheduler offsets or substitute sample/batch timing for request timing.
SORT_FIELDS = {
    "e2el": "e2el_ms", "ttft": "ttft_ms", "itl": "itl_ms",
    "tpot": "tpot_ms", "latency": "latency_ms",
    "input_tokens": "input_tokens", "output_tokens": "output_tokens",
}
FAILURE_STATES = {
    "failed", "service_failed", "sandbox_failed", "errored", "error",
    "cancelled", "canceled", "incomplete", "not_started", "interrupted",
    "timeout", "timed_out", "partial_failed", "aborted",
}
SUCCESS_STATES = {"success", "complete", "completed"}


def mapping(value):
    return value if isinstance(value, dict) else {}


def preview(value, limit=TEXT_LIMIT):
    """Escape terminal controls and bound untrusted text, visibly."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = safe_text(value)
    if len(text) > limit:
        return text[:limit] + f" … [截断；原文 {len(text)} 字符，见 raw]"
    return text


def failed(record, status_key="status"):
    return str(record.get(status_key)) in FAILURE_STATES or bool(
        record.get("error")
    )


def sample_failed(sample):
    return failed(sample, "state") or failed(mapping(sample.get("facts")))


def visible_samples(entry):
    linked = zip(entry["row"]["sample_indices"], entry["samples"])
    return sorted(linked, key=lambda item: not sample_failed(item[1]))[
        :LINK_LIMIT
    ]


def make_entry(detail, row, task, case, sequence):
    index = row["request_index"]
    request = detail["requests"][index] if index is not None else {}
    response = mapping(request.get("response"))
    samples = [detail["samples"][i] for i in row["sample_indices"]]
    values = {
        name: response.get(field) if finite(response.get(field)) else None
        for name, field in SORT_FIELDS.items()
    }
    request_failed = index is not None and (
        failed(request) or failed(response)
    )
    return {
        "row": row, "request": request, "samples": samples,
        "task": task["name"], "case": case["id"],
        "source": detail["source"], "sequence": sequence, "values": values,
        "request_failed": request_failed,
        "error_related": request_failed or any(map(sample_failed, samples)),
        "unknown": str(row["status"]) not in (
            FAILURE_STATES | SUCCESS_STATES
        ),
    }


def protect_output(payload, result_path, root, output_path):
    """A text export must not overwrite input or any recorded Case file."""
    if output_path is None:
        return
    output = Path(output_path).resolve()
    paths = [Path(result_path)] if result_path is not None else []
    root = Path(root).resolve()
    for task in validate_result(payload):
        for case in task["cases"]:
            recorded = case.get("raw")
            if isinstance(recorded, str):
                original = Path(recorded)
                paths.append(original if original.is_absolute()
                             else root / original)
            base = (root / "tasks" / task["name"]
                    if payload["kind"] == "run" else root)
            paths.append(base / "raw" / (case["id"] + ".json"))
    for path in paths:
        if output == path.resolve() or (
            output.exists() and path.exists() and output.samefile(path)
        ):
            raise ValueError("output must not overwrite input/raw JSON")


def collect_entries(payload, selected, root, details, sort, limit):
    """Read one Case at a time; keep at most limit selected request rows."""
    counts = Counter()
    notices = []

    def entries():
        sequence = 0
        for task, case in selected:
            label = f"{task['name']}/{case['id']}"
            detail = load_details(
                case, task, root, run=payload["kind"] == "run",
                allow_partial=True,
            )
            notices.append(f"[{label}] Case status={case['status']}")
            coverage = mapping(case.get("coverage"))
            if any(coverage.get(k) for k in ("failed", "incomplete")):
                notices.append("  Case 样本统计: failed="
                               + number(coverage.get("failed"))
                               + " incomplete="
                               + number(coverage.get("incomplete")))
            for key in ("error", "reason", "metric_errors"):
                if case.get(key):
                    notices.append(f"  {key}: {preview(case[key])}")
            if detail["status"] != "available":
                counts["unavailable"] += 1
                notices.append("  详情不可用: " + detail["reason"])
                continue
            counts["available"] += 1
            counts["partial"] += bool(detail["warnings"])
            notices.extend("  " + value for value in detail["warnings"])
            if not detail["rows"]:
                notices.append("  无请求或样本记录；不能据此证明没有失败。")
            for row in detail["rows"]:
                entry = make_entry(detail, row, task, case, sequence)
                sequence += 1
                counts["requests" if row["request_index"] is not None
                       else "samples_only"] += 1
                counts["failed_requests"] += entry["request_failed"]
                counts["unknown"] += entry["unknown"]
                if details == "errors" and not entry["error_related"]:
                    continue
                counts["matched"] += 1
                if sort and entry["values"][sort] is None:
                    counts["missing_sort"] += 1
                yield entry

    def order(entry):
        value = entry["values"][sort] if sort else None
        return (value is None, -value if value is not None else 0,
                entry["sequence"])

    shown = heapq.nsmallest(limit, entries(), key=order)
    return shown, counts, notices


def request_table(entries, width):
    fields = ("input_tokens", "output_tokens", "ttft", "itl", "tpot", "e2el")
    headers = ("Request ID", "Status", "In", "Out", "TTFT", "ITL", "TPOT",
               "E2EL")
    yield "Request ID=请求标识；Status=状态。"
    yield "In/Out=Token 数；TTFT/E2EL=ms；ITL/TPOT=ms/token。"
    if width >= 80:
        extra = width - 80
        sizes = [14 + extra * 2 // 5, 7 + extra // 10,
                 4 + extra // 20, 4 + extra // 20]
        sizes.extend([6 + extra // 10] * 3)
        sizes.append(width - 25 - sum(sizes))
        border = "+" + "+".join("-" * (s + 2) for s in sizes) + "+"
        yield border
        yield from table_rows(headers, sizes)
        yield border
        previous = None
        for entry in entries:
            label = (entry["task"], entry["case"])
            if label != previous:
                yield from table_rows(
                    [preview(f"[{label[0]}/{label[1]}]", 240)], [width - 4]
                )
                yield border
                previous = label
            cells = [preview(entry["row"]["request_id"], 120),
                     preview(entry["row"]["status"], 60)]
            cells.extend(number(entry["values"][field]) for field in fields)
            yield from table_rows(cells, sizes)
        yield border
    else:
        for index, entry in enumerate(entries, 1):
            yield from boxed([
                f"#{index} [{entry['task']}/{entry['case']}]",
                "Request ID=" + preview(entry["row"]["request_id"], 120),
                "Status=" + preview(entry["row"]["status"], 60),
                "  ".join(f"{field}={number(entry['values'][field])}"
                          for field in fields),
            ], width)


def error_lines(entry, seen):
    """Share message references, retaining each row's source/type fields."""
    request = entry["request"]
    sources = [("request", request),
               ("response", mapping(request.get("response")))]
    for index, sample in visible_samples(entry):
        sources.extend(((f"sample[{index}]", sample),
                        (f"sample[{index}].facts",
                         mapping(sample.get("facts")))))
    groups = {}
    for label, record in sources:
        error, error_type = record.get("error"), record.get("error_type")
        if not error and not error_type:
            continue
        if isinstance(error, dict):
            error_type = error.get("type", error_type)
            message = error.get("message", error)
        else:
            message = error
        # Compare complete saved messages, not their truncated previews.
        key = json.dumps(message, ensure_ascii=False, sort_keys=True)
        group = groups.setdefault(key, {"message": message, "sources": []})
        field = ".error" if error else ".error_type"
        group["sources"].append(
            f"{label}{field}: error_type={preview(error_type)}"
        )
    for key, group in groups.items():
        if key in seen:
            yield f"  Error/错误: [{seen[key]}] 同前，消息已展示。"
        else:
            seen[key] = f"E{len(seen) + 1}"
            yield f"  Error/错误: [{seen[key]}] " + preview(group["message"])
        for source in group["sources"]:
            yield "  来源/Source: " + source


def body_lines(entry):
    yield "  Body/正文（仅预览，最多每字段 2000 字符）:"
    request = entry["request"]
    sources = [("request", request),
               ("response", mapping(request.get("response")))]
    for index, sample in visible_samples(entry):
        sources.append((f"sample[{index}].facts",
                        mapping(sample.get("facts"))))
        sources.append((f"sample[{index}].source（数据集原文，非请求确认）",
                        mapping(sample.get("source"))))
    found = False
    for label, record in sources:
        for key in ("prompt", "messages", "conversations", "text",
                    "output_text", "raw_output"):
            if key in record and record[key] is not None:
                found = True
                text = ("[已保存空正文]" if record[key] == "" else
                        preview(record[key], BODY_LIMIT))
                yield f"  {label}.{key}: {text}"
    if not found:
        yield "  未保存可识别的 Prompt/回答正文；不会重新执行请求。"


def entry_lines(entry, index, verbose, include_body, details="requests",
                seen_errors=None):
    row, request = entry["row"], entry["request"]
    response = mapping(request.get("response"))
    kind = "request" if row["request_index"] is not None else "sample-only"
    expanded = verbose or details == "errors" or entry["error_related"]
    diagnostics = []
    if kind == "sample-only":
        diagnostics.append(
            "  仅有样本记录，未保存关联请求；表中状态为样本状态。"
        )
    if entry["unknown"]:
        diagnostics.append("  状态未知，不能判定为成功或失败。")
    if any(value is not None and value < 0
           for value in entry["values"].values()):
        diagnostics.append(
            "  注意：保存的指标含负值；原值保留，请核查采集数据。"
        )
    if request.get("logical_requests", 1) != 1:
        diagnostics.extend((
            "  logical_requests=" + preview(request.get("logical_requests")),
            "  这是批次/逻辑请求记录，不能拆为单请求延迟。",
        ))
    if response.get("connection_cleanup_error"):
        diagnostics.append("  response.connection_cleanup_error="
                           + preview(response["connection_cleanup_error"]))
    if not expanded:
        if diagnostics or include_body:
            yield f"#{index} [{entry['task']}/{entry['case']}] {kind}"
            yield from diagnostics
            if include_body:
                yield from body_lines(entry)
        return
    yield (f"#{index} [{entry['task']}/{entry['case']}] {kind} "
           f"id={preview(row['request_id'], 120)} "
           f"sample_id={preview(row['sample_id'], 120)}")
    if kind == "request":
        yield f"  raw_index={row['request_index']}"
    yield from diagnostics
    if response.get("status") is not None:
        yield "  response.status=" + preview(response["status"], 60)
    for field in ("latency_ms", "duration_ms"):
        if field in response:
            yield f"  response.{field}={number(response[field])} ms（原字段）"
    yield from error_lines(entry, {} if seen_errors is None else seen_errors)
    if response.get("finish_reason"):
        yield "  response.finish_reason=" + preview(response["finish_reason"])
    if entry["request_failed"] and not (
        request.get("error") or response.get("error")
    ):
        yield "  请求失败，但未保存错误描述；错误类型 N/A。"
    for _, sample in visible_samples(entry):
        facts = mapping(sample.get("facts"))
        yield (f"  Sample/样本 {preview(sample.get('sample_id'), 120)}: "
               f"state={preview(sample.get('state'), 60)} "
               f"facts.status={preview(facts.get('status'), 60)}")
        saved = [f"{key}={number(facts[key])}" for key in
                 (*SORT_FIELDS.values(), "duration_ms") if key in facts]
        if saved:
            yield "  样本事实（不混入请求指标）: " + "  ".join(saved)
        if sample_failed(sample) and not (sample.get("error")
                                          or facts.get("error")):
            yield "  样本未成功完成；未保存错误描述。"
    if len(entry["samples"]) > LINK_LIMIT:
        yield (f"  另有 {len(entry['samples']) - LINK_LIMIT} 条关联样本，"
               "详情见 raw；不按完成顺序猜测关联。")
    if verbose:
        for key in ("scheduled_offset_s", "admission_offset_s",
                    "start_offset_s", "end_offset_s", "submission_offset_s",
                    "submission_delay_s", "scheduling_delay_s"):
            if key in request:
                yield f"  request.{key}={preview(request[key])}（原始秒值）"
    if include_body:
        yield from body_lines(entry)


def render_details(payload, *, raw_root, details="requests", tasks=(),
                   cases=(), sort=None, limit=DEFAULT_LIMIT, verbose=False,
                   include_body=False, width=None, result_path=None,
                   output_path=None):
    """Display saved evidence; missing data never implies a successful run."""
    if details not in ("errors", "requests"):
        raise ValueError("details must be errors or requests")
    if sort is not None and sort not in SORT_FIELDS:
        raise ValueError("invalid sort; choose: " + ", ".join(SORT_FIELDS))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("--limit must be a positive integer (>= 1)")
    all_tasks = validate_result(payload)
    selected_tasks, selected = select_cases(all_tasks, tasks, cases)
    protect_output(payload, result_path, raw_root, output_path)
    entries, counts, notices = collect_entries(
        payload, selected, raw_root, details, sort, limit,
    )
    width = max(20, min(MAX_REPORT_WIDTH, width or
                        shutil.get_terminal_size((100, 24)).columns))
    lines = list(boxed([
        "LuBan-Meter 请求详情 / " + details,
        f"运行: {preview(payload.get('id', payload.get('name')), 120)}  "
        f"status={payload['status']}",
        f"Case: 选择 {len(selected)}，可读 {counts['available']}，"
        f"不可用 {counts['unavailable']}，结构不完整 {counts['partial']}",
        f"已保存请求 {counts['requests']}，失败/未完成请求 "
        f"{counts['failed_requests']}，仅样本记录 {counts['samples_only']}",
        f"匹配记录 {counts['matched']}，展示 {len(entries)}，"
        f"省略 {counts['matched'] - len(entries)}（--limit {limit}）",
    ], width))
    if payload.get("errors"):
        lines.append("运行诊断: " + preview(payload["errors"]))
    for task in selected_tasks:
        if task.get("errors"):
            lines.append(f"[{task['name']}] 任务诊断: "
                         + preview(task["errors"]))
    lines.extend(notices)
    if include_body:
        lines.append(
            "注意：正文可能包含敏感信息；仅显式 --include-body 输出。"
        )
    if sort:
        lines.append(f"按 {sort} 降序；同值保持原始记录顺序。")
        lines.append(f"无有效排序值 {counts['missing_sort']} 条，保留在末尾；"
                     "也受 --limit 限制。耗时较慢不等于性能异常。")
    else:
        lines.append(
            "顺序：Task/Case 顺序及原始文件记录顺序，不等同发起顺序。"
        )
    if entries:
        lines.extend(request_table(entries, width))
        seen_errors = {}
        for index, entry in enumerate(entries, 1):
            lines.extend(entry_lines(
                entry, index, verbose, include_body, details, seen_errors,
            ))
    elif details == "errors":
        lines.append(
            "已读取记录中没有错误/未完成记录。"
            if counts["requests"] or counts["samples_only"] else
            "没有可供错误检查的请求或样本记录。"
        )
        if (
            counts["unavailable"] or counts["partial"]
            or not counts["requests"]
        ):
            lines.append("原始数据不完整或无请求记录，不能确认全部请求无错误。")
    else:
        lines.append("没有可展示的请求或样本记录。")
    if counts["unknown"]:
        lines.append(f"有 {counts['unknown']} 条记录状态未知；"
                     "不视为成功，用 --details requests 查看。")
    lines.extend((
        "N/A=原字段未保存/不适用/非有效数值；0 保留。未重算任何指标。",
        "ITL/TPOT 仅读取 response.itl_ms/tpot_ms；不从 Case 均值反填。",
        "质量样本 facts 使用 --verbose 查看；调度时间不代替响应耗时。",
    ))
    rendered = "\n".join(part for line in lines
                         for part in wrapped(safe_text(line), width))
    # Keep evidence paths and commands copyable, as in the summary view.
    paths = list(dict.fromkeys(entry["source"] for entry in entries))
    if paths:
        rendered += "\n" + "\n".join("raw: " + safe_text(p) for p in paths)
    if result_path is not None:
        command = "luban-meter report --input " + shlex.quote(
            str(Path(result_path).resolve())
        ) + " --format text"
        for key, selectors in (("--task", tasks), ("--case", cases)):
            for selector in selectors:
                command += " " + key + " " + shlex.quote(selector)
        rendered += "\n结果/JSON: " + safe_text(Path(result_path).resolve())
        rendered += "\n汇总/Summary: " + safe_text(command + " --verbose")
    return rendered
