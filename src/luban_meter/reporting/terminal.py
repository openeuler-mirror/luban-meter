"""Plain terminal summaries of recorded metrics, without model access."""

from __future__ import annotations

import json
import shlex
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from luban_meter.core.token_totals import saved_throughput_metric

from .best_summary import best_summary_lines
from .data import format_number as number
from .data import metric_groups, validate_result

CASE_LIMIT = 5
MAX_REPORT_WIDTH = 120
SUMMARY_PATHS = (
    "request_view/ttft", "request_view/itl", "request_view/tpot",
    "request_view/e2el", "request_view/input_tokens",
    "request_view/output_tokens", "service_view/request_throughput",
    "service_view/output_token_throughput",
)


def safe_text(value):
    if value is None or value == "":
        return "N/A"
    text = str(value)
    return "".join(
        f"\\u{ord(char):04x}"
        if unicodedata.category(char).startswith("C") else char
        for char in text
    )


def display_width(text):
    return sum(
        0 if unicodedata.combining(char) else
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        for char in text
    )


def wrapped(text, width):
    """Wrap even unbroken paths and wide characters without ANSI codes."""
    line, size = "", 0
    for char in text:
        length = display_width(char)
        if size + length > width and line:
            split = line.rfind(" ")
            if split > 0:
                yield line[:split]
                line = line[split + 1:]
                size = display_width(line)
            else:
                yield line
                line, size = "", 0
        line += char
        size += length
    yield line


def select_cases(tasks, task_names, case_names):
    selected_tasks = [t for t in tasks if not task_names or
                      t["name"] in task_names]
    unknown = set(task_names) - {t["name"] for t in tasks}
    if unknown:
        raise ValueError(f"unknown task: {', '.join(sorted(unknown))}")
    rows = [(t, c) for t in selected_tasks for c in t["cases"]]
    if not case_names:
        return selected_tasks, rows
    matched = set()
    selected = []
    for task, case in rows:
        names = {case["id"], f"{task['name']}/{case['id']}"}
        matches = set(case_names) & names
        if matches:
            matched.update(matches)
            selected.append((task, case))
    unknown = set(case_names) - matched
    if unknown:
        raise ValueError(
            "unknown case within selected tasks: " + ", ".join(sorted(unknown))
        )
    matched_tasks = {id(task) for task, _ in selected}
    return [t for t in selected_tasks if id(t) in matched_tasks], selected


def metric_entries(metric):
    """Discover saved statistics without a scenario or metric allowlist."""
    return list(metric_groups(metric.get("values", metric)))


def is_breakdown(path):
    """Recognize explicit breakdown groups, not arbitrary nested metrics."""
    return any(
        part in {"by_subject", "by_category", "by_subcategory"}
        or part.endswith(("_by_subject", "_by_category", "_by_subcategory"))
        for part in path.split("/")[:-1]
    )


def summary_entries(entries):
    """Prioritize familiar performance fields, without losing extensions."""
    breakdowns = [entry for entry in entries if is_breakdown(entry[0])]
    entries = [entry for entry in entries if not is_breakdown(entry[0])]
    preferred = [entry for suffix in SUMMARY_PATHS for entry in entries
                 if entry[0] == suffix or entry[0].endswith("/" + suffix)]
    if not preferred:
        return entries, breakdowns
    paths = {path for path, _ in preferred}
    other = [(path, item) for path, item in entries if path not in paths]
    extra = [entry for entry in other
             if not any(view in entry[0].split("/")
                        for view in ("request_view", "service_view"))]
    hidden = [entry for entry in other if entry not in extra]
    return preferred + extra, hidden + breakdowns


def distribution_values(item):
    percentiles = item.get("percentiles")
    percentiles = percentiles if isinstance(percentiles, dict) else {}
    count = item.get("sample_count", item.get("count"))
    empty = isinstance(count, (int, float)) and not isinstance(
        count, bool
    ) and count == 0
    return (
        number(None if empty else item.get("value", item.get("mean"))),
        number(None if empty else item.get(
            "p50", item.get("median", percentiles.get("p50"))
        )),
        number(None if empty else item.get("p95", percentiles.get("p95"))),
        safe_text(item.get("unit")),
        number(count),
    )


def boxed(lines, width):
    """Frame text using terminal cell widths, including CJK characters."""
    border = "+" + "-" * (width - 2) + "+"
    yield border
    for line in lines:
        for part in wrapped(line, width - 4):
            yield "| " + part + " " * (width - 4 - display_width(part)) + " |"
    yield border


def table_rows(cells, sizes):
    columns = [list(wrapped(cell, size))
               for cell, size in zip(cells, sizes)]
    for index in range(max(map(len, columns))):
        parts = []
        for column, size in zip(columns, sizes):
            value = column[index] if index < len(column) else ""
            parts.append(value + " " * (size - display_width(value)))
        yield "| " + " | ".join(parts) + " |"


def metric_lines(entries, width, verbose=False):
    context = []
    if not verbose:
        # Counts/durations are context, not distributions. Keep all saved
        # fields but avoid repeating N/A percentile columns for each scalar.
        context = [(path, item) for path, item in entries
                   if "value" in item
                   and item.get("unit") in ("sample", "request", "token", "s")]
        entries = [(path, item) for path, item in entries
                   if path not in {p for p, _ in context}]
    scalar_only = all("value" in item and not any(
        key in item for key in (
            "mean", "median", "p50", "p95", "percentiles"
        )
    ) for _, item in entries)
    value_sizes = [12, 9, 6] if scalar_only else [10, 7, 7, 9, 6]
    for _, item in entries:
        values = distribution_values(item)
        if scalar_only:
            values = (values[0], values[3], values[4])
        value_sizes = [
            max(size, display_width(value))
            for size, value in zip(value_sizes, values)
        ]
    # Reserve complete value cells first; let metric labels wrap.
    label_size = width - sum(value_sizes) - 3 * (len(value_sizes) + 1) - 1
    sizes = (label_size, *value_sizes)
    if width >= 76 and label_size >= 16:
        border = "+" + "+".join("-" * (s + 2) for s in sizes) + "+"
        labels = {
            "ttft": "TTFT 首Token延迟", "e2el": "E2EL 总耗时",
            "itl": "ITL Token间隔", "tpot": "TPOT 每Token耗时",
            "input_tokens": "Input Tokens 输入Token分布",
            "output_tokens": "Output Tokens 输出Token分布",
            "request_throughput": "Request Throughput 请求吞吐量",
            "output_token_throughput": (
                "Output Token Throughput 输出Token吞吐量"
            ),
            "total_input_tokens": "Total Input Tokens 输入Token总数",
            "total_output_tokens": "Total Output Tokens 输出Token总数",
            "queue_time_ms": "Queue Time 服务端排队等待分布",
            "queue_time_waited_mean": (
                "Queue Time Mean 平均排队等待（等待过的请求）"
            ),
            "queue_time_reported_requests": (
                "Queue Timing Reported 上报计时请求数"
            ),
            "queue_time_missing_requests": (
                "Queue Timing Missing 缺失计时请求数"
            ),
            "accuracy": "Accuracy 准确率",
            "exact_match": "Exact Match 完全匹配",
            "token_f1": "Token F1", "pass_at_1": "pass@1",
        }
        previous_view = None
        for path, item in entries:
            view, _, name = path.rpartition("/")
            if view != previous_view:
                if previous_view is not None:
                    yield border
                view_label = {
                    "request_view": "request_view / 请求统计",
                    "service_view": "service_view / 服务统计",
                    "": "metrics / 指标",
                }.get(view, view)
                yield from boxed([safe_text(view_label)], width)
                headers = (("Metric", "Value", "Unit", "N") if scalar_only
                           else ("Metric", "Mean/Value", "P50", "P95",
                                 "Unit", "N"))
                chinese = (("指标", "数值", "单位", "样本数") if scalar_only
                           else ("指标", "均值/标量", "中位数", "95分位",
                                 "单位", "样本数"))
                yield from table_rows(headers, sizes)
                yield from table_rows(chinese, sizes)
                yield border
                previous_view = view
            label = labels.get(name, name)
            if view == "request_view" and name.endswith("throughput"):
                label = "Per-request " + label
            values = distribution_values(item)
            if scalar_only:
                values = (values[0], values[3], values[4])
            yield from table_rows((safe_text(label), *values), sizes)
        if entries:
            yield border
    else:
        # Narrow panes or wide values use a readable stacked box.
        lines = []
        for path, item in entries:
            value, p50, p95, unit, count = distribution_values(item)
            lines.extend((safe_text(path), f"[{unit}] N={count}"))
            if "value" in item:
                lines.append(f"value={value}")
            else:
                lines.extend((
                    f"mean/value={value}", f"P50={p50}", f"P95={p95}",
                ))
        yield from boxed(lines, width)
    if context:
        by_view = {}
        for path, item in context:
            view, _, name = path.rpartition("/")
            value = f"{safe_text(name)}={distribution_values(item)[0]} "
            value += safe_text(item.get("unit"))
            if item.get("count") is not None:
                value += f" N={number(item['count'])}"
            by_view.setdefault(view, []).append(value)
        for view, values in by_view.items():
            yield "计数/窗口 " + safe_text(view) + ": " + "  ".join(values)
    for path, item in [*entries, *context]:
        notes = []
        keys = ("condition", "reason")
        if verbose:
            keys += ("aggregation", "percentile_method", "weight_sum")
        for key in keys:
            if item.get(key) is not None:
                notes.append(f"{key}={safe_text(item[key])}")
        if item.get("sample_count", item.get("count")) == 0:
            notes.append("无统计样本，显示 N/A")
        if notes:
            yield safe_text(path) + ": " + "; ".join(notes)


def context_lines(case, verbose=False):
    parameters = case.get("parameters") or {}
    labels = {
        "model": "模型", "data": "数据", "profile": "策略",
        "max_concurrency": "并发", "num_warmups": "预热",
        "input_tokens": "输入长度", "output_tokens": "输出长度",
    }
    fields_to_show = [("model", "data", "profile"),
                      ("max_concurrency", "num_warmups")]
    if verbose:
        fields_to_show.append(("input_tokens", "output_tokens"))
    for fields in fields_to_show:
        yield "  ".join(
            f"{labels[k]}/{k}={safe_text(parameters.get(k))}" for k in fields
        )
    if parameters.get("dataset_path"):
        yield "dataset_path=" + safe_text(parameters["dataset_path"])
    coverage = case.get("coverage") or {}
    yield "样本/Samples: " + "  ".join(
        f"{key}={number(coverage.get(key))}"
        for key in ("selected", "complete", "failed", "incomplete",
                    "not_started")
    )
    window = case.get("window") or {}
    yield (
        f"Case 窗口/Window: duration_s={number(window.get('duration_s'))} s  "
        f"capacity={number(window.get('request_capacity'))}  "
        f"peak={number(window.get('peak_requests'))}"
    )
    if verbose:
        yield (
            f"Unix 秒: started_at={safe_text(window.get('started_at'))}  "
            f"ended_at={safe_text(window.get('ended_at'))}"
        )


def saved_token_view(result):
    metric = saved_throughput_metric(result)
    if not isinstance(metric, dict):
        return {}
    values = metric.get("values") or {}
    return values.get("service_view") or {}


def case_lines(task, case, width, verbose=False):
    yield ""
    heading = f"[{safe_text(task['name'])}/{safe_text(case['id'])}] " + (
        f"{safe_text(case['status'])}  "
        f"scenario={safe_text(task.get('scenario'))}"
    )
    rate = None
    if case["status"] not in ("discarded", "skipped", "interrupted"):
        rate = (saved_token_view(case).get("total_token_throughput")
                or {}).get("value")
    label = f"total_token_throughput={number(rate)} tokens/s"
    labels = [heading, *context_lines(case, verbose)]
    if task.get("category") == "generation_performance":
        labels.append(label)
    yield from boxed(labels, width)
    if case["status"] == "discarded":
        yield "指标: N/A (Case 已丢弃，残留指标不作为有效结果)"
        if case.get("reason") or case.get("error"):
            yield "诊断: " + safe_text(case.get("reason") or case["error"])
        return
    metrics = case.get("metrics") or {}
    if not metrics:
        yield "指标: N/A (未保存指标或指标计算未完成)"
    for name, metric in metrics.items():
        if (name == "token_throughput"
                and task.get("category") == "model_service_quality"):
            continue
        if not isinstance(metric, dict):
            yield f"{safe_text(name)}: N/A (指标结构不可用)"
            continue
        yield f"指标 {safe_text(name)}: status=" + safe_text(
            metric.get("status")
        ) + "  boundary=" + safe_text(metric.get("observation_boundary"))
        if metric.get("condition"):
            yield "condition=" + safe_text(metric["condition"])
        entries = metric_entries(metric)
        if not entries:
            yield "N/A: " + safe_text(
                metric.get("reason") or metric.get("error") or
                "未保存可用统计；指标可能不适用或尚未计算"
            )
            continue
        shown, hidden = (entries, []) if verbose else summary_entries(entries)
        yield from metric_lines(shown, width, verbose)
        if hidden:
            breakdown_count = sum(is_breakdown(path) for path, _ in hidden)
            if breakdown_count:
                yield (
                    f"已隐藏 {breakdown_count} 项学科/子类别明细，"
                    "使用 --verbose 查看全部。"
                )
            other_count = len(hidden) - breakdown_count
            if other_count:
                yield (
                    f"另有 {other_count} 项已保存指标，"
                    "使用 --verbose 查看全部。"
                )
            for path, item in hidden:
                if item.get("reason") or item.get("condition"):
                    yield safe_text(path) + ": " + safe_text(
                        item.get("reason") or item.get("condition")
                    )
                if "fail" in path and isinstance(item.get("value"),
                                                (int, float)):
                    if item["value"] > 0:
                        yield safe_text(path) + "=" + number(item["value"])
        if metric.get("reason") or metric.get("error"):
            yield "指标诊断: " + safe_text(
                metric.get("reason") or metric.get("error")
            )
        values = metric.get("values") or {}
        service = (
            values.get("service_view") or {}
            if isinstance(values, dict) else {}
        )
        if verbose and metric.get("metadata"):
            yield "metadata: " + safe_text(json.dumps(
                metric["metadata"], ensure_ascii=False,
            ))
        if service and verbose:
            yield "吞吐量范围: 正式 Case 窗口；仅成功请求及其 Token 计入分子。"
            metadata = metric.get("metadata") or {}
            details = []
            for key in ("benchmark_duration", "total_requests",
                        "successful_requests", "failed_requests"):
                if key in service:
                    value = (service[key] or {}).get("value")
                elif key in metadata:
                    value = metadata[key]
                else:
                    continue
                unit = " s" if key == "benchmark_duration" else ""
                details.append(f"{key}={number(value)}{unit}")
            if details:
                yield "  ".join(details)
        if verbose and any(
            path.startswith("request_view/") for path, _ in entries
        ):
            yield "请求分布: 正式成功请求；ITL/TPOT 的 N 为请求数。"
            yield "weight_sum 为 Token/间隔权重；已保存的加权分位数直接展示。"
    for key in ("error", "reason", "cleanup_error", "metric_errors"):
        if case.get(key):
            yield f"诊断 {key}: {safe_text(case[key])}"

    if verbose:
        for key in ("raw", "stop_reason", "metrics_status",
                    "collection_complete"):
            if case.get(key) is not None:
                yield f"{key}: {safe_text(case[key])}"


def result_links(result_path, tasks=(), cases=(), failures=()):
    """Link saved summaries and opt-in request diagnostics."""
    path = Path(result_path).resolve()
    command = "luban-meter report --input " + shlex.quote(str(path))
    text_command = command + " --format text --verbose --all"
    for key, selectors in (("--task", tasks), ("--case", cases)):
        for selector in selectors:
            text_command += " " + key + " " + shlex.quote(selector)
    lines = [
        "结果/JSON: " + str(path),
        "重新查看/Details: " + text_command,
    ]
    for task, case in failures:
        selector = f"{task['name']}/{case['id']}"
        lines.append(
            "定位异常: " + command + " --format text --verbose --case "
            + shlex.quote(selector)
        )
        lines.append(
            "请求错误: " + command + " --format text --details errors --case "
            + shlex.quote(selector)
        )
    if failures:
        lines.extend((
            "原始文件齐全时也可导出详情 HTML：",
            command + " --include-requests",
        ))
    return lines


def render_text(payload, *, tasks=(), cases=(), all_cases=False, width=None,
                verbose=False, result_path=None):
    """Render only saved final results, retaining their units and counts."""
    all_tasks = validate_result(payload)
    selected_tasks, selected = select_cases(all_tasks, tasks, cases)
    # Terminal/COLUMNS may report a very wide SSH or IDE window. Never
    # allocate all that spare space to Metric; cap the report at 120 cells.
    width = max(20, min(MAX_REPORT_WIDTH, width or
                        shutil.get_terminal_size((100, 24)).columns))
    run = payload["kind"] == "run"
    shown = selected if all_cases else selected[:CASE_LIMIT]
    statuses = Counter(case["status"] for _, case in selected)
    lines = [
        "LuBan-Meter 终端结果摘要",
        f"运行: {safe_text(payload.get('id', payload.get('name')))}  "
        f"状态/Status: {safe_text(payload.get('status'))}",
        f"任务: {len(selected_tasks)}/{len(all_tasks)}  "
        f"Case: 匹配 {len(selected)}，展示 {len(shown)}",
        "Case 状态: " + ("  ".join(
            f"{key}={count}" for key, count in sorted(statuses.items())
        ) or "N/A (无 Case)"),
    ]
    if run:
        lines.append(f"整轮 duration={number(payload.get('duration_s'))} s")
    if run and any(t.get("category") == "generation_performance"
                   for t in all_tasks):
        view = saved_token_view(payload)
        total = (view.get("total_tokens") or {}).get("value")
        formal_duration = (view.get("benchmark_duration") or {}).get("value")
        rate = (view.get("total_token_throughput") or {}).get("value")
        lines.extend((
            f"性能评测正式测试时长={number(formal_duration)} s",
            f"性能评测输入＋输出 Token 总数={number(total)} token",
            f"性能评测总体 total_token_throughput={number(rate)} tokens/s",
        ))
        metric = (payload.get("metrics") or {}).get("token_throughput") or {}
        if metric.get("reason"):
            lines.append("整轮指标诊断: " + safe_text(metric["reason"]))
        elif not metric:
            lines.append("整轮指标未保存")
        if verbose and metric.get("metadata"):
            lines.append("整轮统计依据: " + safe_text(json.dumps(
                metric["metadata"], ensure_ascii=False,
            )))
    lines = list(boxed(lines, width))
    lines.extend(boxed(
        [safe_text(line) for line in best_summary_lines(selected)], width
    ))
    if payload.get("errors"):
        lines.append("运行诊断: " + safe_text(payload["errors"]))
    for task in selected_tasks:
        if task.get("errors") or not task["cases"]:
            lines.append(
                f"任务 {safe_text(task['name'])}: {safe_text(task['status'])} "
                f"errors={safe_text(task.get('errors'))}"
            )
    for task, case in shown:
        lines.extend(case_lines(
            task, case, width, verbose=verbose
        ))
    if not selected:
        lines.append("无 Case 结果；请检查任务状态与输入结果。")
    if len(shown) < len(selected):
        lines.append(
            f"省略 {len(selected) - len(shown)} 个 Case；使用 --all 查看全部，"
            "或 --task NAME --case ID 筛选。"
        )
    lines.append("N/A=未记录/不适用；0 为真实值。详细口径使用 --verbose。")
    if verbose:
        lines.extend((
            "Mean/Value=已保存均值/标量；标量的 P50/P95 不适用。",
            "N=已保存样本数；长度参数为目标/上限，实际值见指标。",
        ))
        for key in ("started_at", "finished_at", "errors"):
            if payload.get(key) is not None:
                lines.append(f"{key}: {safe_text(payload[key])}")
    if len(all_tasks) > 1:
        lines.append("同名 Case 属于各自任务；用 --case TASK/CASE 精确选择。")
    rendered = "\n".join(
        part for line in lines for part in wrapped(line, width)
    )
    if result_path is not None:
        # Keep commands unbroken so copying from SSH/less remains executable.
        rendered += "\n" + "\n".join(
            safe_text(line) for line in result_links(
                result_path, tasks, cases,
                failures=[(task, case) for task, case in shown
                          if case["status"] != "success"
                          or (case.get("coverage") or {}).get("failed")
                          or case.get("metric_errors")
                          or any(isinstance(metric, dict) and (
                              metric.get("error") or metric.get("status")
                              in ("failed", "partial_failed", "interrupted")
                          ) for metric in
                              (case.get("metrics") or {}).values())],
            )
        )
    return rendered
