"""Self-contained HTML reports over the new, unversioned result contract."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

from .data import format_number, metric_rows, validate_result
from .data import load_result as load_result
from .prefix_cache import METRIC, cache_report_metric, redact_cache_salts
from .request_details import load_details


def report_case(case):
    metrics = case.get("metrics", {})
    if METRIC not in metrics:
        return case
    return {**case, "metrics": {
        **metrics, METRIC: cache_report_metric(metrics[METRIC]),
    }}


def report_data(payload, *, include_requests=False, raw_root=None):
    tasks = validate_result(payload)
    tasks = [{**task, "cases": [report_case(c) for c in task["cases"]]}
             for task in tasks]
    rows = []
    observations = []
    for task_index, task in enumerate(tasks):
        for case_index, case in enumerate(task["cases"]):
            if case["status"] == "discarded":
                continue
            label = {
                "task": task["name"],
                "case": case["id"],
                "category": task["category"],
                "task_index": task_index,
                "case_index": case_index,
            }
            for row in metric_rows(case.get("metrics", {})):
                rows.append({**label, **row})
            server_metrics = case.get("observation", {}).get("metrics", {})
            for row in metric_rows(server_metrics, "/server_observation"):
                rows.append(
                    {**label, "category": "service_observation", **row}
                )
            device = case.get("device_observation", {})
            snapshots = device.get("timeseries", [])
            step = max(1, (len(snapshots) + 499) // 500)
            origin = case.get("window", {}).get("started_at")
            for snapshot in snapshots[::step]:
                timestamp = snapshot.get("finished_at")
                if timestamp is None and origin is not None:
                    timestamp = origin + snapshot["elapsed"]
                if timestamp is None:
                    continue
                for sample in snapshot.get("devices", []):
                    for name, value in sample.items():
                        if name == "index" or value is None:
                            continue
                        observations.append(
                            {
                                **label,
                                "metric": "device/" + name,
                                "labels": {"device": sample["index"]},
                                "time": timestamp,
                                "value": value,
                            }
                        )
                for name, value in (snapshot.get("cpu") or {}).items():
                    if value is not None:
                        observations.append(
                            {
                                **label,
                                "metric": "cpu/" + name,
                                "labels": {},
                                "time": timestamp,
                                "value": value,
                            }
                        )
            host = case.get("host_observation", {})
            for row in metric_rows(host.get("summary", {}), "/host"):
                if row["metric"].endswith("/covered_seconds"):
                    row["unit"] = "s"
                rows.append({**label, "category": "host_observation", **row})
            samples = host.get("timeseries", [])
            step = max(1, (len(samples) + 499) // 500)
            for sample in samples[::step]:
                if sample.get("error"):
                    continue
                for group in ("cpu", "memory", "filesystems", "disks",
                              "networks"):
                    nodes = sample.get(group, {})
                    nodes = nodes if isinstance(nodes, list) else [nodes]
                    for node in nodes:
                        labels = {k: node[k] for k in (
                            "device", "mountpoint", "fstype"
                        ) if k in node}
                        for key, value in node.items():
                            if key in labels or not (
                                value is None or isinstance(
                                    value, (int, float)
                                )
                            ):
                                continue
                            observations.append({
                                **label, "metric": f"host/{group}/{key}",
                                "labels": labels,
                                "time": sample["finished_at"],
                                "value": value,
                            })
            samples = case.get("observation", {}).get("snapshots", [])
            # Sample charts only; preserve complete raw observations.
            step = max(1, (len(samples) + 499) // 500)
            for sample in samples[::step]:
                for name, metric in sample.get("metrics", {}).items():
                    if metric.get("type") not in (
                        "gauge",
                        "counter",
                        "untyped",
                    ):
                        continue
                    for series in metric.get("samples", []):
                        observations.append(
                            {
                                **label,
                                "metric": name,
                                "labels": series.get("labels", {}),
                                "time": sample["finished_at"],
                                "value": series["value"],
                            }
                        )
    rendered_tasks = []
    for task in tasks:
        rendered_cases = []
        for case in task["cases"]:
            rendered = dict(case)
            if "observation" in case:
                observation = case["observation"]
                rendered["observation"] = {
                    key: value
                    for key, value in observation.items()
                    if key != "snapshots"
                }
                failures = [
                    s
                    for s in observation.get("snapshots", [])
                    if s.get("error")
                ]
                rendered["observation"]["failure_examples"] = failures[:20]
            if "device_observation" in case:
                rendered["device_observation"] = {
                    key: value
                    for key, value in case["device_observation"].items()
                    if key != "timeseries"
                }
            if "host_observation" in case:
                host = case["host_observation"]
                rendered["host_observation"] = {
                    key: value for key, value in host.items()
                    if key != "timeseries"
                }
                rendered["host_observation"]["failure_examples"] = [
                    s for s in host.get("timeseries", []) if s.get("error")
                ][:20]
            rendered_cases.append(rendered)
        rendered_tasks.append({**task, "cases": rendered_cases})
    rendered_result = (
        {**payload, "tasks": rendered_tasks}
        if payload["kind"] == "run"
        else rendered_tasks[0]
    )
    data = {
        "result": rendered_result,
        "rows": rows,
        "observations": observations,
    }
    if include_requests:
        if raw_root is None:
            raise ValueError("full report requires raw_root")
        data["request_details"] = [
            [load_details(case, task, raw_root, run=payload["kind"] == "run")
             for case in task["cases"]]
            for task in tasks
        ]
    return redact_cache_salts(data)


def run_summary_html(payload):
    """Render saved run metrics, without deriving values or reading raw."""
    if payload["kind"] != "run":
        return ""
    metric = (payload.get("metrics") or {}).get("token_throughput") or {}
    view = (metric.get("values") or {}).get("service_view") or {}

    def text(value):
        if value is None:
            return "N/A"
        if type(value) in (float, int):
            return format_number(value)
        return escape(str(value))

    nodes = [
        ("整轮 duration", {"value": payload.get("duration_s"), "unit": "s"}),
        ("性能评测正式测试时长", view.get("benchmark_duration") or {}),
        ("性能评测输入＋输出 Token 总数", view.get("total_tokens") or {}),
        ("total_token_throughput", view.get("total_token_throughput") or {}),
    ]
    has_performance = any(t.get("category") == "generation_performance"
                          for t in payload.get("tasks", []))
    if not has_performance:
        nodes = nodes[:1]
    # Lead with the saved run-level total throughput as a large number;
    # the table below keeps the full ledger of the same saved values.
    hero = ""
    if has_performance:
        throughput = view.get("total_token_throughput") or {}
        hero = (
            '<div class="run-hero" aria-label="整轮总 Token 吞吐量">'
            '<span class="run-hero-label">'
            "总 Token 吞吐量（输入＋输出，整轮平均）</span>"
            '<span class="run-hero-value">' + text(throughput.get("value"))
            + '</span><span class="run-hero-unit">'
            + text(throughput.get("unit") or "token/s") + "</span></div>"
        )
    rows = "".join(
        "<tr><td>" + escape(label) + "</td><td>" + text(node.get("value"))
        + "</td><td>" + text(node.get("unit")) + "</td></tr>"
        for label, node in nodes
    )
    reason = metric.get("reason")
    if not metric and len(nodes) > 1:
        reason = "性能评测指标未保存"
    diagnostic = (
        '<p class="muted">' + text(reason) + "</p>" if reason else ""
    )
    metadata = escape(json.dumps(metric.get("metadata") or {},
                                 ensure_ascii=False, indent=2))
    return (
        '<div id="run-throughput-summary" class="overview-result-column">'
        "<h3>运行信息与性能评测汇总</h3>" + hero
        + '<div class="table-wrap"><table><thead><tr><th>指标</th>'
        "<th>保存值</th><th>单位</th></tr></thead><tbody>"
        + rows + "</tbody></table></div>" + diagnostic
        + "<details><summary>统计范围与计算口径</summary><pre>"
        + metadata + "</pre></details></div>"
    )


def write_html(payload, destination, *, include_requests=False, raw_root=None):
    data = json.dumps(
        report_data(
            payload, include_requests=include_requests, raw_root=raw_root
        ), ensure_ascii=False, allow_nan=False
    )
    data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    replacements = {
        "__REPORT_DATA__": data,
        "__RUN_SUMMARY__": run_summary_html(redact_cache_salts(payload)),
    }
    html = re.sub(
        r"__REPORT_DATA__|__RUN_SUMMARY__",
        lambda match: replacements[match.group()],
        TEMPLATE,
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


TEMPLATE = (
    Path(__file__).with_name("template.html").read_text(encoding="utf-8")
).replace(
    "__REQUEST_SCRIPT__",
    Path(__file__).with_name("requests.js").read_text(encoding="utf-8"),
).replace(
    "__CASE_SCRIPT__",
    Path(__file__).with_name("case_analysis.js").read_text(encoding="utf-8"),
).replace(
    "__SUMMARY_SCRIPT__",
    Path(__file__).with_name("summary_cards.js").read_text(encoding="utf-8"),
).replace(
    "__OVERVIEW_SCRIPT__",
    Path(__file__).with_name("report_overview.js").read_text(encoding="utf-8"),
).replace(
    "__RESOURCE_SCRIPT__",
    Path(__file__).with_name("resources.js").read_text(encoding="utf-8"),
)
