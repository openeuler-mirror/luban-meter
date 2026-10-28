"""Self-contained HTML reports over the new, unversioned result contract."""

from __future__ import annotations

import json
from pathlib import Path

from .data import load_result as load_result
from .data import metric_rows, validate_result
from .request_details import load_details


def report_data(payload, *, include_requests=False, raw_root=None):
    tasks = validate_result(payload)
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
            origin = case.get("window", {}).get("started_at") or 0
            for snapshot in snapshots[::step]:
                for sample in snapshot.get("devices", []):
                    for name, value in sample.items():
                        if name == "index" or value is None:
                            continue
                        observations.append(
                            {
                                **label,
                                "metric": "device/" + name,
                                "labels": {"device": sample["index"]},
                                "time": origin + snapshot["elapsed"],
                                "value": value,
                            }
                        )
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
    return data


def write_html(payload, destination, *, include_requests=False, raw_root=None):
    data = json.dumps(
        report_data(
            payload, include_requests=include_requests, raw_root=raw_root
        ), ensure_ascii=False, allow_nan=False
    )
    data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    html = TEMPLATE.replace("__REPORT_DATA__", data)
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
)
