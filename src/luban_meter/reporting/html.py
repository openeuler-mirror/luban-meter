"""Self-contained HTML reports over the new, unversioned result contract."""

from __future__ import annotations

import json
from pathlib import Path

STATUSES = {
    "success",
    "partial_failed",
    "failed",
    "interrupted",
    "skipped",
    "discarded",
}


def validate_result(payload):
    if not isinstance(payload, dict) or "schema_version" in payload:
        raise ValueError(
            "expected a new result object without a version field"
        )
    if payload.get("kind") not in ("run", "task"):
        raise ValueError("result.kind must be run or task")
    if payload.get("status") not in STATUSES:
        raise ValueError("invalid result status")
    tasks = payload.get("tasks") if payload["kind"] == "run" else [payload]
    if not isinstance(tasks, list):
        raise ValueError("result.tasks must be a list")
    for task in tasks:
        if not isinstance(task, dict) or task.get("status") not in STATUSES:
            raise ValueError("invalid task result")
        if not isinstance(task.get("cases"), list):
            raise ValueError("task.cases must be a list")
        if any(
            not isinstance(task.get(key), str) for key in ("name", "category")
        ):
            raise ValueError("task name and category must be strings")
        for case in task["cases"]:
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("id"), str)
                or case.get("status") not in STATUSES
            ):
                raise ValueError("invalid Case result")
            if not isinstance(case.get("metrics", {}), dict):
                raise ValueError("Case metrics must be an object")
    return tasks


def metric_rows(value, path="", unit="", count=None):
    if isinstance(value, dict):
        unit = value.get("unit", unit)
        count = value.get("sample_count", value.get("count", count))
        if "value" in value:
            yield {
                "metric": path,
                "value": value["value"],
                "unit": unit,
                "count": count,
            }
        else:
            for name, item in value.items():
                if name not in ("unit", "count", "sample_count", "weight_sum"):
                    yield from metric_rows(item, f"{path}/{name}", unit, count)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from metric_rows(item, f"{path}/{index}", unit, count)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield {"metric": path, "value": value, "unit": unit, "count": count}


def report_data(payload):
    tasks = validate_result(payload)
    rows = []
    observations = []
    for task in tasks:
        for case in task["cases"]:
            if case["status"] == "discarded":
                continue
            label = {
                "task": task["name"],
                "case": case["id"],
                "category": task["category"],
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
    return {
        "result": rendered_result,
        "rows": rows,
        "observations": observations,
    }


def write_html(payload, destination):
    data = json.dumps(
        report_data(payload), ensure_ascii=False, allow_nan=False
    )
    data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    html = TEMPLATE.replace("__REPORT_DATA__", data)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


def load_result(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_result(payload)
    return payload


TEMPLATE = (
    Path(__file__).with_name("template.html").read_text(encoding="utf-8")
)
