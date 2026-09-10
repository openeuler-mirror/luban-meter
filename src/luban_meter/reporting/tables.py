"""Declarative tables and a generic fallback for arbitrary metric dicts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from luban_meter.reporting.data import Task, lookup, metric_rows, numeric


@dataclass
class Table:
    title: str
    columns: list[dict[str, str]]
    records: list[dict[str, Any]]
    charts: list[dict[str, Any]] = field(default_factory=list)
    unit: str = ""
    mapping: bool = False

    def reading(self, record: dict[str, Any], path: str) -> tuple[Any, str]:
        """Keep source units when reading a detached table record."""
        unit = self.unit
        if self.mapping and path != "/value" and not path.startswith(
            "/value/"
        ):
            unit = ""
        return lookup(record, path, unit=unit)


def display(value: Any, unit: str = "") -> str:
    if value is None or isinstance(value, (dict, list)):
        return "—"
    if isinstance(value, float) and not numeric(value):
        return "—"
    text = format(value, ".6g") if numeric(value) else str(value)
    return f"{text} {unit}" if unit else text


def _table(spec: Any, metrics: Mapping[str, Any]) -> Table:
    if not isinstance(spec, Mapping):
        raise ValueError("table 必须是字典")
    title = spec.get("title", "指标摘要")
    if not isinstance(title, str):
        raise ValueError("table.title 必须是字符串")
    columns = spec.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("table.columns 必须是非空列表")
    for column in columns:
        if not isinstance(column, dict) or not isinstance(
            column.get("label"), str
        ):
            raise ValueError("column 必须声明 path 和 label")
        lookup({}, column.get("path"))
    value, unit = lookup(metrics, spec.get("path", ""), unwrap=False)
    mapping = spec.get("mapping", False)
    if not isinstance(mapping, bool):
        raise ValueError("table.mapping 必须是布尔值")
    if mapping:
        if value is not None and not isinstance(value, Mapping):
            raise ValueError("mapping 表格需要字典")
        records = [
            {"key": key, "value": item}
            for key, item in sorted((value or {}).items())
            if key != "unit" or not isinstance(item, str)
        ]
    elif isinstance(value, list):
        if any(not isinstance(row, dict) for row in value):
            raise ValueError("表格记录必须是字典")
        records = value
    elif isinstance(value, dict):
        records = [value]
    elif value is None:
        records = []
    else:
        raise ValueError("table.path 必须指向字典或字典列表")
    charts = spec.get("charts", [])
    if not isinstance(charts, list):
        raise ValueError("table.charts 必须是列表")
    for chart in charts:
        if (
            not isinstance(chart, dict)
            or not isinstance(chart.get("type"), str)
            or chart["type"] not in {"line", "bar"}
        ):
            raise ValueError("chart.type 只支持 line 或 bar")
        lookup({}, chart.get("y"))
        if chart.get("x") is not None:
            lookup({}, chart["x"])
        if chart["type"] == "line" and not chart.get("x"):
            raise ValueError("折线图需要 x")
        groups = chart.get("group_by", [])
        if not isinstance(groups, list):
            raise ValueError("group_by 必须是路径列表")
        for path in groups:
            lookup({}, path)
    return Table(title, columns, records, charts, unit, mapping)


def tables_for(task: Task) -> list[Table]:
    metrics = task.data.get("metrics", {})
    spec = task.data.get("metadata", {}).get("report")
    if spec is not None:
        try:
            if (
                not isinstance(spec, Mapping)
                or not isinstance(spec.get("tables"), list)
                or not spec["tables"]
            ):
                raise ValueError("metadata.report.tables 必须是非空列表")
            return [_table(table, metrics) for table in spec["tables"]]
        except ValueError as exc:
            note = f"报告声明无效，使用自动摘要：{exc}"
            if note not in task.notes:
                task.notes.append(note)
    rows = []
    for row in metric_rows(metrics):
        if len(rows) == 30:
            note = "自动摘要显示前 30 个指标，完整指标见 CSV。"
            if note not in task.notes:
                task.notes.append(note)
            break
        rows.append(
            {
                "metric": row["metric"],
                "reading": row["value"],
                "measure_unit": row["unit"],
            }
        )
    return [
        Table(
            "指标摘要",
            [
                {"path": "/metric", "label": "指标"},
                {"path": "/reading", "label": "值"},
                {"path": "/measure_unit", "label": "单位"},
            ],
            rows,
        )
    ]


def table_values(table: Table) -> list[list[str]]:
    return [
        [
            display(*table.reading(record, column["path"]))
            for column in table.columns
        ]
        for record in table.records
    ]
