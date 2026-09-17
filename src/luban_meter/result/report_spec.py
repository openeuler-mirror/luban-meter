"""Small helpers for optional, data-only Benchmark report declarations."""

from typing import Any


def table(
    title: str,
    path: str,
    columns: dict[str, str],
    *,
    mapping: bool = False,
    charts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "path": path,
        "mapping": mapping,
        "columns": [
            {"path": key, "label": label} for key, label in columns.items()
        ],
        "charts": charts or [],
    }


def line(
    x_metric_path: str, y_metric_path: str, group_by: list[str]
) -> dict[str, Any]:
    return {
        "type": "line",
        "x": x_metric_path,
        "y": y_metric_path,
        "group_by": group_by,
    }


def bar(
    y_metric_path: str, x_metric_path: str | None = None
) -> dict[str, Any]:
    return {"type": "bar", "x": x_metric_path, "y": y_metric_path}
