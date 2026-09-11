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


def line(x: str, y: str, group_by: list[str]) -> dict[str, Any]:
    return {"type": "line", "x": x, "y": y, "group_by": group_by}


def bar(y: str, x: str | None = None) -> dict[str, Any]:
    return {"type": "bar", "x": x, "y": y}
