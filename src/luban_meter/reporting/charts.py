"""Render declared static charts without knowing Benchmark names."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from luban_meter.reporting.data import numeric
from luban_meter.reporting.tables import Table, display


@lru_cache(maxsize=1)
def _fonts() -> list[str]:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [
        name
        for name in (
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "PingFang SC",
            "Arial Unicode MS",
            "Microsoft YaHei",
            "SimHei",
        )
        if name in installed
    ] + ["DejaVu Sans"]


def plot_table(
    table: Table,
    output: Path,
    prefix: str,
) -> list[tuple[str, Path]]:
    if not table.charts:
        return []
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    labels = {column["path"]: column["label"] for column in table.columns}
    images = []
    fonts = _fonts()
    cjk = len(fonts) > 1
    for index, chart in enumerate(table.charts):
        groups = {}
        for record in table.records:
            value, unit = table.reading(record, chart["y"])
            dims = [
                table.reading(record, p)[0]
                for p in chart.get("group_by", [])
            ]
            x, x_unit = (
                table.reading(record, chart["x"])
                if chart.get("x")
                else ("Overall", "")
            )
            key = (
                json.dumps(dims, ensure_ascii=False, sort_keys=True),
                unit,
                x_unit,
            )
            if chart["type"] == "line" and not numeric(x):
                continue
            groups.setdefault(key, []).append((x, value))
        for (group, unit, x_unit), points in groups.items():
            if not any(numeric(value) for _, value in points):
                continue
            if chart["type"] == "line":
                points.sort(key=lambda point: point[0])
            page_size = 12 if chart["type"] == "bar" else max(1, len(points))
            for start in range(0, len(points), page_size):
                page = points[start : start + page_size]
                title = labels.get(chart["y"], chart["y"])
                dimensions = ", ".join(
                    f"{labels.get(key, key)}={display(value)}"
                    for key, value in zip(
                        chart.get("group_by", []), json.loads(group)
                    )
                )
                caption = " · ".join(
                    part for part in (title, dimensions, unit) if part
                )
                digest = hashlib.sha256(
                    f"{prefix}:{index}:{group}:{unit}:{x_unit}:{start}".encode()
                ).hexdigest()[:16]
                output.mkdir(parents=True, exist_ok=True)
                path = output / f"{digest}.png"
                with matplotlib.rc_context({"font.family": fonts}):
                    fig = Figure(figsize=(10, 4.5), layout="constrained")
                    FigureCanvasAgg(fig)
                    ax = fig.subplots()
                    values = [v if numeric(v) else math.nan for _, v in page]
                    if chart["type"] == "line":
                        ax.plot([x for x, _ in page], values, marker="o")
                    else:
                        ticks = [str(x) for x, _ in page]
                        if not cjk:
                            aliases = [
                                tick if tick.isascii() else f"Item {i + 1}"
                                for i, tick in enumerate(ticks)
                            ]
                            renamed = [
                                f"{alias}={tick}"
                                for alias, tick in zip(aliases, ticks)
                                if alias != tick
                            ]
                            if renamed:
                                caption += " · " + "; ".join(renamed)
                            ticks = aliases
                        ax.bar(range(len(page)), values, width=0.6)
                        ax.set_xticks(range(len(page)), ticks, rotation=30)
                        margin = max(0.5, (4 - len(page)) / 2)
                        ax.set_xlim(-margin, len(page) - 1 + margin)
                    x_path = chart.get("x") or ""
                    xlabel = labels.get(x_path, x_path)
                    ylabel = title if cjk or title.isascii() else chart["y"]
                    if not cjk and not xlabel.isascii():
                        xlabel = chart.get("x", "")
                    ax.set_xlabel(display(xlabel, x_unit))
                    ax.set_ylabel(display(ylabel, unit))
                    ax.grid(axis="y", alpha=0.25)
                    if unit == "ratio" and all(
                        not numeric(v) or 0 <= v <= 1 for v in values
                    ):
                        ax.set_ylim(0, 1.05)
                    fig.savefig(path, dpi=150)
                images.append((caption, path))
    return images
