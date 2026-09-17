"""Render declared static charts without knowing Benchmark names."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from luban_meter.reporting.result_reader import is_finite_number
from luban_meter.reporting.tables import MetricTable, format_metric_value


@lru_cache(maxsize=1)
def available_chart_fonts() -> list[str]:
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
    table: MetricTable,
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
    fonts = available_chart_fonts()
    supports_cjk = len(fonts) > 1
    for index, chart in enumerate(table.charts):
        groups = {}
        for record in table.records:
            value, unit = table.read_metric(record, chart["y"])
            dimension_values = [
                table.read_metric(record, dimension_path)[0]
                for dimension_path in chart.get("group_by", [])
            ]
            x_value, x_unit = (
                table.read_metric(record, chart["x"])
                if chart.get("x")
                else ("Overall", "")
            )
            key = (
                json.dumps(
                    dimension_values, ensure_ascii=False, sort_keys=True
                ),
                unit,
                x_unit,
            )
            if chart["type"] == "line" and not is_finite_number(x_value):
                continue
            groups.setdefault(key, []).append((x_value, value))
        for (group, unit, x_unit), points in groups.items():
            if not any(is_finite_number(value) for _, value in points):
                continue
            if chart["type"] == "line":
                points.sort(key=lambda point: point[0])
            page_size = 12 if chart["type"] == "bar" else max(1, len(points))
            for start in range(0, len(points), page_size):
                page = points[start : start + page_size]
                title = labels.get(chart["y"], chart["y"])
                dimensions = ", ".join(
                    f"{labels.get(key, key)}={format_metric_value(value)}"
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
                    figure = Figure(figsize=(10, 4.5), layout="constrained")
                    FigureCanvasAgg(figure)
                    axes = figure.subplots()
                    values = [
                        metric_value
                        if is_finite_number(metric_value)
                        else math.nan
                        for _, metric_value in page
                    ]
                    if chart["type"] == "line":
                        axes.plot(
                            [x_value for x_value, _ in page],
                            values,
                            marker="o",
                        )
                    else:
                        ticks = [str(x_value) for x_value, _ in page]
                        if not supports_cjk:
                            aliases = [
                                tick
                                if tick.isascii()
                                else f"Item {tick_index + 1}"
                                for tick_index, tick in enumerate(ticks)
                            ]
                            renamed = [
                                f"{alias}={tick}"
                                for alias, tick in zip(aliases, ticks)
                                if alias != tick
                            ]
                            if renamed:
                                caption += " · " + "; ".join(renamed)
                            ticks = aliases
                        axes.bar(range(len(page)), values, width=0.6)
                        axes.set_xticks(range(len(page)), ticks, rotation=30)
                        margin = max(0.5, (4 - len(page)) / 2)
                        axes.set_xlim(-margin, len(page) - 1 + margin)
                    x_path = chart.get("x") or ""
                    xlabel = labels.get(x_path, x_path)
                    ylabel = (
                        title
                        if supports_cjk or title.isascii()
                        else chart["y"]
                    )
                    if not supports_cjk and not xlabel.isascii():
                        xlabel = chart.get("x", "")
                    axes.set_xlabel(format_metric_value(xlabel, x_unit))
                    axes.set_ylabel(format_metric_value(ylabel, unit))
                    axes.grid(axis="y", alpha=0.25)
                    if unit == "ratio" and all(
                        not is_finite_number(metric_value)
                        or 0 <= metric_value <= 1
                        for metric_value in values
                    ):
                        axes.set_ylim(0, 1.05)
                    figure.savefig(path, dpi=150)
                images.append((caption, path))
    return images
