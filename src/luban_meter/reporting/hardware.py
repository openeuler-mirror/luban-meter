"""A single hardware overview built only from recorded run information."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from luban_meter.reporting.charts import available_chart_fonts
from luban_meter.reporting.result_reader import ReportTask, is_finite_number
from luban_meter.reporting.tables import format_metric_value

LABELS = {
    "vendor": "厂商",
    "tool": "采集工具",
    "devices": "设备",
    "index": "编号",
    "name": "名称",
    "cpu": "CPU",
    "model": "型号",
    "cores": "核心数",
    "sockets": "插槽数",
    "architecture": "架构",
    "os": "操作系统",
    "kernel": "内核",
    "driver_version": "驱动版本",
    "memory_total_gb": "总内存 (GB)",
    "memory_total_mb": "总内存 (MB)",
    "memory_used_avg_mb": "平均内存占用 (MB)",
    "device_count": "设备数",
    "sample_count": "采样数",
    "duration_seconds": "监控时长 (s)",
    "interval_seconds": "采样间隔 (s)",
    "error_count": "采集错误数",
    "total_power_avg_watts": "总平均功耗 (W)",
    "total_energy_wh": "总能耗 (Wh)",
    "source": "数据来源",
}
for _stat in ("avg", "p50", "p90", "p99"):
    _label = "Mean" if _stat == "avg" else _stat.upper()
    LABELS.update(
        {
            f"utilization_{_stat}": f"利用率 {_label} (%)",
            f"power_{_stat}_watts": f"功耗 {_label} (W)",
            f"temperature_{_stat}_celsius": f"温度 {_label} (°C)",
        }
    )

# scope, field, Chinese title, English title, unit, legacy chart
# filename
PANELS = (
    (
        "devices",
        "utilization",
        "设备利用率",
        "Device utilization",
        "%",
        "gpu_utilization.png",
    ),
    ("devices", "power", "设备功耗", "Device power", "W", "gpu_power.png"),
    (
        "devices",
        "temperature",
        "设备温度",
        "Device temperature",
        "°C",
        "gpu_temperature.png",
    ),
    (
        "devices",
        "memory_used",
        "设备内存占用",
        "Device memory",
        "MB",
        "gpu_memory.png",
    ),
    (
        "cpu",
        "utilization",
        "CPU 利用率",
        "CPU utilization",
        "%",
        "cpu_utilization.png",
    ),
    (
        "cpu",
        "memory_used",
        "系统内存占用",
        "System memory",
        "MB",
        "memory_usage.png",
    ),
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _wrap(text: str, width: int, supports_cjk: bool) -> str:
    """Wrap long identifiers and CJK text without hiding recorded
    values.
    """
    if not supports_cjk:
        text = "".join(
            char.encode("unicode_escape").decode("ascii")
            if unicodedata.east_asian_width(char) in "WF"
            else char
            for char in text
        )
    lines = []
    for paragraph in text.splitlines() or [""]:
        current, used = "", 0
        for char in paragraph:
            size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
            if used + size > width:
                lines.append(current)
                current, used = "", 0
            current += char
            used += size
        lines.append(current)
    return "\n".join(lines)


def _iter_hardware_facts(value: Any, supports_cjk: bool, prefix: str = ""):
    """Keep every static or aggregate field, including unknown fields."""
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            label = LABELS.get(key, key) if supports_cjk else key
            path = f"{prefix} / {label}" if prefix else str(label)
            yield from _iter_hardware_facts(item, supports_cjk, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            # Device identifiers may differ from their position in the
            # list.
            identity = _as_mapping(item).get("index", index)
            yield from _iter_hardware_facts(
                item, supports_cjk, f"{prefix} {identity}"
            )
    else:
        yield prefix, format_metric_value(value)


def _table_rows(facts, supports_cjk):
    cells = [
        (_wrap(label, 36, supports_cjk), _wrap(value, 42, supports_cjk))
        for label, value in facts
    ]
    rows = []
    for start in range(0, len(cells), 2):
        row = [*cells[start]]
        row.extend(cells[start + 1] if start + 1 < len(cells) else ("", ""))
        rows.append(row)
    return rows


def _collect_monitoring_curves(monitoring):
    raw = monitoring.get("timeseries", [])
    if not isinstance(raw, list):
        return []
    samples = sorted(
        (
            sample
            for sample in raw
            if isinstance(sample, Mapping)
            and is_finite_number(sample.get("elapsed"))
        ),
        key=lambda sample: sample["elapsed"],
    )
    devices = []
    cpu = []
    for sample in samples:
        values = sample.get("devices", [])
        devices.append(
            {
                str(device["index"]): device
                for device in values
                if isinstance(device, Mapping)
                and isinstance(device.get("index"), (str, int))
                and not isinstance(device["index"], bool)
            }
            if isinstance(values, list)
            else {}
        )
        cpu.append(_as_mapping(sample.get("cpu")))
    indices = sorted({key for sample in devices for key in sample})
    elapsed = [sample["elapsed"] for sample in samples]
    panels = []
    for scope, field, chinese_title, english_title, unit, _filename in PANELS:
        series = {}
        records = (
            {
                f"Device {index}": [row.get(index, {}) for row in devices]
                for index in indices
            }
            if scope == "devices"
            else {"CPU": cpu}
        )
        for label, rows in records.items():
            values = [row.get(field) for row in rows]
            if any(is_finite_number(value) for value in values):
                series[label] = [
                    value if is_finite_number(value) else math.nan
                    for value in values
                ]
        if series:
            panels.append(
                (chinese_title, english_title, unit, elapsed, series)
            )
    return panels


def _load_saved_charts(task: ReportTask, monitoring):
    """Use existing PNGs if a saved v2 result has no drawable samples."""
    from matplotlib.image import imread

    names = monitoring.get("charts", [])
    if not isinstance(names, list):
        return []
    directories = []
    if task.source:
        directories.append(task.source.parent / "raw" / "artifacts")
    declared = _as_mapping(task.result_payload.get("artifacts")).get(
        "directory"
    )
    if isinstance(declared, str):
        directory = Path(declared)
        if not directory.is_absolute() and task.source:
            directory = task.source.parent / directory
        directories.append(directory)
    charts = []
    for name in dict.fromkeys(
        chart_name for chart_name in names if isinstance(chart_name, str)
    ):
        if Path(name).name != name or Path(name).suffix.lower() != ".png":
            continue
        for directory in directories:
            try:
                pixels = imread(directory / name)
            except (OSError, ValueError, SyntaxError):
                continue
            charts.append((name, pixels))
            break
    return charts


def build_hardware_figure(task: ReportTask):
    """Build one figure, without querying hardware or changing the
    result.
    """
    environment = _as_mapping(task.result_payload.get("environment"))
    monitoring = _as_mapping(environment.get("device_monitoring"))
    hardware = _as_mapping(environment.get("hardware_environment"))
    if not hardware:
        hardware = _as_mapping(monitoring.get("hardware_environment"))
    summary = {
        key: value
        for key, value in monitoring.items()
        if key not in {"timeseries", "charts", "hardware_environment"}
    }
    if not hardware and not monitoring:
        return None

    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fonts = available_chart_fonts()
    supports_cjk = len(fonts) > 1
    sections = []
    for chinese_title, english_title, data in (
        ("硬件环境", "Hardware environment", hardware),
        ("监控摘要", "Monitoring summary", summary),
    ):
        rows = _table_rows(
            _iter_hardware_facts(data, supports_cjk), supports_cjk
        )
        if rows:
            sections.append(
                (chinese_title if supports_cjk else english_title, rows)
            )
    curves = _collect_monitoring_curves(monitoring)
    saved = [] if curves else _load_saved_charts(task, monitoring)
    if not sections and not curves and not saved:
        return None

    heights = [
        0.4
        + sum(
            max(
                escape_markdown_cell.count("\n") + 1
                for escape_markdown_cell in row
            )
            * 0.24
            for row in rows
        )
        for _, rows in sections
    ]
    count = len(curves) or len(saved)
    heights.extend([3.2] * ((count + 1) // 2))
    with matplotlib.rc_context({"font.family": fonts, "font.size": 9}):
        figure = Figure(
            figsize=(14, sum(heights) + 0.85), layout="constrained"
        )
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(len(heights), 2, height_ratios=heights)
        title = "硬件环境与监控总览" if supports_cjk else "Hardware overview"
        name = _as_mapping(task.result_payload.get("metadata")).get(
            "display_name"
        )
        figure.suptitle(
            _wrap(f"{title} · {name or task.name}", 125, supports_cjk),
            fontsize=14,
        )
        for index, (title, rows) in enumerate(sections):
            axes = figure.add_subplot(grid[index, :])
            axes.axis("off")
            axes.set_title(title, loc="left", fontsize=11)
            table = axes.table(
                cellText=rows,
                cellLoc="left",
                colWidths=[0.19, 0.31, 0.19, 0.31],
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            units = [
                max(
                    escape_markdown_cell.count("\n") + 1
                    for escape_markdown_cell in row
                )
                for row in rows
            ]
            for (
                row,
                column,
            ), escape_markdown_cell in table.get_celld().items():
                escape_markdown_cell.set_height(units[row] / sum(units))
                escape_markdown_cell.set_edgecolor("#dce3ec")
                escape_markdown_cell.set_facecolor(
                    "#edf2f8" if column % 2 == 0 else "white"
                )
                escape_markdown_cell.PAD = 0.04
        for index, (
            chinese_title,
            english_title,
            unit,
            elapsed,
            series,
        ) in enumerate(curves):
            axes = figure.add_subplot(
                grid[len(sections) + index // 2, index % 2]
            )
            for label, values in series.items():
                axes.plot(elapsed, values, label=label, linewidth=1.4)
            axes.set_title(
                chinese_title if supports_cjk else english_title,
                loc="left",
                fontsize=11,
            )
            axes.set_xlabel("经过时间 (s)" if supports_cjk else "Elapsed (s)")
            axes.set_ylabel(unit)
            axes.grid(alpha=0.25)
            axes.legend(fontsize=8)
        for index, (name, pixels) in enumerate(saved):
            axes = figure.add_subplot(
                grid[len(sections) + index // 2, index % 2]
            )
            axes.imshow(pixels)
            axes.axis("off")
            labels = {
                panel_definition[5]: panel_definition[2]
                if supports_cjk
                else panel_definition[3]
                for panel_definition in PANELS
            }
            axes.set_title(labels.get(name, Path(name).stem), fontsize=11)
    return figure


def write_hardware_overview(task: ReportTask, output: Path, index: int):
    figure = build_hardware_figure(task)
    if figure is None:
        return None
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"hardware-{index}-overview.png"
    figure.savefig(path, dpi=150)
    figure.clear()
    return path
