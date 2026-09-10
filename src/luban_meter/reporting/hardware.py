"""A single hardware overview built only from recorded run information."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from luban_meter.reporting.charts import _fonts
from luban_meter.reporting.data import Task, numeric
from luban_meter.reporting.tables import display

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

# scope, field, Chinese title, English title, unit, legacy chart filename
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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _wrap(text: str, width: int, cjk: bool) -> str:
    """Wrap long identifiers and CJK text without hiding recorded values."""
    if not cjk:
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


def _facts(value: Any, cjk: bool, prefix: str = ""):
    """Keep every static or aggregate field, including unknown fields."""
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            label = LABELS.get(key, key) if cjk else key
            path = f"{prefix} / {label}" if prefix else str(label)
            yield from _facts(item, cjk, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            # Device identifiers may differ from their position in the list.
            identity = _mapping(item).get("index", index)
            yield from _facts(item, cjk, f"{prefix} {identity}")
    else:
        yield prefix, display(value)


def _table_rows(facts, cjk):
    cells = [
        (_wrap(label, 36, cjk), _wrap(value, 42, cjk))
        for label, value in facts
    ]
    rows = []
    for start in range(0, len(cells), 2):
        row = [*cells[start]]
        row.extend(cells[start + 1] if start + 1 < len(cells) else ("", ""))
        rows.append(row)
    return rows


def _curves(monitoring):
    raw = monitoring.get("timeseries", [])
    if not isinstance(raw, list):
        return []
    samples = sorted(
        (
            sample
            for sample in raw
            if isinstance(sample, Mapping) and numeric(sample.get("elapsed"))
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
        cpu.append(_mapping(sample.get("cpu")))
    indices = sorted({key for sample in devices for key in sample})
    elapsed = [sample["elapsed"] for sample in samples]
    panels = []
    for scope, field, cn, en, unit, _filename in PANELS:
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
            if any(numeric(value) for value in values):
                series[label] = [
                    value if numeric(value) else math.nan for value in values
                ]
        if series:
            panels.append((cn, en, unit, elapsed, series))
    return panels


def _saved_charts(task: Task, monitoring):
    """Use existing PNGs if a saved v2 result has no drawable samples."""
    from matplotlib.image import imread

    names = monitoring.get("charts", [])
    if not isinstance(names, list):
        return []
    directories = []
    if task.source:
        directories.append(task.source.parent / "raw" / "artifacts")
    declared = _mapping(task.data.get("artifacts")).get("directory")
    if isinstance(declared, str):
        directory = Path(declared)
        if not directory.is_absolute() and task.source:
            directory = task.source.parent / directory
        directories.append(directory)
    charts = []
    for name in dict.fromkeys(n for n in names if isinstance(n, str)):
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


def build_hardware_figure(task: Task):
    """Build one figure, without querying hardware or changing the result."""
    environment = _mapping(task.data.get("environment"))
    monitoring = _mapping(environment.get("device_monitoring"))
    hardware = _mapping(environment.get("hardware_environment"))
    if not hardware:
        hardware = _mapping(monitoring.get("hardware_environment"))
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

    fonts = _fonts()
    cjk = len(fonts) > 1
    sections = []
    for cn, en, data in (
        ("硬件环境", "Hardware environment", hardware),
        ("监控摘要", "Monitoring summary", summary),
    ):
        rows = _table_rows(_facts(data, cjk), cjk)
        if rows:
            sections.append((cn if cjk else en, rows))
    curves = _curves(monitoring)
    saved = [] if curves else _saved_charts(task, monitoring)
    if not sections and not curves and not saved:
        return None

    heights = [
        0.4
        + sum(max(cell.count("\n") + 1 for cell in row) * 0.24 for row in rows)
        for _, rows in sections
    ]
    count = len(curves) or len(saved)
    heights.extend([3.2] * ((count + 1) // 2))
    with matplotlib.rc_context({"font.family": fonts, "font.size": 9}):
        fig = Figure(figsize=(14, sum(heights) + 0.85), layout="constrained")
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(len(heights), 2, height_ratios=heights)
        title = "硬件环境与监控总览" if cjk else "Hardware overview"
        name = _mapping(task.data.get("metadata")).get("display_name")
        fig.suptitle(
            _wrap(f"{title} · {name or task.name}", 125, cjk),
            fontsize=14,
        )
        for index, (title, rows) in enumerate(sections):
            ax = fig.add_subplot(grid[index, :])
            ax.axis("off")
            ax.set_title(title, loc="left", fontsize=11)
            table = ax.table(
                cellText=rows,
                cellLoc="left",
                colWidths=[0.19, 0.31, 0.19, 0.31],
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            units = [max(cell.count("\n") + 1 for cell in row) for row in rows]
            for (row, column), cell in table.get_celld().items():
                cell.set_height(units[row] / sum(units))
                cell.set_edgecolor("#dce3ec")
                cell.set_facecolor("#edf2f8" if column % 2 == 0 else "white")
                cell.PAD = 0.04
        for index, (cn, en, unit, elapsed, series) in enumerate(curves):
            ax = fig.add_subplot(grid[len(sections) + index // 2, index % 2])
            for label, values in series.items():
                ax.plot(elapsed, values, label=label, linewidth=1.4)
            ax.set_title(cn if cjk else en, loc="left", fontsize=11)
            ax.set_xlabel("经过时间 (s)" if cjk else "Elapsed (s)")
            ax.set_ylabel(unit)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        for index, (name, pixels) in enumerate(saved):
            ax = fig.add_subplot(grid[len(sections) + index // 2, index % 2])
            ax.imshow(pixels)
            ax.axis("off")
            labels = {p[5]: p[2] if cjk else p[3] for p in PANELS}
            ax.set_title(labels.get(name, Path(name).stem), fontsize=11)
    return fig


def write_hardware_overview(task: Task, output: Path, index: int):
    figure = build_hardware_figure(task)
    if figure is None:
        return None
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"hardware-{index}-overview.png"
    figure.savefig(path, dpi=150)
    figure.clear()
    return path
