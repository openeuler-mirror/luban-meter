"""Independent Markdown and CSV output for each run or Suite."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from luban_meter.reporting.charts import plot_table
from luban_meter.reporting.evaluation_context import evaluation_condition_rows
from luban_meter.reporting.hardware import write_hardware_overview
from luban_meter.reporting.result_reader import ResultReport, iter_metric_rows
from luban_meter.reporting.tables import build_task_tables, format_table_rows


def escape_markdown_cell(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "无可展示的指标。"
    return "\n".join(
        [
            "| " + " | ".join(map(escape_markdown_cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *(
                "| " + " | ".join(map(escape_markdown_cell, row)) + " |"
                for row in rows
            ),
        ]
    )


def relative_artifact_url(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path, output).replace(os.sep, "/"), safe="/")


def _escape_csv_value(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(report: ResultReport, path: Path) -> None:
    fields = [
        "report_id",
        "name",
        "task",
        "module",
        "benchmark",
        "run_id",
        "status",
        "metric",
        "value",
        "unit",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in report.tasks:
            common = {
                "report_id": report.run_id,
                "name": report.name,
                "task": task.name,
                **{
                    key: task.result_payload.get(key, "")
                    for key in ("module", "benchmark", "run_id", "status")
                },
            }
            found = False
            for row in iter_metric_rows(
                task.result_payload.get("metrics", {})
            ):
                found = True
                writer.writerow(
                    {
                        column_name: _escape_csv_value(column_value)
                        for column_name, column_value in {
                            **common,
                            **row,
                        }.items()
                    }
                )
            if not found:
                writer.writerow(
                    {
                        column_name: _escape_csv_value(column_value)
                        for column_name, column_value in common.items()
                    }
                )


def write_report(report: ResultReport, output: Path) -> tuple[Path, str]:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {escape_markdown_cell(report.name)}",
        f"运行：`{escape_markdown_cell(report.run_id)}`；状态：**{escape_markdown_cell(report.status)}**。",
        f"[结果 JSON]({relative_artifact_url(report.source, output)}) · "
        "[完整聚合指标 CSV](results.csv)",
    ]
    console = [
        "============ LuBan-Meter Benchmark Result ============",
        f"{report.name} | {report.run_id} | {report.status}",
    ]
    if report.kind == "suite":
        lines.append(
            markdown_table(
                ["任务", "脚本", "状态"],
                [
                    [
                        task.name,
                        f"{task.result_payload.get('module', '')}/"
                        f"{task.result_payload.get('benchmark', '')}",
                        task.result_payload.get("status", ""),
                    ]
                    for task in report.tasks
                ],
            )
        )
    for index, task in enumerate(report.tasks):
        tables = build_task_tables(task)
        data = task.result_payload
        lines.extend(
            [
                f"## {escape_markdown_cell(task.name)}",
                f"脚本：`{escape_markdown_cell(data.get('module', ''))}/"
                f"{escape_markdown_cell(data.get('benchmark', ''))}`；"
                f"状态：**{escape_markdown_cell(data.get('status', ''))}**。",
            ]
        )
        console.append(f"\n[{task.name}] {data.get('status', '')}")
        if task.source and task.source.is_file():
            lines.append(
                f"[任务结果]({relative_artifact_url(task.source, output)})"
            )
        notes = list(task.notes)
        if data.get("error"):
            error = data["error"]
            notes.append(f"{error.get('type')}: {error.get('message')}")
        skipped = data.get("metadata", {}).get("skipped_cases", [])
        if skipped:
            notes.append(f"跳过 Case：{skipped}")
        lines.extend(escape_markdown_cell(note) for note in notes)
        console.extend(notes)
        hardware = write_hardware_overview(task, output / "figures", index)
        if hardware:
            lines.extend(
                [
                    "### 硬件环境与监控",
                    "![硬件环境与监控总览]"
                    f"({relative_artifact_url(hardware, output)})",
                ]
            )
        context = evaluation_condition_rows(data)
        if context:
            pairs = [
                context[start]
                + (
                    context[start + 1]
                    if start + 1 < len(context)
                    else ["", ""]
                )
                for start in range(0, len(context), 2)
            ]
            rendered = markdown_table(["条件", "记录", "条件", "记录"], pairs)
            lines.extend(["### 评测条件", rendered])
            console.extend(["评测条件", rendered])
        for table_index, table in enumerate(tables):
            rendered = markdown_table(
                [col["label"] for col in table.columns],
                format_table_rows(table),
            )
            lines.extend(
                [f"### {escape_markdown_cell(table.title)}", rendered]
            )
            console.append(rendered)
            images = plot_table(
                table, output / "figures", f"{index}-{table_index}"
            )
            for title, path in images:
                image_title = (
                    escape_markdown_cell(title)
                    .replace("[", "")
                    .replace("]", "")
                )
                lines.extend(
                    [
                        f"![{image_title}]"
                        f"({relative_artifact_url(path, output)})",
                        escape_markdown_cell(title),
                    ]
                )
    # Write complete files before replacing previous report artifacts.
    file_descriptor, temporary = tempfile.mkstemp(dir=output, suffix=".csv")
    os.close(file_descriptor)
    try:
        _write_csv(report, Path(temporary))
        os.replace(temporary, output / "results.csv")
    finally:
        Path(temporary).unlink(missing_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(dir=output, suffix=".md")
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n\n".join(lines) + "\n")
        os.replace(temporary, output / "report.md")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output / "report.md", "\n".join(console)
