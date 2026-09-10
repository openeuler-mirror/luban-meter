"""Hardware overview semantics, missing samples, and portable reports."""

import copy
import json
import math
import re

from luban_meter.reporting.data import from_payload, load_report
from luban_meter.reporting.hardware import build_hardware_figure
from luban_meter.reporting.render import write_report
from tests.test_reports_v2 import result, save, suite


def table_text(figure):
    return "\n".join(
        cell.get_text().get_text()
        for axes in figure.axes
        for table in axes.tables
        for cell in table.get_celld().values()
    )


def test_environment_only_preserves_all_fields_and_long_names(tmp_path):
    payload = result({"score": 0})
    payload["environment"]["hardware_environment"] = {
        "cpu": {"model": "Example CPU", "cores": 64, "sockets": 2},
        "memory_total_gb": 512,
        "driver_version": "example-driver",
        "devices": [{"index": 3, "name": "Example accelerator"}],
        "extra_field": "extra-" * 35,
    }
    original = copy.deepcopy(payload)
    report = from_payload(payload, tmp_path / "result.json")
    figure = build_hardware_figure(report.tasks[0])
    text = table_text(figure)
    assert "Example CPU" in text
    assert "Example accelerator" in text
    assert "example-driver" in text
    assert "extra-" * 35 in text.replace("\n", "")
    assert "512" in text
    assert not any(axes.lines for axes in figure.axes)
    assert payload == original
    figure.clear()


def test_per_device_timestamps_keep_missing_samples_and_zero(tmp_path):
    payload = result()
    payload["environment"]["device_monitoring"] = {
        "timeseries": [
            {"elapsed": 2, "devices": [{"index": 0, "utilization": 50}]},
            {"elapsed": 0, "devices": [{"index": 0, "utilization": 0}]},
            {"elapsed": 1, "devices": [{"index": 1, "utilization": 70}]},
            {"elapsed": None, "devices": [{"index": 0, "utilization": 90}]},
        ]
    }
    task = from_payload(payload, tmp_path / "result.json").tasks[0]
    figure = build_hardware_figure(task)
    curves = {
        line.get_label(): line for axes in figure.axes for line in axes.lines
    }
    assert set(curves) == {"Device 0", "Device 1"}
    assert list(curves["Device 0"].get_xdata()) == [0, 1, 2]
    first = curves["Device 0"].get_ydata()
    second = curves["Device 1"].get_ydata()
    assert first[0] == 0 and math.isnan(first[1]) and first[2] == 50
    assert math.isnan(second[0]) and second[1] == 70 and math.isnan(second[2])
    figure.clear()


def test_cpu_only_and_summary_without_timeseries(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "luban_meter.reporting.hardware._fonts", lambda: ["DejaVu Sans"]
    )
    payload = result()
    monitoring = {
        "cpu": {"utilization_avg": 0, "memory_total_mb": 65536},
        "sample_count": 2,
        "timeseries": [
            {"elapsed": 0, "cpu": {"utilization": 0, "memory_used": 2048}},
            {"elapsed": 1, "cpu": {"utilization": 40, "memory_used": 4096}},
        ],
    }
    payload["environment"]["device_monitoring"] = monitoring
    task = from_payload(payload, tmp_path / "result.json").tasks[0]
    figure = build_hardware_figure(task)
    axes = [axes for axes in figure.axes if axes.lines]
    assert len(axes) == 2
    assert [axes.get_ylabel() for axes in axes] == ["%", "MB"]
    assert list(axes[1].lines[0].get_ydata()) == [2048, 4096]
    assert "65536" in table_text(figure)
    figure.clear()
    del monitoring["timeseries"]
    figure = build_hardware_figure(task)
    assert "65536" in table_text(figure)
    assert not any(axes.lines for axes in figure.axes)
    figure.clear()


def test_suite_json_alone_generates_one_overview_per_task(tmp_path):
    outputs = []
    for index in (0, 1):
        data = result({"score": index}, run_id=f"task-{index}")
        data["environment"] = {
            "hardware_environment": {"cpu": {"model": f"CPU-{index}"}},
            "device_monitoring": {
                "charts": ["missing.png"],
                "timeseries": [
                    {"elapsed": 0, "cpu": {"utilization": index * 10}}
                ],
            },
        }
        outputs.append(data)
    source = save(tmp_path / "suite_result.json", suite(outputs))
    original = source.read_bytes()
    path, _ = write_report(load_report(source), tmp_path / "report")
    content = path.read_text()
    assert content.count("### 硬件环境与监控") == 2
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)
    assert len(images) == 2 and len(set(images)) == 2
    assert all((path.parent / name).is_file() for name in images)
    for section in content.split("## task-")[1:]:
        assert section.index("### 硬件环境与监控") < section.index(
            "### 指标摘要"
        )
    assert json.loads(source.read_bytes()) == json.loads(original)


def test_missing_and_malformed_monitoring_does_not_invent_hardware(tmp_path):
    payload = result()
    for monitoring in (None, [], {"timeseries": [None, {}], "charts": []}):
        payload["environment"]["device_monitoring"] = monitoring
        task = from_payload(payload, tmp_path / "result.json").tasks[0]
        assert build_hardware_figure(task) is None
