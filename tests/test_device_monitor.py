"""Device metadata from DCGM responses reaches hardware reports."""

import pytest

from luban_meter.benchmark.generate.common import device_monitor
from luban_meter.reporting.data import from_payload
from luban_meter.reporting.hardware import build_hardware_figure
from tests.test_hardware_overview import table_text
from tests.test_reports_v2 import result

DCGM_METRICS = '\n'.join([
    'DCGM_FI_DEV_GPU_UTIL{gpu="0",modelName="NVIDIA H20"} 0',
    'DCGM_FI_DEV_FB_USED{gpu="0",modelName="NVIDIA H20"} 0',
    'DCGM_FI_DEV_FB_FREE{gpu="0",modelName="NVIDIA H20"} 97356',
    'DCGM_FI_DEV_GPU_UTIL{gpu="3",modelName="NVIDIA A100"} 45',
])


@pytest.mark.parametrize(("metrics", "expected_name"), [
    (
        'DCGM_FI_DEV_GPU_UTIL{gpu="3",modelName="NVIDIA H20"} 0',
        "NVIDIA H20",
    ),
    ('DCGM_FI_DEV_GPU_UTIL{gpu="3"} 0', "GPU-3"),
    ('DCGM_FI_DEV_GPU_UTIL{gpu="3",modelName=""} 0', "GPU-3"),
    ('DCGM_FI_DEV_NAME{gpu="3"} "Legacy GPU"', "Legacy GPU"),
    (
        'DCGM_FI_DEV_NAME{gpu="3",modelName="NVIDIA H20"} 1',
        "NVIDIA H20",
    ),
])
def test_detect_device_model_labels_and_fallbacks(metrics, expected_name):
    devices = device_monitor.detect_devices(metrics)

    assert [(device.index, device.name) for device in devices] == [
        (3, expected_name),
    ]


def test_each_gpu_keeps_its_own_model_name():
    devices = device_monitor.detect_devices(DCGM_METRICS)

    assert [(device.index, device.name) for device in devices] == [
        (0, "NVIDIA H20"),
        (3, "NVIDIA A100"),
    ]


def test_model_names_reach_environment_and_samples(monkeypatch):
    monkeypatch.setattr(
        device_monitor, "_fetch_metrics", lambda url: DCGM_METRICS
    )
    devices = device_monitor.detect_devices(DCGM_METRICS)
    environment = device_monitor.collect_hardware_environment(DCGM_METRICS)
    snapshot = device_monitor.sample_exporter(
        "http://exporter.invalid", devices, 0, 0.0
    )

    assert environment["devices"] == [
        {"index": 0, "name": "NVIDIA H20"},
        {"index": 3, "name": "NVIDIA A100"},
    ]
    assert [sample.name for sample in snapshot.devices] == [
        "NVIDIA H20", "NVIDIA A100",
    ]
    assert snapshot.error is None
    assert snapshot.devices[0].memory_used_mb == 0
    # FREE + USED must not be treated as total memory without FB_TOTAL.
    assert snapshot.devices[0].memory_total_mb is None


def test_exported_model_names_appear_in_hardware_report(tmp_path):
    payload = result()
    payload["environment"]["hardware_environment"] = (
        device_monitor.collect_hardware_environment(DCGM_METRICS)
    )
    task = from_payload(payload, tmp_path / "result.json").tasks[0]
    figure = build_hardware_figure(task)

    try:
        text = table_text(figure)
        assert "NVIDIA H20" in text
        assert "NVIDIA A100" in text
        assert "GPU-0" not in text
        assert "GPU-3" not in text
    finally:
        figure.clear()
