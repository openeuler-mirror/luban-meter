"""Device metadata from DCGM responses reaches hardware reports."""

import pytest

from luban_meter.benchmarking.generation_performance.common import (
    device_monitor,
)
from luban_meter.log import configure_logging
from luban_meter.runtime.device_monitor import DeviceMonitorDaemon

DCGM_METRICS = "\n".join(
    [
        'DCGM_FI_DEV_GPU_UTIL{gpu="0",modelName="NVIDIA H20"} 0',
        'DCGM_FI_DEV_FB_USED{gpu="0",modelName="NVIDIA H20"} 0',
        'DCGM_FI_DEV_FB_FREE{gpu="0",modelName="NVIDIA H20"} 97356',
        'DCGM_FI_DEV_GPU_UTIL{gpu="3",modelName="NVIDIA A100"} 45',
    ]
)


@pytest.mark.parametrize(
    ("metrics", "expected_name"),
    [
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
    ],
)
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
        "NVIDIA H20",
        "NVIDIA A100",
    ]
    assert snapshot.error is None
    assert snapshot.devices[0].memory_used_mb == 0
    # FREE + USED must not be treated as total memory without FB_TOTAL.
    assert snapshot.devices[0].memory_total_mb is None


@pytest.mark.parametrize("metrics", [DCGM_METRICS, ""])
def test_hardware_detection_messages_use_logs(tmp_path, capsys, metrics):
    devices = device_monitor.detect_devices(metrics)
    with configure_logging(tmp_path) as path:
        device_monitor.print_hardware_info(devices)
    captured = capsys.readouterr()
    assert captured.out == ""
    text = path.read_text()
    assert text == captured.err
    assert (" | WARNING | " in text) is not bool(devices)
    for device in devices:
        assert device.name in text


def test_missing_monitor_devices_emit_one_warning(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(device_monitor, "detect_devices", lambda **kw: [])
    with configure_logging(tmp_path) as path:
        daemon = DeviceMonitorDaemon("http://exporter.invalid")
        daemon.start()
    captured = capsys.readouterr()
    assert captured.out == ""
    text = path.read_text()
    assert text == captured.err
    assert text.count(" | WARNING | ") == 1
    assert daemon.summary() is None
