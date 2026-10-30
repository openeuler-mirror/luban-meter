"""Synthetic counter/lifecycle tests; never used as report example data."""

import copy
import time

import httpx
import pytest

from luban_meter.core.benchmark import GenerationPerformanceBenchmark
from luban_meter.core.parameters import validate_parameters
from luban_meter.core.planning import bootstrap, schema_for
from luban_meter.reporting.html import report_data
from luban_meter.runtime import evaluation_session
from luban_meter.runtime.host_monitor import (
    HostMonitorDaemon,
    derive_rates,
    parse_host,
    summarize,
)


def exposition(step):
    lines = [
        f'node_cpu_seconds_total{{cpu="0",mode="idle"}} {100 + step * 8}',
        f'node_cpu_seconds_total{{cpu="0",mode="user"}} {10 + step * 2}',
        'node_memory_MemTotal_bytes 1073741824',
        'node_memory_MemAvailable_bytes 805306368',
        'node_uname_info{nodename="synthetic-only"} 1',
        f'node_disk_read_bytes_total{{device="vdb"}} {step * 1048576}',
        f'node_disk_written_bytes_total{{device="vdb"}} {step * 2097152}',
        f'node_disk_reads_completed_total{{device="vdb"}} {step * 10}',
        f'node_disk_writes_completed_total{{device="vdb"}} {step * 20}',
        f'node_disk_io_time_seconds_total{{device="vdb"}} {step / 2}',
        f'node_network_receive_bytes_total{{device="lo"}} {step * 100}',
        f'node_network_transmit_bytes_total{{device="lo"}} {step * 100}',
        'node_network_receive_errs_total{device="lo"} 0',
        'node_network_transmit_errs_total{device="lo"} 0',
    ]
    for key, value in [("size", 1000), ("free", 400), ("avail", 300)]:
        lines.append('node_filesystem_' + key + '_bytes{device="/dev/vdb",'
                     'mountpoint="/data",fstype="ext4"} ' + str(value))
    return "\n".join(lines)


def snapshot(step, previous=None):
    s = parse_host(exposition(step))
    s.update(started_at=step - 0.01, finished_at=step,
             interval_seconds=1 if previous else None,
             previous_finished_at=previous["finished_at"] if previous
             else None)
    derive_rates(s, previous, s["interval_seconds"])
    return s


def test_rates_units_zero_and_first_baseline():
    first = snapshot(0)
    assert first["cpu"]["utilization"] is None
    assert first["networks"][0]["receive_bytes_per_second"] is None
    second = snapshot(1, first)
    assert second["cpu"]["utilization"] == pytest.approx(20)
    assert second["memory"]["used_bytes"] == 268435456
    assert second["filesystems"][0]["used_bytes"] == 600
    assert second["filesystems"][0]["utilization"] == 60
    assert second["disks"][0]["read_bytes_per_second"] == 1048576
    assert second["disks"][0]["utilization"] == 50
    assert second["networks"][0]["receive_errors_per_second"] == 0
    assert second["networks"][0]["receive_dropped_per_second"] is None


def test_counter_resets_and_identity_changes_do_not_create_negative_rates():
    first = snapshot(10)
    reset = snapshot(0, first)
    assert reset["cpu"]["utilization"] is None
    assert reset["disks"][0]["read_bytes_per_second"] is None
    assert "counter_reset:vdb" in reset["rate_issues"]
    changed = parse_host(exposition(11))
    changed["raw"]["cpu_ids"] = ["1"]
    changed["raw"]["networks"][0]["device"] = "eth0"
    derive_rates(changed, first, 1)
    assert changed["cpu"]["utilization"] is None
    assert changed["networks"][0]["receive_bytes_per_second"] is None


def test_missing_gauges_stay_missing_and_partial_sources_are_accepted():
    s = parse_host('node_network_receive_bytes_total{device="lo"} 0')
    assert s["memory"]["utilization"] is None
    assert s["raw"]["cpu_seconds"] == {}
    assert s["raw"]["networks"][0]["receive_bytes"] == 0


def test_summary_uses_only_formal_intervals_and_preserves_original_samples():
    first = snapshot(0)
    second = snapshot(1, first)
    third = snapshot(2, second)
    samples = [first, second, third]
    before = copy.deepcopy(samples)
    summary = summarize(samples, 0.5, 2.5)
    assert samples == before
    assert summary["cpu"]["utilization"]["count"] == 1
    assert summary["cpu"]["utilization"]["covered_seconds"] == 1
    assert summary["memory"]["used_bytes"]["count"] == 2
    assert summary["networks"][0]["receive_bytes_per_second"]["unit"] == (
        "bytes/s"
    )


class ObservedFixture(GenerationPerformanceBenchmark):
    def samples(self, context):
        yield {}
        yield {}

    def evaluate(self, sample, context):
        time.sleep(0.32)
        return {"value": 1}


@pytest.mark.parametrize("gpu,host,failed", [
    (True, False, False), (False, True, False),
    (True, True, False), (True, True, True),
])
def test_runtime_sources_are_independent(
    monkeypatch, tmp_path, gpu, host, failed
):
    bootstrap()
    calls = []

    class GPU:
        def __init__(self, *args):
            pass

        def start(self):
            calls.append("gpu-start")

        def stop(self):
            calls.append("gpu-stop")

        def summary(self):
            return {"device_count": 1}

    count = 0

    def response(request):
        nonlocal count
        count += 1
        if failed:
            raise httpx.ConnectError("synthetic offline source")
        return httpx.Response(200, text=exposition(count))

    def monitor(url, interval):
        daemon = HostMonitorDaemon(url, interval)
        daemon.client.close()
        daemon.client = httpx.Client(transport=httpx.MockTransport(response))
        return daemon

    monkeypatch.setattr(evaluation_session, "DeviceMonitorDaemon", GPU)
    monkeypatch.setattr(evaluation_session, "HostMonitorDaemon", monitor)
    params = validate_parameters(schema_for(ObservedFixture, {}), {
        "num_requests": 2, "metrics": [],
        "device_exporter_url": "http://gpu.invalid" if gpu else "",
        "host_exporter_url": "http://host.invalid" if host else "",
        "host_sample_interval": 0.1,
    }, base_dir=tmp_path)
    (tmp_path / "raw").mkdir()
    result = evaluation_session.run_case(
        ObservedFixture, {"id": "fixture", "parameters": params},
        tmp_path, tmp_path / "stop",
    )
    assert result["status"] == "success", result
    assert bool(calls) == gpu
    if host:
        observed = result["host_observation"]
        assert observed["status"] == ("failed" if failed else "success")
        if not failed:
            assert observed["summary"]["cpu"]["utilization"]["count"] >= 1
            payload = {"kind": "task", "name": "fixture",
                       "category": "generation_performance",
                       "status": "success", "cases": [result]}
            data = report_data(payload)
            assert any(p["metric"].startswith("host/networks/")
                       for p in data["observations"])
            assert "timeseries" not in (
                data["result"]["cases"][0]["host_observation"]
            )
    else:
        assert "host_observation" not in result


def test_failed_scrape_breaks_the_rate_baseline():
    daemon = HostMonitorDaemon("http://host.invalid", 1)
    replies = iter([exposition(0), None, exposition(20), exposition(21)])

    def response(request):
        text = next(replies)
        if text is None:
            raise httpx.ReadTimeout("synthetic scrape failure")
        return httpx.Response(200, text=text)

    daemon.client.close()
    daemon.client = httpx.Client(transport=httpx.MockTransport(response))
    for _ in range(4):
        daemon.sample()
    daemon.stop()
    assert "error" in daemon.samples[1]
    assert daemon.samples[2]["cpu"]["utilization"] is None
    assert daemon.samples[3]["cpu"]["utilization"] is not None


def test_invalid_sampling_interval_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="host_sample_interval"):
        validate_parameters(schema_for(ObservedFixture, {}), {
            "host_sample_interval": 0,
        }, base_dir=tmp_path)
