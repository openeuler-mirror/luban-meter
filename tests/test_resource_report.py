"""Saved resource facts are displayed offline without changing collection."""

import copy
import json
import sys
from pathlib import Path

from luban_meter.reporting.html import report_data, write_html


def resource_fixture():
    start = 1790000000
    snapshots = [
        {
            "elapsed": elapsed,
            "devices": [{
                "index": 0, "utilization": value,
                "memory_used": 2048, "power": 0, "temperature": 38,
            }, {
                "index": 1, "utilization": 70,
                "memory_used": None, "power": 110, "temperature": None,
            }],
            "cpu": {"utilization": value, "memory_used": 4096},
        }
        for elapsed, value in [(0, 0), (1, 20), (6, 80)]
    ]
    case = {
        "id": "case_1", "status": "success", "parameters": {
            "model": "fixture-model",
        },
        "window": {"started_at": start, "ended_at": start + 10},
        "device_observation": {
            "sample_count": 3, "error_count": 0, "interval_seconds": 1,
            "devices": [{
                "index": 0, "name": "Synthetic GPU",
                "utilization_avg": 33.33, "memory_used_avg_mb": 2048,
                "memory_total_mb": None, "power_avg_watts": 0,
                "temperature_avg_celsius": 38,
            }, {"index": 1, "name": "Synthetic GPU"}],
            "cpu": {"utilization_avg": 33.33}, "timeseries": snapshots,
        },
        "observation": {
            "valid_samples": 2, "failed_samples": 0,
            "snapshots": [{
                "finished_at": start + i,
                "metrics": {"vllm:num_requests_running": {
                    "type": "gauge", "samples": [{
                        "labels": {"model_name": "fixture-model"},
                        "value": i,
                    }],
                }},
            } for i in (0, 1)],
        },
    }
    task = {
        "kind": "task", "name": "duplicate", "scenario": "online_serving",
        "category": "generation_performance", "status": "success",
        "cases": [case, {"id": "empty", "status": "failed"}],
    }
    other = copy.deepcopy(task)
    other["cases"] = [copy.deepcopy(case)]
    c = other["cases"][0]
    c["window"] = {"started_at": start + 20, "ended_at": start + 21}
    c["device_observation"].update(
        devices=[{"index": 0, "utilization_avg": 0}],
        cpu={"utilization_avg": 0, "memory_used_avg_mb": 0},
        timeseries=[{"elapsed": 0, "devices": [{
            "index": 0, "utilization": 0, "memory_used": -1,
            "power": None, "temperature": None,
        }]}],
    )
    c.pop("observation")
    return {
        "kind": "run", "id": "synthetic-resource-run", "status": "success",
        "started_at": start - 5, "finished_at": start + 25,
        "tasks": [task, other],
    }


def test_cpu_samples_use_existing_observation_contract_without_mutation():
    payload = resource_fixture()
    before = copy.deepcopy(payload)
    data = report_data(payload)
    assert payload == before
    assert set(data) == {"result", "rows", "observations"}
    samples = [p for p in data["observations"]
               if p["metric"] == "cpu/utilization"]
    assert [p["value"] for p in samples] == [0, 20, 80]
    assert [p["time"] for p in samples] == [
        payload["tasks"][0]["cases"][0]["window"]["started_at"] + elapsed
        for elapsed in (0, 1, 6)
    ]
    assert all(p["task_index"] == 0 and p["case_index"] == 0
               and p["labels"] == {} for p in samples)
    gpu = [p for p in data["observations"]
           if p["metric"] == "device/utilization"]
    assert {(p["task_index"], p["case_index"]) for p in gpu} == {
        (0, 0), (1, 0)
    }
    rendered = data["result"]["tasks"][0]["cases"][0]
    assert "timeseries" not in rendered["device_observation"]
    assert "snapshots" not in rendered["observation"]
    assert rendered["device_observation"]["devices"] == (
        before["tasks"][0]["cases"][0]["device_observation"]["devices"]
    )


def test_missing_case_clock_does_not_fabricate_epoch_timestamps():
    payload = resource_fixture()
    payload["tasks"][0]["cases"][0].pop("window")
    observations = report_data(payload)["observations"]
    assert not any(p["task_index"] == 0 and
                   p["metric"].startswith(("device/", "cpu/"))
                   for p in observations)
    assert any(p["metric"] == "vllm:num_requests_running"
               for p in observations)


def test_resource_script_is_embedded_for_offline_reports(tmp_path):
    html = write_html(resource_fixture(), tmp_path / "resources.html")
    text = html.read_text("utf-8")
    assert "__RESOURCE_SCRIPT__" not in text
    assert "function initResources()" in text
    assert 'id="resource-case"' in text
    assert '<script src=' not in text
    raw = text.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(raw)["result"]["id"] == "synthetic-resource-run"


def test_cpu_and_gpu_charts_share_the_existing_sampling_budget():
    payload = resource_fixture()
    device = payload["tasks"][0]["cases"][0]["device_observation"]
    sample = device["timeseries"][0]
    device["timeseries"] = [
        {**sample, "elapsed": i} for i in range(1501)
    ]
    device["sample_count"] = 1501
    points = report_data(payload)["observations"]
    for metric in ("device/utilization", "cpu/utilization"):
        times = {p["time"] for p in points
                 if p["task_index"] == 0 and p["metric"] == metric}
        assert len(times) == 376
        assert len(times) <= 500
        assert sorted(times)[1] - min(times) == 4
    assert len(device["timeseries"]) == 1501


if __name__ == "__main__":
    write_html(resource_fixture(), Path(sys.argv[1]))
