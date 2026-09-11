"""Synthetic raw records through every bundled processor and v2 report."""

from __future__ import annotations

import csv
import json
import re

import pytest

from luban_meter.benchmark.generate.common.prometheus import (
    parse_prometheus_text,
)
from luban_meter.core.models import RawRunArtifacts, RunRequest
from luban_meter.core.registry import BenchmarkRegistry
from luban_meter.reporting.data import load_report, lookup
from luban_meter.reporting.render import write_report
from luban_meter.reporting.tables import tables_for
from luban_meter.result.manager import ResultManager
from luban_meter.result.schema import RESULT_SCHEMA

BENCHMARKS = [
    ("generate", "serving-online", 4),
    ("generate", "vllm-engine-offline", 6),
    ("generate", "vllm_metrics", 0),
    ("inference", "ceval", 2),
    ("inference", "cmmlu", 2),
    ("inference", "gsm8k", 1),
    ("inference", "humaneval", 1),
]


def online_case(rate):
    return {
        "input_length": 10,
        "output_length": 3,
        "request_rate": rate,
        "benchmark_duration_seconds": 2 / rate,
        "maximum_request_concurrency": 2,
        "peak_concurrent_requests": 2,
        "requests": [
            {
                "status": "success",
                "start_offset_ms": index * 100.0,
                "dispatch_delay_ms": 0.1,
                "duration_ms": 100.0,
                "ttft_ms": 10.0 + rate,
                "e2el_ms": 100.0,
                "itl_samples_ms": [40.0, 45.0],
                "input_tokens": 10,
                "output_tokens": 3,
            }
            for index in range(2)
        ],
    }


def engine_case(batch_size, output_tokens):
    return {
        "input_length": 4,
        "output_length": output_tokens,
        "request_batch_size": batch_size,
        "rounds": [
            {
                "round_index": 0,
                "requests": [
                    {
                        "request_index": index,
                        "round_index": 0,
                        "input_length": 4,
                        "output_length": output_tokens,
                        "request_batch_size": batch_size,
                        "actual_prompt_tokens": 4,
                        "actual_output_tokens": output_tokens,
                        "internal_ttft_seconds": 0.1 + index * 0.01,
                        "scheduled_ts": 10.0,
                        "first_token_ts": 10.1 + index * 0.01,
                        "last_token_ts": (10.1 if output_tokens == 1 else 10.3)
                        + index * 0.01,
                    }
                    for index in range(batch_size)
                ],
            }
        ],
    }


def metrics_snapshot(index):
    lines = []
    for name, value in (
        ("num_requests_running", 2),
        ("num_requests_waiting", 1),
        ("kv_cache_usage_perc", 0.5),
    ):
        lines += [
            f"# TYPE vllm:{name} gauge",
            f"vllm:{name} {value}",
        ]
    for name, value in (
        ("request_success_total", index * 10),
        ("prompt_tokens_total", index * 100),
        ("generation_tokens_total", index * 30),
    ):
        lines += [
            f"# TYPE vllm:{name} counter",
            f"vllm:{name} {value}",
        ]
    for name in (
        "time_to_first_token_seconds",
        "request_time_per_output_token_seconds",
    ):
        lines += [
            f"# TYPE vllm:{name} histogram",
            f'vllm:{name}_bucket{{le="0.1"}} 8',
            f'vllm:{name}_bucket{{le="0.2"}} 10',
            f'vllm:{name}_bucket{{le="+Inf"}} 10',
            f"vllm:{name}_count 10",
            f"vllm:{name}_sum 0.8",
        ]
    return {"metrics": parse_prometheus_text("\n".join(lines)), "error": None}


def raw_fixture(benchmark):
    raw = {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "environment": {},
        "metadata": {"source": "synthetic report test"},
    }
    if benchmark == "serving-online":
        metrics = {"cases": [online_case(rate) for rate in (1, 4, 8)]}
    elif benchmark == "vllm-engine-offline":
        metrics = {
            "cases": [
                engine_case(size, length)
                for length in (1, 3)
                for size in (1, 4, 8)
            ]
        }
        raw["environment"]["kv_cache"] = {
            "num_gpu_blocks": 10,
            "block_size": 16,
            "kv_cache_size_tokens": 160,
            "kv_cache_max_concurrency": 8.0,
        }
    elif benchmark == "vllm_metrics":
        metrics = {"snapshots": [metrics_snapshot(i) for i in (0, 1)]}
        raw["metadata"]["collect_duration"] = 10
    elif benchmark == "humaneval":
        metrics = {
            "samples": [
                {
                    "task_id": "example",
                    "sample_id": 0,
                    "status": "success",
                    "execution_status": "passed",
                    "passed": True,
                }
            ]
        }
    else:
        answer = 7.0 if benchmark == "gsm8k" else "B"
        metrics = {
            "samples": [
                {
                    "subject": "数学",
                    "status": "success",
                    "prediction": answer,
                    "reference": answer,
                    "correct": True,
                    "input_tokens": 10,
                    "output_tokens": 1,
                }
            ]
        }
    raw["metrics"] = metrics
    return raw


def processed_fixture(root, module, benchmark):
    run_id = f"synthetic-{benchmark}"
    raw_dir = root / run_id / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "raw_result.json"
    raw_path.write_text(json.dumps(raw_fixture(benchmark)), encoding="utf-8")
    config = raw_dir.parent / "config.yaml"
    config.write_text("example: true\n", encoding="utf-8")
    request = RunRequest(
        run_id=run_id,
        module=module,
        benchmark=benchmark,
        config=config,
        model_name="synthetic-model",
        model_path=None,
        output_dir=root,
        display_name=f"{benchmark} 合成数据示例",
    )
    manager = ResultManager()
    result = manager.process(
        BenchmarkRegistry().resolve(request),
        RawRunArtifacts(
            raw_result=raw_path,
            stdout_log=raw_dir / "stdout.log",
            stderr_log=raw_dir / "stderr.log",
            artifact_dir=raw_dir / "artifacts",
        ),
    )
    manager.write(request, result)
    return raw_dir.parent / "result.json"


@pytest.mark.parametrize("module,benchmark,image_count", BENCHMARKS)
def test_bundled_processors_export_real_metric_paths(
    tmp_path, module, benchmark, image_count, monkeypatch
):
    # Exercise font fallback, including the overall bar without an x path.
    monkeypatch.setattr(
        "luban_meter.reporting.charts._fonts", lambda: ["DejaVu Sans"]
    )
    source = processed_fixture(tmp_path, module, benchmark)
    original = source.read_bytes()
    payload = json.loads(original)
    assert payload["schema_version"] == RESULT_SCHEMA
    assert isinstance(payload["metrics"], dict)
    report = load_report(source)
    tables = tables_for(report.tasks[0])
    assert not report.tasks[0].notes
    for table in tables:
        assert table.records
        for record in table.records:
            for column in table.columns:
                value = record
                for key in column["path"].split("/")[1:]:
                    assert key in value, column["path"]
                    value = value[key]
    path, _ = write_report(report, source.parent / "report")
    text = path.read_text()
    assert "合成数据示例" in text
    assert len(re.findall(r"!\[", text)) == image_count
    assert "suggestion" not in text
    with (path.parent / "results.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        value, unit = lookup(payload["metrics"], row["metric"])
        assert row["value"] == (str(value) if value is not None else "")
        assert row["unit"] == unit
    assert source.read_bytes() == original
