"""Exercise stable file protocols across the internal naming migration."""

import json
from pathlib import Path

import pytest

from luban_meter.core.benchmark_registry import BenchmarkRegistry
from luban_meter.core.run_contracts import RunRequest
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.suite.suite_contracts import SuiteTask, SuiteTaskResult
from luban_meter.utils.json_io import to_jsonable


@pytest.mark.parametrize(
    "category_directory,collector_name,processor_name",
    [
        ("generation_performance", "collect_raw.py", "calculate_metrics.py"),
        ("generate", "benchmark.py", "result.py"),
    ],
)
def test_collector_protocol_survives_internal_renames(
    tmp_path, category_directory, collector_name, processor_name
):
    benchmark_root = tmp_path / "benchmarking"
    benchmark_directory = benchmark_root / category_directory / "smoke"
    benchmark_directory.mkdir(parents=True)
    (benchmark_directory / collector_name).write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--request')\n"
        "parser.add_argument('--output')\n"
        "args = parser.parse_args()\n"
        "request = json.loads(Path(args.request).read_text())['request']\n"
        "assert request['module'] == 'generate'\n"
        "assert request['benchmark'] == 'smoke'\n"
        "assert request['model_name'] == 'served-model'\n"
        "assert 'benchmark_name' not in request\n"
        "raw = {'schema_version': 'luban-meter.raw/v1', "
        "'status': 'success', 'metrics': {'count': 3}}\n"
        "Path(args.output).write_text(json.dumps(raw))\n",
        encoding="utf-8",
    )
    (benchmark_directory / processor_name).write_text(
        "def process(raw_result):\n"
        "    return {'metrics': raw_result['metrics']}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "parameters.yaml"
    config_path.write_text("{}", encoding="utf-8")
    request = RunRequest(
        run_id="naming-smoke",
        category_name="generate",
        benchmark_name="smoke",
        config_path=config_path,
        model_path=None,
        model_name="served-model",
        output_dir=tmp_path / "runs",
    )
    registry = BenchmarkRegistry(benchmark_root)
    assert registry.list_benchmarks("generate") == ("smoke",)
    result = RunCoordinator(registry).run(request)
    assert result.status == "success"
    saved = json.loads(
        (request.output_dir / request.run_id / "result.json").read_text()
    )
    assert saved["metrics"] == {"count": 3}
    assert saved["module"] == "generate"
    assert saved["benchmark"] == "smoke"
    assert saved["model"] == {"name": "served-model", "path": None}
    assert not {"category_name", "benchmark_name", "model_info"} & saved.keys()


def test_nested_suite_contracts_keep_serialized_field_names():
    task = SuiteTask(
        "task", "model_service_quality", "ceval", Path("ceval.yaml")
    )
    result = SuiteTaskResult(
        "task", "model_service_quality", "ceval", "skipped"
    )
    payload = to_jsonable({"task": task, "result": result})
    assert payload["task"] == {
        "name": "task",
        "module": "model_service_quality",
        "benchmark": "ceval",
        "config": "ceval.yaml",
        "timeout": None,
    }
    assert payload["result"]["benchmark"] == "ceval"
    assert payload["result"]["module"] == "model_service_quality"
    assert payload["result"]["output"] is None


@pytest.mark.parametrize(
    "benchmark_name,directory_name",
    [
        ("serving-online", "online_serving"),
        ("vllm-engine-offline", "offline_vllm_engine"),
        ("vllm_metrics", "vllm_service_metrics"),
    ],
)
def test_public_benchmark_names_resolve_to_python_directories(
    tmp_path, benchmark_name, directory_name
):
    config_path = tmp_path / "parameters.yaml"
    config_path.write_text("{}", encoding="utf-8")
    request = RunRequest(
        "resolution",
        "generate",
        benchmark_name,
        config_path,
        None,
        None,
        tmp_path,
    )
    resolved = BenchmarkRegistry().resolve(request)
    definition = resolved.benchmark_definition
    assert definition.benchmark_name == benchmark_name
    assert definition.collector_path.parent.name == directory_name
    assert definition.collector_path.name == "collect_raw.py"
    assert definition.processor_path.name == "calculate_metrics.py"
