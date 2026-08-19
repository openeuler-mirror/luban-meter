import json
import tempfile
import unittest
from pathlib import Path

from luban_meter.core.engine import CoreEngine
from luban_meter.core.errors import ConfigurationError
from luban_meter.core.registry import BenchmarkRegistry
from luban_meter.suite.loader import SuiteLoader
from luban_meter.suite.models import SuiteRequest
from luban_meter.suite.runner import SuiteRunner

BENCHMARK_SOURCE = """\
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.request, encoding="utf-8") as stream:
    payload = json.load(stream)
with open(args.output, "w", encoding="utf-8") as stream:
    json.dump({
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metrics": {"value": payload["parameters"]["value"]},
        "metadata": {},
        "artifacts": {},
    }, stream)
"""

RESULT_SOURCE = """\
def process(raw_result):
    return {"metrics": raw_result["metrics"]}
"""


class SuiteTest(unittest.TestCase):
    def test_loads_and_runs_suite_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dir = root / "benchmark"
            suites_dir = root / "suite" / "definitions"
            configs_dir = suites_dir / "configs"
            configs_dir.mkdir(parents=True)

            for module, benchmark, value in (
                ("generate", "ttft", 10),
                ("inference", "accuracy", 20),
            ):
                tool_dir = benchmark_dir / module / benchmark
                tool_dir.mkdir(parents=True)
                (tool_dir / "benchmark.py").write_text(
                    BENCHMARK_SOURCE,
                    encoding="utf-8",
                )
                (tool_dir / "result.py").write_text(
                    RESULT_SOURCE,
                    encoding="utf-8",
                )
                (configs_dir / f"{benchmark}.yaml").write_text(
                    f"value: {value}\n",
                    encoding="utf-8",
                )

            (suites_dir / "basic.yaml").write_text(
                "name: basic\n"
                "tasks:\n"
                "  - name: ttft\n"
                "    module: generate\n"
                "    benchmark: ttft\n"
                "    config: configs/ttft.yaml\n"
                "  - name: accuracy\n"
                "    module: inference\n"
                "    benchmark: accuracy\n"
                "    config: configs/accuracy.yaml\n",
                encoding="utf-8",
            )

            registry = BenchmarkRegistry(benchmark_dir)
            definition = SuiteLoader(suites_dir).load("basic")
            request = SuiteRequest(
                suite_id="basic-test",
                suite="basic",
                model_path=None,
                model_name=None,
                output_dir=root / "runs",
            )

            result = SuiteRunner(CoreEngine(registry)).run(request, definition)

            self.assertEqual(result.status, "success")
            self.assertFalse(hasattr(result, "vendor"))
            self.assertEqual(
                [task.name for task in result.tasks], ["ttft", "accuracy"]
            )
            self.assertTrue(all(task.status == "success" for task in result.tasks))
            for task in result.tasks:
                task_result = json.loads(Path(task.result).read_text(encoding="utf-8"))
                self.assertEqual(task_result["status"], "success")
            suite_result = root / "runs" / request.suite_id / "suite_result.json"
            self.assertTrue(suite_result.is_file())

    def test_rejects_duplicate_task_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suites_dir = Path(directory) / "suite" / "definitions"
            suites_dir.mkdir(parents=True)
            (suites_dir / "bad.yaml").write_text(
                "tasks:\n"
                "  - name: repeated\n"
                "    module: generate\n"
                "    benchmark: ttft\n"
                "    config: ttft.yaml\n"
                "  - name: repeated\n"
                "    module: generate\n"
                "    benchmark: throughput\n"
                "    config: throughput.yaml\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigurationError, "must be unique"):
                SuiteLoader(suites_dir).load("bad")

    def test_rejects_non_string_suite_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suites_dir = Path(directory) / "suite" / "definitions"
            suites_dir.mkdir(parents=True)
            (suites_dir / "bad.yaml").write_text(
                "name: 123\n"
                "tasks:\n"
                "  - name: ttft\n"
                "    module: generate\n"
                "    benchmark: ttft\n"
                "    config: ttft.yaml\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigurationError, "non-empty string"):
                SuiteLoader(suites_dir).load("bad")

    def test_fail_fast_marks_remaining_tasks_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_dir = root / "benchmark"
            suites_dir = root / "suite" / "definitions"
            suites_dir.mkdir(parents=True)
            (suites_dir / "fail-fast.yaml").write_text(
                "tasks:\n"
                "  - name: missing\n"
                "    module: generate\n"
                "    benchmark: missing\n"
                "    config: missing.yaml\n"
                "  - name: later\n"
                "    module: generate\n"
                "    benchmark: later\n"
                "    config: later.yaml\n",
                encoding="utf-8",
            )

            registry = BenchmarkRegistry(benchmark_dir)
            definition = SuiteLoader(suites_dir).load("fail-fast")
            request = SuiteRequest(
                suite_id="fail-fast-test",
                suite="fail-fast",
                model_path=None,
                model_name=None,
                output_dir=root / "runs",
                fail_fast=True,
            )

            result = SuiteRunner(CoreEngine(registry)).run(request, definition)

            self.assertEqual(result.status, "failed")
            self.assertEqual(
                [task.status for task in result.tasks],
                ["failed", "skipped"],
            )
            self.assertIsNotNone(result.tasks[0].result)
            self.assertIsNone(result.tasks[1].result)


if __name__ == "__main__":
    unittest.main()
