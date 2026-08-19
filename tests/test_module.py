import tempfile
import unittest
from pathlib import Path

from luban_meter.core.models import RunRequest
from luban_meter.core.registry import BenchmarkRegistry


class BenchmarkRegistryTest(unittest.TestCase):
    def test_resolves_benchmark_and_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool_dir = root / "benchmark" / "generate" / "ttft"
            tool_dir.mkdir(parents=True)
            (tool_dir / "benchmark.py").write_text("", encoding="utf-8")
            (tool_dir / "result.py").write_text("", encoding="utf-8")
            config = root / "ttft.yaml"
            config.write_text(
                "rounds: 100\nwarmup: 10\nconcurrency: 1\n",
                encoding="utf-8",
            )

            registry = BenchmarkRegistry(root / "benchmark")
            request = RunRequest(
                run_id="generate-test",
                module="generate",
                benchmark="ttft",
                config=config,
                model_path=None,
                model_name=None,
                output_dir=root / "runs",
            )

            resolved = registry.resolve(request)

            self.assertEqual(
                tuple(name for name, _ in registry.modules()),
                ("generate", "inference"),
            )
            self.assertEqual(registry.list_benchmarks("generate"), ("ttft",))
            self.assertEqual(resolved.benchmark.benchmark, "ttft")
            self.assertEqual(resolved.parameters["rounds"], 100)
            self.assertEqual(resolved.parameters["warmup"], 10)


if __name__ == "__main__":
    unittest.main()
