import tempfile
import unittest
from pathlib import Path

from luban_meter.core.run_contracts import RunRequest
from luban_meter.core.scenario_registry import ScenarioRegistry


class BenchmarkRegistryTest(unittest.TestCase):
    def test_resolves_benchmark_and_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool_dir = (
                root / "benchmarking" / "generation_performance" / "ttft"
            )
            tool_dir.mkdir(parents=True)
            (tool_dir / "collect_raw.py").write_text("", encoding="utf-8")
            (tool_dir / "calculate_metrics.py").write_text(
                "", encoding="utf-8"
            )
            config = root / "ttft.yaml"
            config.write_text(
                "rounds: 100\nwarmup: 10\nconcurrency: 1\n",
                encoding="utf-8",
            )

            registry = ScenarioRegistry(root / "benchmarking")
            request = RunRequest(
                run_id="generate-test",
                category_name="generate",
                scenario_name="ttft",
                config_path=config,
                model_path=None,
                model_name=None,
                output_dir=root / "runs",
            )

            resolved = registry.resolve(request)

            self.assertEqual(
                tuple(name for name, _ in registry.list_categories()),
                ("generate", "model_service_quality"),
            )
            self.assertEqual(registry.list_scenarios("generate"), ("ttft",))
            self.assertEqual(
                resolved.scenario_definition.scenario_name, "ttft"
            )
            self.assertEqual(resolved.parameters["rounds"], 100)
            self.assertEqual(resolved.parameters["warmup"], 10)


if __name__ == "__main__":
    unittest.main()
