import json
import tempfile
import unittest
from pathlib import Path

from luban_meter.core.benchmark_registry import BenchmarkRegistry
from luban_meter.core.run_contracts import RunRequest
from luban_meter.core.run_coordinator import RunCoordinator


class CoreEngineTest(unittest.TestCase):
    def test_resolve_failure_still_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            request = RunRequest(
                run_id="generate-test-run",
                category_name="generate",
                benchmark_name="missing-benchmark",
                config_path=Path("missing-config.yaml"),
                model_path=None,
                model_name=None,
                output_dir=output_dir,
            )

            result = RunCoordinator(BenchmarkRegistry()).run(request)

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.metadata["failure_stage"], "resolve")
            result_path = output_dir / request.run_id / "result.json"
            self.assertTrue(result_path.is_file())
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "failed")
            self.assertNotIn("vendor", saved)
            self.assertEqual(saved["benchmark"], "missing-benchmark")
            self.assertEqual(saved["metrics"], {})
            self.assertEqual(saved["environment"], {})


if __name__ == "__main__":
    unittest.main()
