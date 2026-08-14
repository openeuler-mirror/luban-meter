import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from luban_meter.cli import main
from luban_meter.core.models import BenchmarkResult


class CliTest(unittest.TestCase):
    def test_model_arguments_are_in_request(self) -> None:
        output = io.StringIO()
        result = BenchmarkResult(
            schema_version="luban-meter.result/v1",
            run_id="test-run",
            status="success",
            module="generate",
            vendor="ascend",
            benchmark="ttft",
            config="configs/benchmarks/ascend-ttft.yaml",
        )
        with patch("luban_meter.cli.CoreEngine") as engine_type:
            engine_type.return_value.run.return_value = result
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "run",
                        "--module",
                        "generate",
                        "--vendor",
                        "ascend",
                        "--benchmark",
                        "ttft",
                        "--config",
                        "configs/benchmarks/ascend-ttft.yaml",
                        "--model-path",
                        "/models/Qwen3-8B",
                        "--model-name",
                        "Qwen3-8B",
                    ]
                )

        request = engine_type.return_value.run.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(request.model_path.as_posix(), "/models/Qwen3-8B")
        self.assertEqual(request.model_name, "Qwen3-8B")
        self.assertEqual(request.vendor, "ascend")
        self.assertEqual(request.benchmark, "ttft")
        self.assertEqual(
            request.config.as_posix(),
            "configs/benchmarks/ascend-ttft.yaml",
        )


if __name__ == "__main__":
    unittest.main()
