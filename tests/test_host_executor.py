import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from luban_meter.core.models import BenchmarkSpec, CommandSpec, ResolvedRun, RunRequest
from luban_meter.execution.command import LocalCommandRunner
from luban_meter.execution.host import HostSession


class HostExecutorTest(unittest.TestCase):
    def test_runs_command(self) -> None:
        result = LocalCommandRunner().run(
            CommandSpec(argv=[sys.executable, "-c", "print('ok')"])
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_benchmark_inherits_current_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark.py"
            benchmark.write_text(
                "import argparse, json, os\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--request')\n"
                "parser.add_argument('--output')\n"
                "args = parser.parse_args()\n"
                "with open(args.output, 'w', encoding='utf-8') as stream:\n"
                "    json.dump({\n"
                "        'schema_version': 'luban-meter.raw/v1',\n"
                "        'status': 'success',\n"
                "        'metrics': {},\n"
                "        'metadata': {\n"
                "            'inherited': os.environ['LUBAN_METER_TEST_ENV']\n"
                "        },\n"
                "        'artifacts': {},\n"
                "    }, stream)\n",
                encoding="utf-8",
            )
            request = RunRequest(
                run_id="environment-inheritance-test",
                module="generate",
                vendor="ascend",
                benchmark="ttft",
                config=root / "config.yaml",
                model_path=None,
                model_name=None,
                output_dir=root / "runs",
            )
            run = ResolvedRun(
                request=request,
                benchmark=BenchmarkSpec(
                    vendor="ascend",
                    module="generate",
                    benchmark="ttft",
                    benchmark_entry=benchmark,
                    result_handler=root / "result.py",
                ),
                parameters={},
            )

            with patch.dict(os.environ, {"LUBAN_METER_TEST_ENV": "exported"}):
                artifacts = HostSession().execute(run)

            raw = json.loads(artifacts.raw_result.read_text(encoding="utf-8"))
            self.assertEqual(raw["metadata"]["inherited"], "exported")
            self.assertFalse(
                (request.output_dir / request.run_id / "environment.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
