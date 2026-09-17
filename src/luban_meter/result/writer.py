"""Write one Run result using the versioned JSON contract."""

from pathlib import Path

from luban_meter.core.run_contracts import RunResult
from luban_meter.utils.json_io import write_json_atomic


class ResultWriter:
    def write(self, path: Path, result: RunResult) -> None:
        write_json_atomic(path, result)
