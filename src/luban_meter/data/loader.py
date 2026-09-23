"""Public dataset loading with shared path, format and sampling rules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luban_meter.data.engine import resolve_data_path, select_records
from luban_meter.data.sources import sources


@dataclass(frozen=True)
class DatasetLoader:
    """Load complete local records through a registered format adapter.

    An explicit base_dir anchors relative paths without fallback. Otherwise
    paths use the working directory and the shared repository dataset root.
    Format inference only identifies file syntax; semantic formats such as
    ShareGPT and SQuAD must be selected explicitly.
    """

    path: str | Path
    format: str | None = None
    max_samples: int | None = None
    shuffle: bool = False
    seed: int = 42
    base_dir: str | Path | None = None
    skip_invalid: bool = False
    diagnostics: list | None = None

    def __post_init__(self):
        if (
            not isinstance(self.path, (str, Path))
            or not str(self.path).strip()
        ):
            raise ValueError("dataset path must not be empty")
        if self.max_samples is not None and (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")
        if not isinstance(self.skip_invalid, bool):
            raise ValueError("skip_invalid must be a boolean")
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        sources.get(self.resolved_format)

    @property
    def resolved_format(self) -> str:
        if self.format is not None:
            if not isinstance(self.format, str) or not self.format:
                raise ValueError("dataset format must be a non-empty string")
            return self.format
        suffix = Path(self.path).suffix.lower()
        formats = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}
        if suffix not in formats:
            raise ValueError(
                f"cannot infer dataset format from {suffix!r}; specify format"
            )
        return formats[suffix]

    @property
    def resolved_path(self) -> Path:
        return resolve_data_path(self.path, base_dir=self.base_dir).resolve()

    def load(self) -> list[dict[str, Any]]:
        """Read, validate, then shuffle and limit complete records."""
        path = self.resolved_path
        if not path.is_file():
            raise FileNotFoundError(f"dataset file not found: {path}")
        source = sources.get(self.resolved_format)()
        if self.skip_invalid:
            reader = getattr(source, "read_validated", None)
            if reader is None:
                raise ValueError("source does not support skip_invalid")
            records, errors = reader(path, skip_invalid=True)
            if self.diagnostics is not None:
                self.diagnostics.extend(errors)
        else:
            records = source.read(path)
        if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records
        ):
            raise TypeError("data source must return a list of objects")
        return select_records(
            records,
            max_samples=self.max_samples,
            shuffle=self.shuffle,
            seed=self.seed,
        )
