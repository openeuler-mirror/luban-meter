"""Format adapters preserve fields for performance and quality tasks."""

import csv
from abc import ABC, abstractmethod
from pathlib import Path

from luban_meter.core.registry import Registry
from luban_meter.data.engine import load_records


class DataSource(ABC):
    @abstractmethod
    def read(self, path: Path) -> list[dict]:
        """Read local records without applying task semantics."""


sources = Registry("data source", DataSource)


@sources.register("json")
class JSONSource(DataSource):
    def read(self, path):
        return load_records(path, format="json")


@sources.register("jsonl")
class JSONLinesSource(DataSource):
    def read(self, path):
        return load_records(path, format="jsonl")


@sources.register("csv")
class CSVSource(DataSource):
    def read(self, path):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            names = reader.fieldnames
            if not names or any(not name for name in names):
                raise ValueError("CSV requires named columns")
            if len(names) != len(set(names)):
                raise ValueError("CSV column names must be unique")
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"invalid CSV row at line {reader.line_num}"
                    )
                rows.append(dict(row))
            return rows


@sources.register("sharegpt")
class ShareGPTSource(JSONSource):
    def read(self, path):
        records = super().read(path)
        for row in records:
            turns = row.get("conversations")
            if not isinstance(turns, list) or any(
                not isinstance(turn, dict) for turn in turns
            ):
                raise ValueError("ShareGPT conversations must be objects")
        return records


@sources.register("squad")
class SquadSource(DataSource):
    def read(self, path):
        from luban_meter.benchmarking.model_service_quality import (
            reading_comprehension,
        )

        return reading_comprehension.load_samples(path)


@sources.register("burstgpt")
class BurstGPTSource(CSVSource):
    @staticmethod
    def read_validated(path, *, skip_invalid=False):
        from luban_meter.data.trace_adapter import read_trace

        return read_trace(path, skip_invalid=skip_invalid)

    def read(self, path):
        from luban_meter.data.trace_adapter import read_trace

        records, _ = read_trace(path)
        return records
