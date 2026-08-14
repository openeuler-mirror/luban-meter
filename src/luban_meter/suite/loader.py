"""Load a Suite from one vendor's bundled suites directory."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from luban_meter.core.errors import ConfigurationError
from luban_meter.suite.models import SuiteDefinition, SuiteTask

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SuiteLoader:
    def __init__(self, vendors_dir: Path | None = None) -> None:
        self._vendors_dir = vendors_dir or Path(__file__).parents[1] / "vendors"

    def load(self, vendor: str, suite: str) -> SuiteDefinition:
        self._validate_name("vendor", vendor)
        self._validate_name("suite", suite)
        source = self._vendors_dir / vendor / "suites" / f"{suite}.yaml"
        if not source.is_file():
            raise ConfigurationError(
                f"suite does not exist for vendor {vendor}: {source}"
            )

        data = self._read_yaml(source)
        name = data.get("name", suite)
        if not isinstance(name, str) or not name:
            raise ConfigurationError(f"suite name must be a non-empty string: {source}")
        self._validate_name("suite name", name)
        task_values = data.get("tasks")
        if not isinstance(task_values, list) or not task_values:
            raise ConfigurationError(f"suite tasks must be a non-empty list: {source}")

        tasks = tuple(
            self._load_task(value, source, index)
            for index, value in enumerate(task_values)
        )
        names = [task.name for task in tasks]
        if len(names) != len(set(names)):
            raise ConfigurationError(f"suite task names must be unique: {source}")
        return SuiteDefinition(
            name=name,
            vendor=vendor,
            source=source,
            tasks=tasks,
        )

    @staticmethod
    def _read_yaml(path: Path) -> Mapping[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"invalid suite config {path}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ConfigurationError(f"suite config must contain a mapping: {path}")
        return data

    def _load_task(self, value: Any, source: Path, index: int) -> SuiteTask:
        if not isinstance(value, Mapping):
            raise ConfigurationError(
                f"suite task at index {index} must be a mapping: {source}"
            )

        fields = {}
        for field in ("name", "module", "benchmark", "config"):
            item = value.get(field)
            if not isinstance(item, str) or not item:
                raise ConfigurationError(
                    f"suite task at index {index} requires string field {field!r}"
                )
            fields[field] = item

        for field in ("name", "module", "benchmark"):
            self._validate_name(f"task {field}", fields[field])

        timeout = value.get("timeout")
        if timeout is not None and (
            not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0
        ):
            raise ConfigurationError(
                f"suite task {fields['name']!r} timeout must be a positive integer"
            )

        config = Path(fields["config"]).expanduser()
        if not config.is_absolute():
            config = source.parent / config
        return SuiteTask(
            name=fields["name"],
            module=fields["module"],
            benchmark=fields["benchmark"],
            config=config,
            timeout=timeout,
        )

    @staticmethod
    def _validate_name(field: str, value: str) -> None:
        if not _SAFE_NAME.fullmatch(value):
            raise ConfigurationError(
                f"{field} must match {_SAFE_NAME.pattern!r}: {value!r}"
            )
