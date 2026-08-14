"""Load and validate Benchmark parameter files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from luban_meter.core.errors import ConfigurationError


def load_benchmark_config(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"benchmark config does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid benchmark config {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"benchmark config must contain a mapping: {path}")
    return dict(data)
