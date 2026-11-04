"""One parameter definition for flat CLI options and task validation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luban_meter.log import get_logger

logger = get_logger(__name__)


MISSING = object()


@dataclass(frozen=True)
class Parameter:
    """Describe a scalar or list without depending on a CLI framework."""

    type: type = str
    default: Any = MISSING
    required: bool = False
    choices: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    item_type: type | None = None
    path: bool = False
    secret: bool = False
    matrix: bool = False
    allow_infinity: bool = False
    help: str = ""

    def validate(self, name: str, value: Any) -> Any:
        if value is None:
            if self.required:
                raise ValueError(f"{name} is required")
            if self.default is None:
                return None
            raise ValueError(f"{name} cannot be null")
        if self.type is list:
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list")
            item = Parameter(self.item_type or str)
            return [item.validate(name, entry) for entry in value]
        if self.type is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        elif self.type in (int, float):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a number")
            if self.type is int and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            try:
                value = self.type(value)
            except (ValueError, TypeError) as exc:
                logger.exception("Numeric parameter conversion failed")
                raise ValueError(f"{name} must be a number") from exc
            if not math.isfinite(value):
                if self.allow_infinity and value == math.inf:
                    return "inf"
                raise ValueError(f"{name} must be finite")
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"{name} must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"{name} must be <= {self.maximum}")
        elif not isinstance(value, self.type):
            raise ValueError(f"{name} must be {self.type.__name__}")
        if self.choices and value not in self.choices:
            raise ValueError(f"{name} must be one of {self.choices}")
        if self.required and value == "":
            raise ValueError(f"{name} must not be empty")
        return value


def validate_parameters(schema, supplied, *, base_dir: Path):
    unknown = set(supplied) - set(schema)
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
    result = {}
    for name, spec in schema.items():
        value = supplied.get(name, spec.default)
        if value is MISSING:
            if spec.required:
                raise ValueError(f"missing required parameter: {name}")
            continue
        value = spec.validate(name, copy.deepcopy(value))
        if spec.path and value:
            path = Path(value).expanduser()
            value = str((base_dir / path).resolve())
        result[name] = value
    return result


def public_parameters(parameters, schema):
    return {
        name: "[redacted]" if schema[name].secret and value else value
        for name, value in parameters.items()
    }
