"""Explicit, typed registration without filesystem discovery."""

from __future__ import annotations

import inspect
import re
from typing import Generic, TypeVar

from luban_meter.log import get_logger

logger = get_logger(__name__)


T = TypeVar("T")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class Registry(Generic[T]):
    """Own one extension namespace; reject ambiguous registrations."""

    def __init__(self, label: str, base: type[T]) -> None:
        self.label = label
        self.base = base
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str):
        def decorate(implementation: type[T]) -> type[T]:
            if not NAME.fullmatch(name):
                raise ValueError(f"invalid {self.label} name: {name!r}")
            if not issubclass(implementation, self.base):
                raise TypeError(f"{name} must inherit {self.base.__name__}")
            if inspect.isabstract(implementation):
                raise TypeError(f"{name} has unimplemented abstract methods")
            if name in self._classes:
                raise ValueError(f"duplicate {self.label}: {name}")
            self._classes[name] = implementation
            return implementation

        return decorate

    def get(self, name: str) -> type[T]:
        try:
            return self._classes[name]
        except KeyError as exc:
            logger.exception("Registry lookup failed")
            available = ", ".join(self.names())
            raise ValueError(
                f"unknown {self.label} {name!r}; available: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))

    def items(self):
        return tuple((name, self._classes[name]) for name in self.names())
