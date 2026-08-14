"""Optional local monitoring contracts for future implementations."""

from dataclasses import dataclass

from luban_meter.core.models import CommandSpec


@dataclass(frozen=True)
class MonitorSpec:
    command: CommandSpec
    interval_seconds: float = 1.0
