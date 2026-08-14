"""Create the direct Host ExecutionSession."""

from __future__ import annotations

from luban_meter.execution.host import HostSession
from luban_meter.execution.session import ExecutionSession


class ExecutionManager:
    def open_session(self) -> ExecutionSession:
        return HostSession()
