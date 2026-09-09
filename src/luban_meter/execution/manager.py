"""Create the Host ExecutionSession, optionally with hardware monitoring.

Hardware monitoring is enabled by setting LUBAN_MONITOR_URL to a
Prometheus exporter endpoint (e.g. http://host:9400). If not set,
monitoring is disabled entirely.
"""

from __future__ import annotations

import os

from luban_meter.execution.host import HostSession
from luban_meter.execution.session import ExecutionSession


class ExecutionManager:
    def open_session(self) -> ExecutionSession:
        monitor_url = os.environ.get("LUBAN_MONITOR_URL", "") or None
        monitor_interval = float(os.environ.get("LUBAN_MONITOR_INTERVAL", "1.0"))
        return HostSession(
            monitor_url=monitor_url,
            monitor_interval=monitor_interval,
        )
