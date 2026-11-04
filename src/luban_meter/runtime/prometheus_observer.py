"""Optional Prometheus observation; failures do not stop inference."""

import json
import threading
import time
import urllib.request
from contextvars import copy_context

from luban_meter.benchmarking.generation_performance.common.prometheus import (
    parse_prometheus_text,
)
from luban_meter.log import get_logger

logger = get_logger(__name__)


class PrometheusObserver:
    def __init__(self, url, interval, api_key, emit):
        self.url = url
        self.interval = interval
        self.api_key = api_key
        self.emit = emit
        self.snapshots = []
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        snapshot = {"started_at": time.time(), "metrics": {}}
        try:
            headers = {"Accept": "text/plain"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(self.url, headers=headers)
            with urllib.request.urlopen(request, timeout=5) as response:
                parsed = parse_prometheus_text(response.read().decode("utf-8"))
                json.dumps(parsed, allow_nan=False)
                snapshot["metrics"] = parsed
        except Exception as exc:
            logger.warning("Prometheus sampling failed", exc_info=True)
            snapshot["error"] = str(exc)
        if "error" not in snapshot and self.snapshots:
            if "error" in self.snapshots[-1]:
                logger.info("Prometheus sampling recovered")
        snapshot["finished_at"] = time.time()
        self.snapshots.append(snapshot)
        self.emit("observation", snapshot)

    def start(self):
        logger.info("Prometheus observation started")
        self.sample()
        self.thread = threading.Thread(
            target=copy_context().run, args=(self._loop,), daemon=True
        )
        self.thread.start()

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()
        logger.info("Prometheus observation stopped: samples=%s failures=%s",
                    len(self.snapshots),
                    sum("error" in s for s in self.snapshots))

    def result(self):
        from luban_meter.benchmarking.generation_performance.service_observation import (  # noqa: E501
            process,
        )

        valid = [s for s in self.snapshots if "error" not in s]
        duration = (
            valid[-1]["finished_at"] - valid[0]["finished_at"]
            if len(valid) >= 2
            else 0
        )
        result = {
            "status": "success"
            if len(valid) == len(self.snapshots)
            else ("partial_failed" if valid else "failed"),
            "valid_samples": len(valid),
            "failed_samples": len(self.snapshots) - len(valid),
            "actual_window_s": duration,
            "snapshots": self.snapshots,
            "metrics": {},
        }
        if valid:
            result["metrics"] = process(
                {
                    "metrics": {"snapshots": self.snapshots},
                    "metadata": {"collect_duration": duration},
                }
            )["metrics"]
        return result
