"""Worker entry point: load the same explicit registrations as the parent."""

import json
import signal
import sys
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import bootstrap
from luban_meter.log import configure_logging, get_logger
from luban_meter.runtime.evaluation_session import run_case
from luban_meter.utils.json_io import write_json_atomic

logger = get_logger("luban_meter.runtime.task_worker")


def aggregate_status(results):
    states = [result["status"] for result in results]
    if "interrupted" in states:
        return "interrupted"
    if not states or all(state == "skipped" for state in states):
        return "skipped"
    if all(state == "success" for state in states):
        return "success"
    if all(state in ("failed", "skipped", "discarded") for state in states):
        return "failed"
    return "partial_failed"


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    source = Path(sys.argv[1])
    payload = json.loads(source.read_text())
    source.unlink()
    task = payload["task"]
    with configure_logging(
        payload["log_directory"], filename=payload["log_filename"],
        task=task["name"],
        level=payload["log_level"],
        secrets=payload.get("log_secrets", ()),
    ):
        logger.debug("Worker logging configured: level=%s",
                     payload["log_level"])
        logger.info("Worker initializing: scenario=%s", task["scenario"])
        bootstrap()
        _run_task(payload)


def _run_task(payload):
    task = payload["task"]
    directory = Path(payload["directory"])
    stop_file = Path(payload["stop_file"])
    implementation = benchmarks.get(task["scenario"])
    result = {
        "kind": "task",
        "name": task["name"],
        "scenario": task["scenario"],
        "category": task["category"],
        "cases": [],
        "status": "running",
    }
    stop_reason = None
    for case in task["cases"]:
        if stop_file.exists() or stop_reason:
            collected = {
                "id": case["id"],
                "status": "skipped",
                "reason": stop_reason or "user_interrupt",
            }
            logger.info("Case skipped: reason=%s", collected["reason"],
                        extra={"case": case["id"]})
        else:
            collected = run_case(
                implementation,
                case,
                directory,
                stop_file,
                payload.get("observation"),
                summary_ack=True,
            )
        stop_reason = collected.get("stop_reason") or stop_reason
        result["cases"].append(collected)
        result["status"] = aggregate_status(result["cases"])
        write_json_atomic(directory / "result.json", result)
    if stop_file.exists():
        result["status"] = "interrupted"
    write_json_atomic(directory / "result.json", result)
    logger.info("Worker finished: status=%s result=%s",
                result["status"], directory / "result.json")


if __name__ == "__main__":
    main()
