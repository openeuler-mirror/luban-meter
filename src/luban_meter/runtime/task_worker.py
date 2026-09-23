"""Worker entry point: load the same explicit registrations as the parent."""

import json
import signal
import sys
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import bootstrap
from luban_meter.runtime.evaluation_session import run_case
from luban_meter.utils.json_io import write_json_atomic


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
    bootstrap()
    source = Path(sys.argv[1])
    payload = json.loads(source.read_text())
    source.unlink()
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


if __name__ == "__main__":
    main()
