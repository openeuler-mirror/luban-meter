"""Retain committed collections and discard incomplete Case evidence."""

import json

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.parameters import public_parameters
from luban_meter.core.planning import schema_for
from luban_meter.log import get_logger
from luban_meter.utils.json_io import write_json_atomic

logger = get_logger(__name__)


def discard_case(case, directory, reason):
    """Remove sample/request evidence, retaining only an audit tombstone."""
    raw = directory / "raw"
    for suffix in (".json", ".result.json", ".prepared.json", ".prepared.ack"):
        (raw / f"{case['id']}{suffix}").unlink(missing_ok=True)
    for temporary in raw.glob(f".{case['id']}.*.tmp"):
        temporary.unlink()
    events = raw / "events.jsonl"
    if events.exists():
        temporary = events.with_suffix(".tmp")
        with events.open() as source, temporary.open("w") as target:
            for line in source:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Ignoring invalid recovery event", exc_info=True,
                    )
                    continue
                if event.get("case_id") != case["id"]:
                    target.write(line)
        temporary.replace(events)
    return {
        "id": case["id"],
        "status": "discarded",
        "parameters": case["parameters"],
        "reason": reason,
    }


def recover_cases(task, directory, outcome, status):
    """Reconcile disk commits; never replay requests or overwrite facts."""
    scenario = benchmarks.get(task["scenario"])
    previous = {row["id"]: row for row in outcome["cases"]}
    raw = directory / "raw"
    events = raw / "events.jsonl"
    started = set()
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                started.add(json.loads(line)["case_id"])
            except (ValueError, KeyError):
                logger.warning("Invalid recovery record", exc_info=True)
                continue
    collected = []
    for case in task["cases"]:
        name = case["id"]
        result_path = raw / f"{name}.result.json"
        facts_path = raw / f"{name}.json"
        saved = (
            json.loads(result_path.read_text()) if result_path.exists() else {}
        )
        facts = (
            json.loads(facts_path.read_text()) if facts_path.exists() else {}
        )
        if facts.get("collection_complete"):
            row = (
                saved
                if saved.get("collection_complete")
                else {
                    "id": name,
                    "status": "partial_failed",
                    "parameters": facts["parameters"],
                    "coverage": facts["coverage"],
                    "window": facts["window"],
                    "metrics": {},
                    "collection_complete": True,
                }
            )
            row["raw"] = str(facts_path)
            if row.get("metrics_status") != "complete":
                row["metrics_status"] = (
                    "partial" if row.get("metrics") else "not_computed"
                )
                if row["status"] == "success":
                    row["status"] = "partial_failed"
            # Original collection and its completed metrics stay intact.
            write_json_atomic(result_path, row)
        elif saved.get("status") in ("discarded", "skipped"):
            row = saved
        elif name in started or facts or saved:
            p = case["parameters"]
            public = public_parameters(p, schema_for(scenario, p))
            row = discard_case(
                {"id": name, "parameters": public}, directory, status
            )
            write_json_atomic(result_path, row)
        else:
            row = previous.get(name, {"id": name, "status": "skipped"})
        collected.append(row)
    outcome["cases"] = collected
