"""Retain committed collections and discard incomplete Case evidence."""

import json

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.parameters import public_parameters
from luban_meter.core.planning import schema_for
from luban_meter.log import get_logger
from luban_meter.utils.json_io import write_json_atomic

logger = get_logger(__name__)


def safe_setup_diagnostic(value):
    """Allow only bounded setup states and integer counts in tombstones."""
    if not isinstance(value, dict):
        return {}
    allowed = {
        "status": {
            "pending", "running", "success", "failed", "not_applicable",
            "observed_mismatch",
        },
        "phase": {"seed", "probe", "verify", "capability"},
        "warmup_policy": {"strict", "observe"},
        "reason": {
            "interrupted", "transport_error", "ReadTimeout", "HTTPStatusError",
            "RuntimeError", "ValueError", "JSONDecodeError",
            "missing", "invalid", "input_length", "completion",
            "cache_mismatch",
        },
    }
    result = {
        key: val for key, val in value.items()
        if key in allowed and isinstance(val, str) and val in allowed[key]
    }
    for key in ("prefix_id", "expected_cached_tokens", "cached_tokens",
                "prompt_tokens", "seed_completed", "probe_completed",
                "verify_completed", "warmup_under_hit_prefixes"):
        val = value.get(key)
        if type(val) is int and val >= 0:
            result[key] = val
    for key in ("warmup_completed", "warmup_fully_matched"):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    summary = value.get("warmup_summary")
    if isinstance(summary, dict):
        result["warmup_summary"] = {}
        for phase in ("probe", "verify", "capability"):
            rows = summary.get(phase)
            if not isinstance(rows, list):
                continue
            result["warmup_summary"][phase] = [
                safe_warmup_observation(row) for row in rows
                if isinstance(row, dict)
            ]
    return result


def safe_warmup_observation(row):
    """Retain only numeric cache facts and known observation states."""
    result = {}
    for key in ("prefix_id", "expected_cached_tokens",
                "actual_cached_tokens", "deviation_tokens"):
        val = row.get(key)
        if val is None or (type(val) is int and (
            val >= 0 or key == "deviation_tokens"
        )):
            result[key] = val
    state = row.get("validation_state")
    if isinstance(state, str) and state in {
        "matched", "under_hit", "over_hit",
        "missing", "invalid", "unavailable",
    }:
        result["validation_state"] = state
    return result


def save_setup_diagnostic(directory, case_id, value):
    write_json_atomic(
        directory / "raw" / f"{case_id}.setup.json",
        safe_setup_diagnostic(value),
    )


def discard_case(case, directory, reason):
    """Remove sample/request evidence, retaining only an audit tombstone."""
    raw = directory / "raw"
    setup_path = raw / f"{case['id']}.setup.json"
    try:
        setup = safe_setup_diagnostic(json.loads(setup_path.read_text()))
    except (OSError, ValueError):
        setup = {}
    if setup.get("status") == "running":
        setup.update(status="failed", reason="interrupted")
    for suffix in (".json", ".result.json", ".prepared.json",
                   ".prepared.ack", ".setup.json"):
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
        **({"setup_diagnostic": setup} if setup else {}),
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
        elif (name in started or facts or saved
              or (raw / f"{name}.setup.json").exists()):
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
