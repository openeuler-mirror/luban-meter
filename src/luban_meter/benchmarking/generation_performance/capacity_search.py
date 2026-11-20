"""Step-stage capacity judgment over persisted formal request facts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from luban_meter.api import Metric, metrics
from luban_meter.benchmarking.generation_performance.common.statistics import (
    scalar,
)
from luban_meter.benchmarking.generation_performance.online_serving import (
    compute_goodput,
)
from luban_meter.log import get_logger

logger = get_logger(__name__)


def select_capacity_metrics(parameters, names):
    """Bind the capacity metric to the step profile explicitly.

    An explicitly configured metrics list wins: the capacity metric is
    appended only to the scenario defaults when the step profile is
    selected without an explicit list.
    """
    if "capacity_search" in parameters["metrics"]:
        if parameters.get("profile") != "step":
            raise ValueError("capacity_search requires the step profile")
        return names
    if parameters.get("profile") == "step" and not parameters["metrics"]:
        return [*names, "capacity_search"]
    return names


def is_timeout_failure(record):
    """Classify saved failures whose error type denotes a timeout."""
    response = record.get("response")
    if isinstance(response, Mapping):
        error = response.get("error")
        if isinstance(error, Mapping):
            kind = error.get("type")
            if isinstance(kind, str) and "timeout" in kind.lower():
                return True
    message = record.get("error")
    return isinstance(message, str) and "timed out" in message.lower()


def _stage_of(record, stage_duration):
    offset = record.get("scheduled_offset_s")
    if (
        not isinstance(offset, (int, float))
        or isinstance(offset, bool)
        or not math.isfinite(offset)
        or offset < 0
    ):
        raise ValueError(
            "request scheduled_offset_s must be a finite number >= 0"
        )
    return math.floor(float(offset) / stage_duration)


def _stage_rate(parameters, stage):
    rate = parameters["step_start_rate"] + stage * parameters[
        "step_rate_increment"
    ]
    max_rate = parameters.get("step_max_rate")
    if max_rate is not None and rate > max_rate:
        rate = max_rate
    return float(rate)


def _judge_stage(records, stage, parameters, slo_config):
    """Judge one stage from its admitted request facts."""
    duration = float(parameters["step_stage_duration"])
    min_requests = int(parameters["stage_min_requests"])
    offered_rate = _stage_rate(parameters, stage)
    admitted = len(records)
    successful = [
        record
        for record in records
        if record.get("status") == "success"
        and isinstance(record.get("response"), Mapping)
    ]
    failed = [
        record for record in records if record.get("status") != "success"
    ]
    # Success without a saved response is a data integrity problem, not a
    # judgement outcome; keep it strict like the other streaming metrics.
    if len(successful) + len(failed) != admitted:
        raise ValueError("successful requests must carry a response record")
    timeouts = [
        record for record in failed if is_timeout_failure(record)
    ]
    entry: dict[str, Any] = {
        "stage_index": stage,
        "window": {
            "start_s": stage * duration,
            "end_s": (stage + 1) * duration,
        },
        "offered_request_rate": scalar(offered_rate, "req/s"),
        "admitted_requests": scalar(admitted, "request"),
        "successful_requests": scalar(len(successful), "request"),
        "failed_requests": scalar(len(failed), "request"),
        "timeout_requests": scalar(len(timeouts), "request"),
        "success_rate": scalar(
            len(successful) / admitted if admitted else None, "ratio"
        ),
        "timeout_rate": scalar(
            len(timeouts) / admitted if admitted else None, "ratio"
        ),
    }
    goodput = None
    if slo_config:
        goodput = compute_goodput(
            successful=[record["response"] for record in successful],
            failed_count=len(failed),
            duration_seconds=duration,
            slo_config=slo_config,
        )
        entry["slo"] = {
            "satisfied_count": goodput["slo_satisfied_count"],
            "violated_count": goodput["slo_violated_count"],
            "failed_count": goodput["failed_count"],
            "undetermined_count": goodput["undetermined_count"],
            "satisfied_rate": goodput["slo_satisfied_rate"],
            "goodput_request_throughput": goodput[
                "goodput_request_throughput"
            ],
        }

    violated: list[str] = []
    if admitted < min_requests:
        entry["verdict"] = "insufficient_samples"
        entry["reason"] = (
            f"admitted {admitted} requests; stage_min_requests is "
            f"{min_requests}"
        )
    elif goodput is not None and goodput["determined_count"]["value"] == 0:
        entry["verdict"] = "undetermined"
        entry["reason"] = (
            "no request in this stage has a determined SLO outcome"
        )
    else:
        if len(successful) / admitted < parameters["stage_min_success_rate"]:
            violated.append("success_rate")
        if (
            len(timeouts) / admitted
            > parameters["stage_max_timeout_rate"]
        ):
            violated.append("timeout_rate")
        if goodput is not None:
            satisfied_rate = goodput["slo_satisfied_rate"]["value"]
            if satisfied_rate is None or satisfied_rate < (
                parameters["stage_min_slo_rate"]
            ):
                violated.append("slo_satisfied_rate")
        entry["verdict"] = "unstable" if violated else "stable"
        entry["violated_criteria"] = violated
    return entry


def _search_verdict(stages, max_rate):
    """Walk the stages in order and locate the stability boundary."""
    last_stable = None
    for entry in stages:
        if entry["verdict"] == "stable":
            last_stable = entry
            continue
        rate = (
            last_stable["offered_request_rate"]["value"]
            if last_stable is not None
            else None
        )
        if entry["verdict"] == "unstable":
            return {
                "status": "found" if last_stable is not None else "not_found",
                "max_stable_request_rate": scalar(rate, "req/s"),
                "max_stable_stage": (
                    scalar(last_stable["stage_index"], "stage")
                    if last_stable is not None
                    else scalar(None, "stage")
                ),
                "first_unstable_stage": scalar(
                    entry["stage_index"], "stage"
                ),
                "unstable_criteria": entry.get("violated_criteria"),
                "reason": (
                    "first unstable stage: {}".format(entry["stage_index"])
                    if last_stable is not None
                    else "no stage met the stability criteria; the maximum "
                    "stable rate is below step_start_rate"
                ),
            }
        return {
            "status": "inconclusive",
            "max_stable_request_rate": scalar(rate, "req/s"),
            "max_stable_stage": (
                scalar(last_stable["stage_index"], "stage")
                if last_stable is not None
                else scalar(None, "stage")
            ),
            "first_unjudged_stage": scalar(entry["stage_index"], "stage"),
            "reason": (
                "stage {} {} before any unstable stage: {}".format(
                    entry["stage_index"], entry["verdict"],
                    entry.get("reason", ""),
                )
            ),
            "condition": (
                "lower bound only" if last_stable is not None else None
            ),
        }
    last = stages[-1]
    rate = last["offered_request_rate"]["value"]
    if max_rate is not None and rate >= max_rate:
        return {
            "status": "bounded",
            "max_stable_request_rate": scalar(rate, "req/s"),
            "max_stable_stage": scalar(last["stage_index"], "stage"),
            "reason": (
                "all observed stages were stable up to the configured "
                "step_max_rate"
            ),
            "condition": (
                "capacity is at least this rate; raise step_max_rate to "
                "search further"
            ),
        }
    return {
        "status": "inconclusive",
        "max_stable_request_rate": scalar(rate, "req/s"),
        "max_stable_stage": scalar(last["stage_index"], "stage"),
        "reason": (
            "the request or duration budget ended while all observed "
            "stages were stable"
        ),
        "condition": "lower bound only",
    }


def compute_capacity_search(facts):
    """Compute staged judgments and the maximum stable offered rate."""
    p = facts["parameters"]
    if p.get("profile") != "step":
        return {
            "status": "unavailable",
            "reason": "capacity_search requires the step profile",
        }
    requests = facts.get("requests") or []
    if not requests:
        return {
            "status": "unavailable",
            "reason": "no admitted formal requests",
        }
    stage_duration = float(p["step_stage_duration"])
    max_rate = p.get("step_max_rate")
    max_rate = None if max_rate is None else float(max_rate)
    metadata = facts.get("metadata")
    slo_config = (
        metadata.get("slo_config") if isinstance(metadata, Mapping) else None
    )
    groups: dict[int, list] = {}
    for record in requests:
        groups.setdefault(_stage_of(record, stage_duration), []).append(
            record
        )

    stages = [
        _judge_stage(groups[stage], stage, p, slo_config)
        for stage in sorted(groups)
    ]
    for entry in stages:
        logger.info(
            "Stage judged: index=%s rate=%s admitted=%s verdict=%s "
            "violated=%s",
            entry["stage_index"],
            entry["offered_request_rate"]["value"],
            entry["admitted_requests"]["value"],
            entry["verdict"],
            entry.get("violated_criteria") or [],
        )
    search = _search_verdict(stages, max_rate)
    logger.info(
        "Capacity search verdict: status=%s max_stable_request_rate=%s",
        search["status"], search["max_stable_request_rate"]["value"],
    )
    return {
        "status": "success",
        "values": {
            "condition": (
                "stage membership uses planned submission offsets; "
                "judgments are offline computations over saved facts"
            ),
            "judgment": {
                "stage_min_requests": scalar(
                    int(p["stage_min_requests"]), "request"
                ),
                "stage_min_slo_rate": scalar(
                    float(p["stage_min_slo_rate"]), "ratio"
                ),
                "stage_min_success_rate": scalar(
                    float(p["stage_min_success_rate"]), "ratio"
                ),
                "stage_max_timeout_rate": scalar(
                    float(p["stage_max_timeout_rate"]), "ratio"
                ),
                "slo_config": dict(slo_config) if slo_config else None,
                "slo_rate_definition": (
                    "goodput slo_satisfied_rate; failed requests count as "
                    "determined and not satisfied"
                ),
                "success_rate_definition": (
                    "successful / admitted requests in the stage"
                ),
                "timeout_rule": (
                    "failures whose saved error type contains 'timeout' "
                    "(case-insensitive)"
                ),
                "stage_rate": (
                    "min(step_start_rate + stage_index * "
                    "step_rate_increment, step_max_rate)"
                ),
            },
            "stages": stages,
            "search": search,
        },
    }


@metrics.register("capacity_search")
class CapacitySearch(Metric):
    """Maximum stable offered rate from staged step-load facts."""

    requires = frozenset({"client_stream_timing"})

    def compute(self, facts):
        return compute_capacity_search(facts)
