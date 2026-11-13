"""Resolve independent tasks and Case matrices before execution."""

from __future__ import annotations

import copy
import itertools
import json
import re
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.metric import metrics
from luban_meter.core.parameters import (
    public_parameters,
    validate_parameters,
)
from luban_meter.core.template_contracts import compatible
from luban_meter.log import add_log_secrets, get_logger, log_context
from luban_meter.runtime.load_profiles import profiles

logger = get_logger(__name__)


TASK_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def bootstrap():
    import luban_meter.benchmarking  # noqa: F401
    import luban_meter.data.sources  # noqa: F401


def schema_for(scenario, parameters):
    schema = scenario.parameter_schema()
    profile = parameters.get("profile", schema["profile"].default)
    implementation = profiles.get(profile)
    if implementation.execution_mode not in scenario.profile_modes:
        raise ValueError(f"incompatible profile: {profile}")
    schema.update(implementation.parameter_schema())
    return schema


def cli_schema():
    bootstrap()
    result = {}
    schemas = [cls.parameter_schema() for _, cls in benchmarks.items()]
    schemas.extend(cls.parameter_schema() for _, cls in profiles.items())
    for schema in schemas:
        for name, spec in schema.items():
            reserved = {
                "scenario",
                "config",
                "task",
                "result_dir",
                "log_level",
                "log_dir",
                "fail_fast",
                "observe",
                "observe_url",
                "observe_interval",
                "help",
            }
            if name in reserved or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"invalid or reserved parameter: {name}")
            previous = result.get(name)
            if previous and (
                previous.type is not spec.type
                or previous.item_type is not spec.item_type
                or previous.path != spec.path
            ):
                raise ValueError(f"conflicting parameter declaration: {name}")
            result[name] = spec
    return result


def load_tasks(path):
    if path is None:
        return [{}], Path.cwd()
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"tasks"}:
        raise ValueError("config must be an object containing only tasks")
    tasks = payload["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    if any(not isinstance(task, dict) for task in tasks):
        raise ValueError("each task must be an object")
    return tasks, path.parent


def expand_cases(definition, schema):
    if definition is None:
        return [{}]
    if not isinstance(definition, dict) or len(definition) != 1:
        raise ValueError("cases must contain exactly one of matrix or items")
    if "matrix" in definition:
        matrix = definition["matrix"]
        if not isinstance(matrix, dict) or not matrix:
            raise ValueError("matrix must be a non-empty object")
        if any(not isinstance(v, list) or not v for v in matrix.values()):
            raise ValueError("matrix values must be non-empty lists")
        count = 1
        for values in matrix.values():
            count *= len(values)
        if count > 100000:
            raise ValueError(f"matrix contains too many Cases: {count}")
        items = [
            dict(zip(matrix, values))
            for values in itertools.product(*matrix.values())
        ]
    elif "items" in definition:
        items = definition["items"]
        if not isinstance(items, list) or not items:
            raise ValueError("cases.items must be a non-empty list")
    else:
        raise ValueError("cases must use matrix or items")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each Case must be an object")
        for name in item:
            if name not in schema or not schema[name].matrix:
                raise ValueError(f"parameter cannot vary by Case: {name}")
    return items


def build_plan(base, config=None, selected=(), *, observe=False):
    bootstrap()
    entries, directory = load_tasks(config)
    common = copy.deepcopy(base)
    names = []
    tasks = []
    for index, entry in enumerate(entries):
        name = entry.get("name", f"task_{index + 1}")
        if not isinstance(name, str) or not TASK_NAME.fullmatch(name):
            raise ValueError(f"invalid task name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate task name: {name}")
        names.append(name)
        if selected and name not in selected:
            continue
        try:
            supplied = {**common, **copy.deepcopy(entry)}
            scenario_name = supplied.pop("scenario", None)
            if not scenario_name:
                raise ValueError("scenario is required")
            scenario = benchmarks.get(scenario_name)
            supplied.pop("name", None)
            case_definition = supplied.pop("cases", None)
            schema = schema_for(scenario, supplied)
            add_log_secrets(
                supplied[key] for key, spec in schema.items()
                if spec.secret and isinstance(supplied.get(key), str)
            )
            logger.info("Resolving task: name=%s scenario=%s", name,
                        scenario_name)
            # Resolve CLI paths before applying file-relative paths.
            for key, value in common.items():
                if key not in entry and key in schema and schema[key].path:
                    if value:
                        supplied[key] = str(Path(value).expanduser().resolve())
            case_parameters = []
            for overrides in expand_cases(case_definition, schema):
                parameters = validate_parameters(
                    schema, {**supplied, **overrides}, base_dir=directory
                )
                scenario.validate(parameters)
                for key, spec in schema.items():
                    if spec.path and parameters.get(key):
                        if not Path(parameters[key]).is_file():
                            raise ValueError(f"{key} does not exist")
                chosen = scenario.select_metrics(parameters)
                if len(chosen) != len(set(chosen)):
                    raise ValueError("metric names must be unique")
                for metric_name in chosen:
                    compatible(
                        scenario.provided_fields(parameters),
                        metrics.get(metric_name).required_fields,
                    )
                    missing = (
                        metrics.get(metric_name).requires
                        - scenario.capabilities
                    )
                    if missing:
                        raise ValueError(
                            f"metric {metric_name} requires {sorted(missing)}"
                        )
                parameters["metrics"] = chosen
                if observe and parameters.get("backend") != "openai_http":
                    raise ValueError(
                        "Prometheus observer requires an HTTP backend"
                    )
                case_parameters.append(parameters)
            tasks.append(
                {
                    "name": name,
                    "scenario": scenario_name,
                    "category": scenario.category,
                    "cases": [
                        {"id": f"case_{i + 1}", "parameters": p}
                        for i, p in enumerate(case_parameters)
                    ],
                }
            )
        except (ValueError, TypeError) as exc:
            logger.exception("Task parameter resolution failed")
            raise ValueError(f"task {name}: {exc}") from exc
    unknown = set(selected) - set(names)
    if unknown:
        raise ValueError(f"unknown task selection: {sorted(unknown)}")
    return {"tasks": tasks, "case_count": sum(len(t["cases"]) for t in tasks)}


def validate_plan_parameters(plan, *, base_dir):
    """Recheck saved Case parameters without rewriting their snapshot."""
    bootstrap()
    for task in plan["tasks"]:
        for case in task["cases"]:
            with log_context(task=task["name"], case=case["id"],
                             phase="validate"):
                logger.info("Validating saved Case: scenario=%s",
                            task["scenario"])
                try:
                    scenario = benchmarks.get(task["scenario"])
                    supplied = case["parameters"]
                    parameters = validate_parameters(
                        schema_for(scenario, supplied), supplied,
                        base_dir=base_dir,
                    )
                    scenario.validate(parameters)
                except (ValueError, TypeError):
                    logger.exception(
                        "Saved Case parameter validation failed; update or "
                        "recreate the schedule with a valid task JSON"
                    )
                    raise


def public_plan(plan):
    result = copy.deepcopy(plan)
    for task in result["tasks"]:
        scenario = benchmarks.get(task["scenario"])
        for case in task["cases"]:
            p = case["parameters"]
            case["parameters"] = public_parameters(p, schema_for(scenario, p))
    return result
