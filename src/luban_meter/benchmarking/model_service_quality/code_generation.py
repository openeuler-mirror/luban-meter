"""Code generation collection and scoring."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
    ServiceError,
)
from luban_meter.benchmarking.model_service_quality.common.metrics import (
    pass_at_k,
)
from luban_meter.benchmarking.model_service_quality.common.parameters import (
    boolean_value,
    fixed_value,
    non_empty_string,
    non_negative_integer,
    positive_integer,
    positive_number,
    string_list,
    string_value,
)
from luban_meter.benchmarking.model_service_quality.common.parsers import (
    extract_final_output,
    extract_humaneval_completion,
)
from luban_meter.benchmarking.model_service_quality.common.prompts import (
    validate_prompt_version,
)
from luban_meter.benchmarking.model_service_quality.common.records import (
    object_list,
    scalar,
)
from luban_meter.benchmarking.model_service_quality.common.sandbox import (
    DockerSandboxExecutor,
    SandboxConfig,
)

MEASUREMENT = "humaneval_pass_at_1_online_service"
PROTOCOL = "humaneval-completion-v1"
TRANSPORT = "openai_compatible_completions"
SCORER_VERSION = "humaneval-scorer-v1"
SANDBOX_PROTOCOL = "humaneval-sandbox-v1"


def _fixed_pass_k(parameters: Mapping[str, Any]) -> list[int]:
    value = parameters.get("pass_k", [1])
    if value != [1]:
        raise ValueError("pass_k is fixed to [1] in the pass@1 release")
    return [1]


def validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    config = {
        "service_url": non_empty_string(
            parameters, "service_url", "http://127.0.0.1:8000"
        ),
        "api_key": string_value(parameters, "api_key", ""),
        "request_timeout": positive_number(parameters, "request_timeout", 120),
        "dataset_path": non_empty_string(parameters, "dataset_path", ""),
        "split": non_empty_string(parameters, "split", "test"),
        "max_samples": positive_integer(parameters, "max_samples", 164),
        "shuffle": boolean_value(parameters, "shuffle", False),
        "seed": non_negative_integer(parameters, "seed", 42),
        "prompt_version": non_empty_string(
            parameters, "prompt_version", "humaneval-completion-v1"
        ),
        "max_tokens": positive_integer(parameters, "max_tokens", 1024),
        "stop": string_list(parameters, "stop", []),
        "max_concurrency": positive_integer(parameters, "max_concurrency", 8),
        "sandbox_concurrency": positive_integer(
            parameters, "sandbox_concurrency", 4
        ),
        "docker_host": non_empty_string(
            parameters, "docker_host", "unix:///var/run/docker.sock"
        ),
        "docker_binary": non_empty_string(
            parameters, "docker_binary", "docker"
        ),
        "sandbox_image": non_empty_string(
            parameters,
            "sandbox_image",
            "luban-meter-humaneval-sandbox:v1",
        ),
        "sandbox_runtime": non_empty_string(
            parameters, "sandbox_runtime", "runc"
        ),
        "sandbox_timeout": positive_number(parameters, "sandbox_timeout", 3),
        "sandbox_startup_grace": positive_number(
            parameters, "sandbox_startup_grace", 5
        ),
        "sandbox_memory_mb": positive_integer(
            parameters, "sandbox_memory_mb", 256
        ),
        "sandbox_cpus": positive_number(parameters, "sandbox_cpus", 1),
        "sandbox_pids_limit": positive_integer(
            parameters, "sandbox_pids_limit", 32
        ),
        "sandbox_tmpfs_mb": positive_integer(
            parameters, "sandbox_tmpfs_mb", 64
        ),
        "sandbox_output_limit_bytes": positive_integer(
            parameters, "sandbox_output_limit_bytes", 64 * 1024
        ),
        "pass_k": _fixed_pass_k(parameters),
    }
    fixed_value(parameters, "eval_mode", "gen")
    fixed_value(parameters, "prompt_format", "base")
    fixed_value(parameters, "samples_per_task", 1)
    fixed_value(parameters, "temperature", 0.0)
    fixed_value(parameters, "allow_host_fallback", False)
    config.update(
        {
            "eval_mode": "gen",
            "prompt_format": "base",
            "samples_per_task": 1,
            "temperature": 0.0,
            "allow_host_fallback": False,
        }
    )
    validate_prompt_version("humaneval", config["prompt_version"])
    if not config["docker_host"].startswith("unix://"):
        raise ValueError("docker_host must use a unix:// socket")
    return config


def prepare_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for index, record in enumerate(records):
        values: dict[str, str] = {}
        for field in ("task_id", "prompt", "test", "entry_point"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"sample {index}: {field} must be a non-empty string"
                )
            values[field] = value
        if values["task_id"] in task_ids:
            raise ValueError(f"duplicate task_id: {values['task_id']}")
        task_ids.add(values["task_id"])
        canonical_solution = record.get("canonical_solution")
        if canonical_solution is not None and not isinstance(
            canonical_solution, str
        ):
            raise TypeError(
                f"sample {index}: canonical_solution must be a string"
            )
        prepared.append({**record, **values})
    return prepared


def build_program(sample: Mapping[str, Any], completion: str) -> str:
    program = str(sample["prompt"]) + completion
    if not program.endswith("\n"):
        program += "\n"
    program += str(sample["test"])
    if not program.endswith("\n"):
        program += "\n"
    program += f"check({sample['entry_point']})\n"
    return program


def generate_completion(
    client: OpenAIClient, prompt: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    return client.complete(
        prompt,
        temperature=0.0,
        max_tokens=int(config["max_tokens"]),
        stop=list(config["stop"]),
    )


def collect_sample(
    client: OpenAIClient,
    executor: DockerSandboxExecutor,
    sandbox_semaphore: threading.BoundedSemaphore,
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(sample["task_id"])
    prompt = str(sample["prompt"])
    record: dict[str, Any] = {
        "task_id": task_id,
        "sample_id": 0,
        "mode": "gen",
        "prompt_version": config["prompt_version"],
        "prompt_format": "base",
        "prompt": prompt,
        "raw_output": None,
        "final_output": None,
        "parsed_output": None,
        "program_sha256": None,
        "passed": False,
        "execution_status": None,
        "execution_ms": 0.0,
        "execution_stdout": "",
        "execution_stderr": "",
        "execution_stdout_truncated": False,
        "execution_stderr_truncated": False,
        "latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "status": "service_failed",
        "error": None,
    }
    try:
        response = generate_completion(client, prompt, config)
    except ServiceError as exc:
        record["error"] = str(exc)
        return record

    raw_output = str(response.get("text") or "")
    record.update(
        {
            "raw_output": raw_output,
            "input_tokens": response.get("input_tokens") or 0,
            "output_tokens": response.get("output_tokens") or 0,
            "latency_ms": response.get("latency_ms") or 0.0,
        }
    )
    try:
        record["final_output"] = extract_final_output(raw_output, code=True)
    except ValueError as exc:
        record.update(status="parse_failed", error=str(exc))
        return record
    completion = extract_humaneval_completion(record["final_output"], prompt)
    record["parsed_output"] = completion
    if not completion:
        record["status"] = "parse_failed"
        record["error"] = "model returned no executable completion"
        return record

    program = build_program(sample, completion)
    record["program_sha256"] = hashlib.sha256(
        program.encode("utf-8")
    ).hexdigest()
    with sandbox_semaphore:
        execution = executor.execute(program)
    record.update(
        {
            "passed": execution.passed,
            "execution_status": execution.status,
            "execution_ms": execution.execution_ms,
            "execution_stdout": execution.stdout,
            "execution_stderr": execution.stderr,
            "execution_stdout_truncated": execution.stdout_truncated,
            "execution_stderr_truncated": execution.stderr_truncated,
        }
    )
    if execution.status == "sandbox_error":
        record["status"] = "sandbox_failed"
    else:
        record["status"] = "success"
    if execution.error_type or execution.error_message:
        record["error"] = {
            "type": execution.error_type,
            "message": execution.error_message,
        }
    return record


def _sandbox_config(config: Mapping[str, Any]) -> SandboxConfig:
    return SandboxConfig(
        docker_host=str(config["docker_host"]),
        image=str(config["sandbox_image"]),
        docker_binary=str(config["docker_binary"]),
        runtime=str(config["sandbox_runtime"]),
        timeout_seconds=float(config["sandbox_timeout"]),
        startup_grace_seconds=float(config["sandbox_startup_grace"]),
        memory_mb=int(config["sandbox_memory_mb"]),
        cpus=float(config["sandbox_cpus"]),
        pids_limit=int(config["sandbox_pids_limit"]),
        tmpfs_mb=int(config["sandbox_tmpfs_mb"]),
        output_limit_bytes=int(config["sandbox_output_limit_bytes"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SAMPLE_STATUSES = (
    "success",
    "parse_failed",
    "service_failed",
    "sandbox_failed",
)
EXECUTION_STATUSES = (
    "passed",
    "failed_test",
    "syntax_error",
    "runtime_error",
    "timeout",
    "sandbox_error",
)


def process_sample(raw_sample: Mapping[str, Any]) -> dict[str, Any]:
    task_id = raw_sample.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("sample task_id must be a non-empty string")
    sample_id = raw_sample.get("sample_id")
    if sample_id != 0:
        raise ValueError("pass@1 requires sample_id=0 for every task")
    status = raw_sample.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"sample status must be one of {SAMPLE_STATUSES}")
    passed = raw_sample.get("passed")
    if not isinstance(passed, bool):
        raise TypeError("sample passed must be a boolean")
    execution_status = raw_sample.get("execution_status")
    if (
        execution_status is not None
        and execution_status not in EXECUTION_STATUSES
    ):
        raise ValueError(
            f"execution_status must be one of {EXECUTION_STATUSES} or null"
        )
    if status != "success" and passed:
        raise ValueError("failed collection records cannot be marked passed")
    if status == "success" and execution_status is None:
        raise ValueError("successful collection records need execution_status")
    return {
        "task_id": task_id,
        "status": status,
        "execution_status": execution_status,
        "passed": passed,
    }


def process(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_status = raw_result.get("status")
    metadata = raw_result.get("metadata")
    result_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    if raw_status == "failed":
        result_metadata.setdefault("scorer_version", SCORER_VERSION)
        return {
            "status": "failed",
            "metrics": {},
            "metadata": result_metadata,
            "error": raw_result.get("error")
            or {
                "type": "UnknownError",
                "message": "raw result failed without error detail",
            },
        }
    if raw_status != "success":
        raise ValueError(f"raw status must be success or failed: {raw_status}")

    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")
    raw_samples = object_list(raw_metrics.get("samples"), "raw samples")
    if not raw_samples:
        raise ValueError("raw samples must not be empty")
    samples = [process_sample(sample) for sample in raw_samples]
    task_ids = [sample["task_id"] for sample in samples]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("pass@1 requires exactly one record per task_id")

    total = len(samples)
    passed = sum(1 for sample in samples if sample["passed"])
    pass_at_1 = (
        sum(pass_at_k(1, int(sample["passed"]), 1) for sample in samples)
        / total
    )
    parse_failed = sum(
        1 for sample in samples if sample["status"] == "parse_failed"
    )
    service_failed = sum(
        1 for sample in samples if sample["status"] == "service_failed"
    )
    sandbox_failed = sum(
        1 for sample in samples if sample["status"] == "sandbox_failed"
    )
    timeouts = sum(
        1 for sample in samples if sample["execution_status"] == "timeout"
    )
    infrastructure_failed = service_failed + sandbox_failed
    if infrastructure_failed == total:
        result_status = "failed"
    elif infrastructure_failed:
        result_status = "partial_failed"
    else:
        result_status = "success"

    result_metadata.setdefault("scorer_version", SCORER_VERSION)
    result_metadata.update(
        {
            "passed_samples": passed,
            "parse_failed_samples": parse_failed,
            "service_failed_samples": service_failed,
            "sandbox_failed_samples": sandbox_failed,
        }
    )
    error = None
    if infrastructure_failed:
        error = {
            "type": "InfrastructureFailures",
            "message": (
                f"{service_failed} service and {sandbox_failed} sandbox "
                "failures were counted as failed candidates"
            ),
        }
    return {
        "status": result_status,
        "metrics": {
            "task_view": {
                "humaneval": {
                    "pass_at_1": scalar(pass_at_1, "ratio", total),
                    "total_tasks": scalar(total, "sample"),
                    "passed": scalar(passed, "sample"),
                    "parse_failed": scalar(parse_failed, "sample"),
                    "service_failed": scalar(service_failed, "sample"),
                    "sandbox_failed": scalar(sandbox_failed, "sample"),
                    "execution_timeout": scalar(timeouts, "sample"),
                }
            }
        },
        "metadata": result_metadata,
        "error": error,
    }
