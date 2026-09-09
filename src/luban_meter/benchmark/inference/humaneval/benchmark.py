"""Collect HumanEval pass@1 samples from an OpenAI-compatible service."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from luban_meter.benchmark.inference.common.client import OpenAIClient, ServiceError
from luban_meter.benchmark.inference.common.dataset import (
    load_records,
    resolve_data_path,
    select_records,
)
from luban_meter.benchmark.inference.common.parameters import (
    boolean_value,
    fixed_value,
    non_empty_string,
    non_negative_integer,
    positive_integer,
    positive_number,
    string_list,
    string_value,
)
from luban_meter.benchmark.inference.common.parsers import (
    extract_humaneval_completion,
)
from luban_meter.benchmark.inference.common.prompts import validate_prompt_version
from luban_meter.benchmark.inference.humaneval.executor import (
    DockerSandboxExecutor,
    SandboxConfig,
)

MEASUREMENT = "humaneval_pass_at_1_online_service"
PROTOCOL = "humaneval-completion-v1"
TRANSPORT = "openai_compatible_completions"
SCORER_VERSION = "humaneval-scorer-v1"
SANDBOX_PROTOCOL = "humaneval-sandbox-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HumanEval pass@1 benchmark")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_request(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request")
    parameters = payload.get("parameters")
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be an object")
    return dict(request), dict(parameters)


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
        "sandbox_concurrency": positive_integer(parameters, "sandbox_concurrency", 4),
        "docker_host": non_empty_string(
            parameters, "docker_host", "unix:///var/run/docker.sock"
        ),
        "docker_binary": non_empty_string(parameters, "docker_binary", "docker"),
        "sandbox_image": non_empty_string(
            parameters,
            "sandbox_image",
            "luban-meter-humaneval-sandbox:v1",
        ),
        "sandbox_runtime": non_empty_string(parameters, "sandbox_runtime", "runc"),
        "sandbox_timeout": positive_number(parameters, "sandbox_timeout", 3),
        "sandbox_startup_grace": positive_number(
            parameters, "sandbox_startup_grace", 5
        ),
        "sandbox_memory_mb": positive_integer(parameters, "sandbox_memory_mb", 256),
        "sandbox_cpus": positive_number(parameters, "sandbox_cpus", 1),
        "sandbox_pids_limit": positive_integer(parameters, "sandbox_pids_limit", 32),
        "sandbox_tmpfs_mb": positive_integer(parameters, "sandbox_tmpfs_mb", 64),
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
                raise ValueError(f"sample {index}: {field} must be a non-empty string")
            values[field] = value
        if values["task_id"] in task_ids:
            raise ValueError(f"duplicate task_id: {values['task_id']}")
        task_ids.add(values["task_id"])
        canonical_solution = record.get("canonical_solution")
        if canonical_solution is not None and not isinstance(canonical_solution, str):
            raise TypeError(f"sample {index}: canonical_solution must be a string")
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
    completion = extract_humaneval_completion(raw_output, prompt)
    record.update(
        {
            "raw_output": raw_output,
            "parsed_output": completion,
            "input_tokens": response.get("input_tokens") or 0,
            "output_tokens": response.get("output_tokens") or 0,
            "latency_ms": response.get("latency_ms") or 0.0,
        }
    )
    if not completion:
        record["status"] = "parse_failed"
        record["error"] = "model returned no executable completion"
        return record

    program = build_program(sample, completion)
    record["program_sha256"] = hashlib.sha256(program.encode("utf-8")).hexdigest()
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


def run_benchmark(
    request: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    config = validate_parameters(parameters)
    dataset_file = resolve_data_path(config["dataset_path"])
    samples = prepare_samples(
        select_records(
            load_records(dataset_file),
            max_samples=config["max_samples"],
            shuffle=config["shuffle"],
            seed=config["seed"],
        )
    )
    if not samples:
        raise ValueError("dataset contains no samples")

    sandbox = DockerSandboxExecutor(_sandbox_config(config))
    sandbox_environment = sandbox.preflight()

    model = request.get("model_name")
    if not isinstance(model, str) or not model:
        model = OpenAIClient(
            config["service_url"],
            api_key=config["api_key"],
            timeout=config["request_timeout"],
        ).discover_model()
    client = OpenAIClient(
        config["service_url"],
        model=model,
        api_key=config["api_key"],
        timeout=config["request_timeout"],
    )
    sandbox_semaphore = threading.BoundedSemaphore(config["sandbox_concurrency"])
    with ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
        futures = [
            pool.submit(
                collect_sample,
                client,
                sandbox,
                sandbox_semaphore,
                sample,
                config,
            )
            for sample in samples
        ]
        sample_records = [future.result() for future in futures]

    counts = {
        "total": len(sample_records),
        "passed": sum(1 for record in sample_records if record["passed"]),
        "parse_failed": sum(
            1 for record in sample_records if record["status"] == "parse_failed"
        ),
        "service_failed": sum(
            1 for record in sample_records if record["status"] == "service_failed"
        ),
        "sandbox_failed": sum(
            1 for record in sample_records if record["status"] == "sandbox_failed"
        ),
    }
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "success",
        "metrics": {"samples": sample_records, "counts": counts},
        "metadata": {
            "measurement": MEASUREMENT,
            "protocol": PROTOCOL,
            "transport": TRANSPORT,
            "service_url": config["service_url"],
            "model": model,
            "dataset": "HumanEval",
            "dataset_path": config["dataset_path"],
            "dataset_sha256": _file_sha256(dataset_file),
            "split": config["split"],
            "sample_count": counts["total"],
            "eval_mode": "gen",
            "prompt_format": "base",
            "prompt_version": config["prompt_version"],
            "samples_per_task": 1,
            "pass_k": [1],
            "temperature": 0.0,
            "max_tokens": config["max_tokens"],
            "stop": config["stop"],
            "max_concurrency": config["max_concurrency"],
            "sandbox_concurrency": config["sandbox_concurrency"],
            "sandbox_protocol": SANDBOX_PROTOCOL,
            "sandbox": sandbox_environment,
            "scorer_version": SCORER_VERSION,
        },
        "artifacts": {},
    }


def failure_result(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "luban-meter.raw/v1",
        "status": "failed",
        "metrics": {},
        "metadata": {
            "measurement": MEASUREMENT,
            "protocol": PROTOCOL,
            "transport": TRANSPORT,
            "sandbox_protocol": SANDBOX_PROTOCOL,
        },
        "artifacts": {},
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main() -> None:
    args = parse_args()
    try:
        request, parameters = load_request(args.request)
        result = run_benchmark(request, parameters)
    except Exception as exc:  # noqa: BLE001
        result = failure_result(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
