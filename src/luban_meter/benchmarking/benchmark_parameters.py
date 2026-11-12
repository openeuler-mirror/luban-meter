"""Shared parameter vocabulary. Scenarios opt into appropriate groups."""

from luban_meter.core.parameters import Parameter

HTTP = {
    "backend": Parameter(str, "openai_http", choices=("openai_http",)),
    "base_url": Parameter(str, "http://127.0.0.1:8000", required=True),
    "model": Parameter(str, "", required=True),
    "endpoint": Parameter(
        str,
        "completion",
        choices=("completion", "chat"),
        help="API type: chat or completion (HTTP paths are not accepted)",
    ),
    "api_key": Parameter(str, "", secret=True),
    "http_max_connections": Parameter(
        int, None, minimum=1, matrix=True,
        help="HTTP connection limit; defaults to effective request capacity",
    ),
    "http_keepalive_expiry": Parameter(
        float, 5.0, minimum=0.0, matrix=True,
        help="HTTP idle connection lifetime in seconds (must be positive)",
    ),
    "temperature": Parameter(float, 0.0, choices=(0.0,)),
    "request_params": Parameter(
        dict, {}, help="Additional request body fields as a JSON object",
    ),
}
DATA = {
    "data": Parameter(str, "jsonl"),
    "dataset_path": Parameter(str, "", path=True, required=True),
    "max_samples": Parameter(int, None, minimum=1),
    "shuffle": Parameter(bool, False),
    "split": Parameter(str, "test"),
}
GENERATION = {
    "output_tokens": Parameter(int, 128, minimum=1, matrix=True),
    "stop": Parameter(list, [], item_type=str),
}
FEW_SHOT = {
    "few_shot": Parameter(int, 0, minimum=0),
    "few_shot_path": Parameter(str, "", path=True),
}
SANDBOX = {
    "sandbox_concurrency": Parameter(int, 4, minimum=1),
    "docker_host": Parameter(str, "unix:///var/run/docker.sock"),
    "docker_binary": Parameter(str, "docker"),
    "sandbox_image": Parameter(str, "luban-meter-humaneval-sandbox:v1"),
    "sandbox_runtime": Parameter(str, "runc"),
    "sandbox_timeout": Parameter(float, 3.0, minimum=0.001),
    "sandbox_startup_grace": Parameter(float, 5.0, minimum=0.001),
    "sandbox_memory_mb": Parameter(int, 256, minimum=1),
    "sandbox_cpus": Parameter(float, 1.0, minimum=0.001),
    "sandbox_pids_limit": Parameter(int, 32, minimum=1),
    "sandbox_tmpfs_mb": Parameter(int, 64, minimum=1),
    "sandbox_output_limit_bytes": Parameter(int, 65536, minimum=1),
    "allow_host_fallback": Parameter(bool, False, choices=(False,)),
}
