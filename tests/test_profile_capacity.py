"""Profile-specific options resolve to shared runtime capacity facts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from luban_meter.runtime.load_profiles import Profile
from luban_meter.runtime.request_scheduler import RequestScheduler
from luban_meter.runtime.scheduling_strategies import ImmediateStrategy
from tests.test_extension_worker import extension as _extension
from tests.test_unified_runtime import Handler
from tests.test_unified_runtime import service as _service

PROFILE_SOURCE = '''
from luban_meter.api import Parameter, Profile, profiles
from luban_meter.runtime.scheduling_strategies import ImmediateStrategy

@profiles.register("serial_fixture")
class SerialProfile(Profile):
    def create_strategy(self):
        return ImmediateStrategy()

@profiles.register("streams_fixture")
class StreamsProfile(SerialProfile):
    parameters = {"streams": Parameter(int, 3, minimum=1)}

    @property
    def workers(self):
        return self.parameters["streams"]
'''


@pytest.fixture(scope="module")
def extension(tmp_path_factory):
    environment = _extension.__wrapped__(tmp_path_factory)
    package = Path(environment["PYTHONPATH"]) / "luban_meter/benchmarking"
    (package / "profile_fixture.py").write_text(PROFILE_SOURCE)
    with (package / "__init__.py").open("a") as stream:
        stream.write("\nfrom . import profile_fixture\n")
    return environment


@pytest.fixture
def service(tmp_path):
    yield from _service.__wrapped__(tmp_path)


def test_scheduler_snapshots_profile_capacity():
    class MutableProfile(Profile):
        def create_strategy(self):
            return ImmediateStrategy()

        @property
        def workers(self):
            return self.parameters["streams"]

    parameters = {"streams": 3}
    scheduler = RequestScheduler(
        MutableProfile(parameters), parameters, lambda: False, lambda *a: None
    )
    parameters["streams"] = 0
    assert scheduler.capacity == 3
    assert scheduler.run(lambda ticket: 7, sample_id="0") == 7


@pytest.mark.parametrize("via_config", [False, True])
@pytest.mark.parametrize(
    "profile, options, capacity",
    [
        ("constant", {"request_rate": 10, "max_concurrency": 3}, 3),
        ("poisson", {"request_rate": 10, "max_concurrency": 3}, 3),
        ("gamma", {"request_rate": 10, "max_concurrency": 3,
                   "burstiness": 0.5}, 3),
        ("concurrent", {"max_concurrency": 3}, 3),
        ("throughput", {"max_concurrency": 3}, 3),
        ("synchronous", {}, 1),
        ("serial_fixture", {}, 1),
        ("streams_fixture", {"streams": 3}, 3),
    ],
)
def test_profile_cli_worker_metrics(
    service, extension, tmp_path, via_config, profile, options, capacity
):
    parameters = {
        "scenario": "online_serving",
        "model": "mock",
        "base_url": service,
        "tokenizer": Handler.tokenizer_path,
        "profile": profile,
        "input_tokens": 8,
        "output_tokens": 3,
        "num_requests": 1,
        "http_max_connections": capacity + 2,
        "http_keepalive_expiry": 7.5,
        **options,
    }
    output = tmp_path / "results"
    logs = tmp_path / "logs"
    command = [
        sys.executable, "-m", "luban_meter.cli", "run",
        "--result-dir", str(output), "--log-dir", str(logs),
    ]
    if via_config:
        config = tmp_path / "tasks.json"
        config.write_text(json.dumps({"tasks": [parameters]}))
        command.extend(["--config", str(config)])
    else:
        for name, value in parameters.items():
            command.extend(["--" + name.replace("_", "-"), str(value)])
    completed = subprocess.run(
        command, env=extension, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    (result_path,) = output.glob("*/result.json")
    result = json.loads(result_path.read_text())
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success"
    assert case["coverage"]["complete"] == 1
    assert case["window"]["request_capacity"] == capacity
    assert case["window"]["peak_requests"] == 1
    facts = json.loads(Path(case["raw"]).read_text())
    assert facts["window"]["request_capacity"] == capacity
    settings = facts["metadata"]["http_client"]
    assert settings["max_connections"] == capacity + 2
    assert settings["max_keepalive_connections"] == capacity + 2
    assert settings["keepalive_expiry"] == 7.5
    assert settings["http2"] is False
    assert facts["requests"][0]["status"] == "success"
    assert facts["parameters"]["profile"] == profile
    assert "arrival_process" not in facts["parameters"]
    if profile.endswith("_fixture"):
        assert "max_concurrency" not in facts["parameters"]
    metric = case["metrics"]["streaming_performance"]
    assert metric["status"] == "success"
    values = metric["values"]["service_view"]
    assert values["maximum_request_concurrency"]["value"] == capacity
    assert values["peak_concurrent_requests"]["value"] == 1
    assert result_path.with_name("report.html").is_file()
    log_path, = logs.glob("luban-meter-*.log")
    log = log_path.read_text()
    assert "Resolved Case: scenario=online_serving" in log
    assert f"'profile': '{profile}'" in log
    assert "Worker initializing: scenario=online_serving" in log
    assert "Raw facts saved:" in log
    assert "Run finished: status=success result=" in log
