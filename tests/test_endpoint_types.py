"""Public interface names, preflight restrictions and HTTP wire behavior."""

import json

import pytest

from luban_meter.cli import _build_parser, main
from luban_meter.core.planning import build_plan
from tests.test_unified_runtime import Handler, service as mock_service


SCENARIOS = (
    "online_serving", "multi_turn_serving", "prefix_cache", "ceval",
    "cmmlu", "gsm8k", "squad", "lcsts", "wikitext", "humaneval",
)


def task_parameters(tmp_path, scenario, **overrides):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({
        "prompt": "private-endpoint-prompt", "question": "Q",
        "choices": ["a", "b", "c", "d"], "answer": "A",
    }) + "\n")
    parameters = {"scenario": scenario, "model": "mock"}
    if scenario == "prefix_cache":
        parameters["prefix_cache_target"] = 0.5
        parameters["kv_cache_block_size"] = 16
    else:
        parameters["dataset_path"] = str(dataset)
    return {**parameters, **overrides}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_default_interfaces_preserve_existing_behavior(tmp_path, scenario):
    plan = build_plan(task_parameters(tmp_path, scenario))
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    assert parameters["endpoint"] == (
        "chat" if scenario == "multi_turn_serving" else "completion"
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("endpoint", ["chat", "completion"])
def test_scenario_interface_restrictions(tmp_path, scenario, endpoint):
    parameters = task_parameters(tmp_path, scenario, endpoint=endpoint)
    supported = (
        ("chat",) if scenario == "multi_turn_serving" else
        ("completion",) if scenario in (
            "prefix_cache", "wikitext", "humaneval"
        ) else ("completion", "chat")
    )
    if endpoint not in supported:
        with pytest.raises(ValueError) as error:
            build_plan(parameters)
        assert f"endpoint must be one of {supported}" in str(error.value)
    else:
        plan = build_plan(parameters)
        assert plan["tasks"][0]["cases"][0]["parameters"][
            "endpoint"
        ] == endpoint


@pytest.mark.parametrize("scenario", ["ceval", "cmmlu"])
def test_ppl_requires_completion_interface(tmp_path, scenario):
    parameters = task_parameters(tmp_path, scenario, eval_mode="ppl")
    build_plan({**parameters, "endpoint": "completion"})
    with pytest.raises(
        ValueError, match="supports only endpoint='completion'",
    ):
        build_plan({**parameters, "endpoint": "chat"})


@pytest.mark.parametrize("command", ["plan", "run"])
def test_interface_help_uses_public_names(command, capsys):
    parser, _ = _build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args([command, "--help"])
    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "chat or completion" in help_text
    assert "HTTP paths are not accepted" in help_text


@pytest.fixture
def service(tmp_path):
    yield from mock_service.__wrapped__(tmp_path)


@pytest.mark.parametrize("source", ["cli", "json"])
@pytest.mark.parametrize("scenario", ["online_serving", "cmmlu"])
@pytest.mark.parametrize("endpoint", ["chat", "completion"])
def test_public_names_reach_correct_http_path_and_body(
    tmp_path, service, source, scenario, endpoint,
):
    parameters = task_parameters(
        tmp_path, scenario, base_url=service, endpoint=endpoint,
        profile="synchronous", num_requests=1,
    )
    if scenario == "online_serving":
        parameters["data"] = "jsonl"
    arguments = ["run", "--result-dir", str(tmp_path / "runs"),
                 "--no-summary", "--no-progress"]
    if source == "json":
        config = tmp_path / "tasks.json"
        config.write_text(json.dumps({"tasks": [parameters]}))
        arguments += ["--config", str(config)]
    else:
        for key, value in parameters.items():
            arguments += ["--" + key.replace("_", "-"), str(value)]
    assert main(arguments) == 0
    assert len(Handler.posts) == 1
    path, payload = Handler.posts[0]
    if endpoint == "chat":
        assert path == "/v1/chat/completions"
        assert payload["messages"][0]["role"] == "user"
        assert "prompt" not in payload
    else:
        assert path == "/v1/completions"
        assert isinstance(payload["prompt"], str)
        assert "messages" not in payload
    assert payload["stream"] is (scenario == "online_serving")
    directory, = (tmp_path / "runs").glob("run-*")
    result = json.loads((directory / "result.json").read_text())
    case = result["tasks"][0]["cases"][0]
    assert case["parameters"]["endpoint"] == endpoint
    raw = json.loads((directory / "tasks/task_1/raw/case_1.json").read_text())
    assert raw["parameters"]["endpoint"] == endpoint
    assert raw["requests"][0]["status"] == "success"
    logs = list((directory / "log").glob("luban-meter-*.log"))
    assert len(logs) == 1
    log = logs[0].read_text()
    assert "Preparation started" in log
    assert "CLI run completed: status=success" in log
    assert f"'endpoint': '{endpoint}'" in log
    assert "result.json" in log
    assert "private-endpoint-prompt" not in log


@pytest.mark.parametrize("source", ["cli", "json"])
@pytest.mark.parametrize("endpoint", [
    "/v1/completions", "/v1/chat/completions", "completions", "Chat", "",
])
def test_invalid_and_old_names_fail_before_requests(
    tmp_path, service, source, endpoint,
):
    parameters = task_parameters(
        tmp_path, "online_serving", base_url=service, endpoint=endpoint,
        data="jsonl",
    )
    log_dir = tmp_path / "log"
    arguments = ["run", "--result-dir", str(tmp_path / "runs"),
                 "--log-dir", str(log_dir), "--no-progress"]
    if source == "json":
        config = tmp_path / "tasks.json"
        config.write_text(json.dumps({"tasks": [parameters]}))
        arguments += ["--config", str(config)]
    else:
        for key, value in parameters.items():
            arguments += ["--" + key.replace("_", "-"), str(value)]
    assert main(arguments) == 2
    assert Handler.posts == []
    assert not list((tmp_path / "runs").glob("run-*/tasks"))
    log_file, = log_dir.glob("luban-meter-*.log")
    log = log_file.read_text()
    assert "endpoint must be one of ('completion', 'chat')" in log
    assert "task task_1" in log
    assert "Traceback (most recent call last)" in log


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("endpoint", [
    "/v1/completions", "/v1/chat/completions",
])
def test_old_paths_rejected_in_every_http_scenario(
    tmp_path, scenario, endpoint,
):
    with pytest.raises(ValueError, match="endpoint must be one of"):
        build_plan(task_parameters(tmp_path, scenario, endpoint=endpoint))
