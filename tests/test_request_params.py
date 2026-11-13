"""Custom JSON fields reach inference without changing benchmark controls."""

import copy
import json

import httpx
import pytest

from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
)
from luban_meter.cli import main
from luban_meter.core.planning import build_plan
from luban_meter.runtime.request_params import (
    CORE_REQUEST_FIELDS,
    append_request_params,
    validate_request_params,
)

EXTENSIONS = [
    {},
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"thinking": {"type": "disabled"}},
    {"custom": {"values": [True, False, None, 1, 0.5, "中文"]}},
]


@pytest.mark.parametrize("extras", EXTENSIONS)
def test_cli_preserves_nested_json_types(extras, capsys):
    assert main([
        "plan", "--scenario", "online_serving", "--model", "mock",
        "--request-params", json.dumps(extras),
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    params = plan["tasks"][0]["cases"][0]["parameters"]
    assert params["request_params"] == extras


def test_task_json_replaces_cli_object_without_deep_merge(tmp_path):
    config = tmp_path / "tasks.json"
    first = {"chat_template_kwargs": {"enable_thinking": True}}
    config.write_text(json.dumps({"tasks": [
        {"name": "inherited"},
        {"name": "replace", "request_params": first},
        {"name": "clear", "request_params": {}},
    ]}))
    base = {"scenario": "online_serving", "model": "mock",
            "request_params": {"top_p": 0.8}}
    before = copy.deepcopy(base)
    plan = build_plan(base, config)
    assert base == before
    assert [t["cases"][0]["parameters"]["request_params"]
            for t in plan["tasks"]] == [{"top_p": 0.8}, first, {}]


@pytest.mark.parametrize("scenario, extra", [
    ("online_serving", {}),
    ("cmmlu", {"eval_mode": "gen", "endpoint": "chat"}),
    ("cmmlu", {"eval_mode": "ppl"}),
    ("gsm8k", {}),
    ("squad", {}),
    ("lcsts", {}),
    ("humaneval", {}),
    ("wikitext", {}),
])
def test_public_inference_scenarios_accept_extensions(
    tmp_path, scenario, extra
):
    data = tmp_path / "data.jsonl"
    data.write_text('{}\n')
    extensions = {"custom": {"enabled": True}}
    plan = build_plan({
        "scenario": scenario, "model": "mock", "dataset_path": str(data),
        "request_params": extensions, **extra,
    })
    assert plan["tasks"][0]["cases"][0]["parameters"][
        "request_params"
    ] == extensions


@pytest.mark.parametrize("key", sorted(CORE_REQUEST_FIELDS))
@pytest.mark.parametrize("scenario", ["online_serving", "cmmlu"])
def test_core_fields_rejected_before_execution(tmp_path, key, scenario):
    data = tmp_path / "data.jsonl"
    data.write_text('{}\n')
    message = "cannot override core fields: " + key
    with pytest.raises(ValueError, match=message):
        build_plan({
            "scenario": scenario, "model": "mock", "dataset_path": str(data),
            "request_params": {key: None},
        })


@pytest.mark.parametrize("value", [
    None, [], "{}", True, 1,
    {"nested": {1: "not a string key"}},
    {"nested": float("nan")}, {"nested": float("inf")},
    {"nested": {"value": b"bytes"}}, {"nested": (1, 2)},
])
def test_invalid_extension_values_rejected(value):
    with pytest.raises(ValueError, match="request_params"):
        build_plan({
            "scenario": "online_serving", "model": "mock",
            "request_params": value,
        })


@pytest.mark.parametrize("value", ['broken', '[]', 'null', '{"x":NaN}'])
def test_cli_rejects_invalid_json_objects(value, capsys):
    args = ["plan", "--scenario", "online_serving", "--model", "mock",
            "--request-params", value]
    if value == 'broken':
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
    else:
        assert main(args) != 0
    assert capsys.readouterr().err


@pytest.mark.parametrize("field, value", [
    ("thinking_protocol", "deepseek"), ("enable_thinking", False),
])
def test_removed_thinking_options_rejected(field, value):
    with pytest.raises(ValueError, match="unknown parameters: " + field):
        build_plan({"scenario": "online_serving", "model": "mock",
                    field: value})


def test_request_merging_copies_nested_data_and_rejects_collisions():
    extras = {"custom": {"values": [1]}}
    payload = {"model": "mock"}
    result = append_request_params(payload, extras)
    result["custom"]["values"].append(2)
    assert extras == {"custom": {"values": [1]}}
    assert payload == {"model": "mock"}
    with pytest.raises(ValueError, match="conflicts with request fields"):
        append_request_params({"custom": 1}, extras)
    assert validate_request_params(extras) == extras


@pytest.mark.parametrize("extras", EXTENSIONS)
@pytest.mark.parametrize("method", ["chat", "complete", "completion_logprobs"])
def test_extensions_reach_inference_and_reasoning_stays_separate(
    extras, method
):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "text": "答案：B",
            "message": {"reasoning_content": "答案：A", "content": "答案：B"},
            "logprobs": {"tokens": ["B"], "token_logprobs": [-0.1]},
            "finish_reason": "stop",
        }], "usage": {
            "prompt_tokens": 10, "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 6},
        }})

    service = OpenAIClient("http://mock", "mock", request_params=extras)
    service.http.close()
    service.http = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        argument = [{"role": "user", "content": "Q"}] if (
            method == "chat"
        ) else "Q"
        result = getattr(service, method)(argument)
    finally:
        service.close()
    for key, value in extras.items():
        assert payloads[0][key] == value
    for key in ("thinking", "chat_template_kwargs"):
        assert (key in payloads[0]) is (key in extras)
    assert payloads[0]["stream"] is False
    assert payloads[0]["model"] == "mock"
    if method == "chat":
        assert result["text"] == "答案：B"
        assert result["reasoning_text"] == "答案：A"
        assert result["output_tokens"] == 8
        assert result["reasoning_tokens"] == 6
    elif method == "completion_logprobs":
        assert payloads[0]["echo"] is True
        assert payloads[0]["logprobs"] == 1
        assert "stop" not in payloads[0]


def test_extensions_do_not_leak_into_tokenize_or_metadata():
    payloads = []
    extras = {"thinking": {"type": "enabled"}}

    def handle(request):
        payloads.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"tokens": [1, 2]})

    service = OpenAIClient("http://mock", request_params=extras)
    service.http.close()
    service.http = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        assert service.tokenize("Q") == [1, 2]
        service._post("/health", {})
    finally:
        service.close()
    assert all("thinking" not in body for _, body in payloads)


def test_nested_extensions_not_modified_by_request_runner():
    params = {"custom": {"enabled": True}}

    def runner(operation):
        return operation(None)

    def handle(request):
        assert json.loads(request.content)["custom"] == {"enabled": True}
        return httpx.Response(200, json={"choices": [{"text": "B"}]})

    service = OpenAIClient("http://mock", request_params=params,
                           request_runner=runner)
    service.http.close()
    service.http = httpx.Client(transport=httpx.MockTransport(handle))
    params["custom"]["enabled"] = False
    try:
        assert service.complete("Q")["text"] == "B"
    finally:
        service.close()
    assert service.request_params == {"custom": {"enabled": True}}
