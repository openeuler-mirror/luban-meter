"""Public defaults preserve explicit overrides and non-chat compatibility."""

import json

import httpx
import pytest

from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
)
from luban_meter.cli import _build_parser
from luban_meter.core.planning import build_plan


@pytest.mark.parametrize("command", ["plan", "run"])
@pytest.mark.parametrize("enabled", [False, True])
def test_thinking_metrics_cli_boolean_flags(command, enabled):
    flag = "--thinking-metrics" if enabled else "--no-thinking-metrics"
    parser, _ = _build_parser()
    parsed = parser.parse_args([command, flag])
    assert parsed.thinking_metrics is enabled


@pytest.mark.parametrize("scenario", ["online_serving", "multi_turn_serving"])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_phase_metrics_require_opt_in(tmp_path, scenario, enabled):
    dataset = tmp_path / "data.json"
    dataset.write_text('[]')
    parameters = {
        "scenario": scenario, "model": "mock", "dataset_path": str(dataset),
    }
    if enabled is not None:
        parameters["thinking_metrics"] = enabled
    plan = build_plan(parameters)
    saved = plan["tasks"][0]["cases"][0]["parameters"]
    expected = ["streaming_performance"]
    if enabled:
        expected.append("streaming_phase_timing")
    assert saved["metrics"] == expected
    assert saved["request_params"] == {}


@pytest.mark.parametrize("scenario", ["online_serving", "multi_turn_serving"])
@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_phase_metric_requires_switch(tmp_path, scenario, enabled):
    dataset = tmp_path / "data.json"
    dataset.write_text('[]')
    parameters = {
        "scenario": scenario, "model": "mock", "dataset_path": str(dataset),
        "metrics": ["streaming_phase_timing"], "thinking_metrics": enabled,
    }
    if not enabled:
        with pytest.raises(ValueError, match="requires thinking_metrics=true"):
            build_plan(parameters)
    else:
        plan = build_plan(parameters)
        assert plan["tasks"][0]["cases"][0]["parameters"]["metrics"] == [
            "streaming_phase_timing",
        ]


@pytest.mark.parametrize("scenario", ["online_serving", "multi_turn_serving"])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_thinking_metrics_opt_in_preserves_explicit_metric_selection(
    tmp_path, scenario, enabled,
):
    dataset = tmp_path / "data.json"
    dataset.write_text('[]')
    parameters = {
        "scenario": scenario, "model": "mock",
        "dataset_path": str(dataset), "metrics": ["streaming_performance"],
    }
    if enabled is not None:
        parameters["thinking_metrics"] = enabled
    plan = build_plan(parameters)
    saved = plan["tasks"][0]["cases"][0]["parameters"]
    assert saved["thinking_metrics"] is (enabled is True)
    assert saved["metrics"] == ["streaming_performance"]
    assert saved["request_params"] == {}


@pytest.mark.parametrize("enabled", ["true", 1, None])
def test_thinking_metrics_requires_boolean(enabled):
    with pytest.raises(ValueError, match="thinking_metrics"):
        build_plan({
            "scenario": "online_serving", "model": "mock",
            "thinking_metrics": enabled,
        })


@pytest.mark.parametrize(
    "scenario", ["online_serving", "ceval", "cmmlu", "gsm8k", "squad", "lcsts"]
)
@pytest.mark.parametrize(
    "endpoint", ["/v1/chat/completions", "/v1/completions"]
)
def test_task_defaults_depend_on_endpoint(tmp_path, scenario, endpoint):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n')
    plan = build_plan({
        "scenario": scenario, "model": "mock", "endpoint": endpoint,
        "dataset_path": str(dataset),
    })
    params = plan["tasks"][0]["cases"][0]["parameters"]
    assert params["request_params"] == {}
    assert "enable_thinking" not in params
    assert "thinking_protocol" not in params
    assert params["stop"] == []


@pytest.mark.parametrize("scenario", ["cmmlu", "wikitext", "humaneval"])
def test_non_generation_modes_do_not_inherit_thinking_default(
    tmp_path, scenario
):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n')
    settings = {"eval_mode": "ppl"} if scenario == "cmmlu" else {}
    plan = build_plan({
        "scenario": scenario, "model": "mock", "dataset_path": str(dataset),
        **settings,
    })
    params = plan["tasks"][0]["cases"][0]["parameters"]
    assert params["request_params"] == {}


@pytest.mark.parametrize("stop", [[], ["\n", "END"]])
@pytest.mark.parametrize("method", ["chat", "complete"])
def test_generation_transmits_empty_or_custom_stop(stop, method):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "text": "a\nb", "message": {"content": "a\nb"},
        }]})

    client = OpenAIClient("http://mock")
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        argument = [{"role": "user", "content": "Q"}] if (
            method == "chat"
        ) else "Q"
        response = getattr(client, method)(argument, stop=stop)
    finally:
        client.close()
    assert payloads[0]["stream"] is False
    assert payloads[0]["stop"] == stop
    assert response["text"] == "a\nb"


@pytest.mark.parametrize("method", ["chat", "complete"])
def test_low_level_generation_default_stop_is_explicit(method):
    client = OpenAIClient("http://mock")
    payloads = []

    def post(path, payload):
        payloads.append(payload)
        return {"choices": [{"text": "B", "message": {"content": "B"}}]}, 1

    client._post = post
    try:
        argument = [{"role": "user", "content": "Q"}] if (
            method == "chat"
        ) else "Q"
        getattr(client, method)(argument)
    finally:
        client.close()
    assert payloads[0]["stop"] == []


def test_logprobs_scoring_request_is_unchanged():
    client = OpenAIClient("http://mock")
    payloads = []

    def post(path, payload):
        payloads.append(payload)
        return {"choices": [{"logprobs": {
            "tokens": ["A"], "token_logprobs": [-0.2],
        }}]}, 1

    client._post = post
    try:
        client.completion_logprobs("Q")
    finally:
        client.close()
    assert payloads[0]["echo"] is True
    assert "stop" not in payloads[0]
    assert "thinking" not in payloads[0]
    assert "chat_template_kwargs" not in payloads[0]


@pytest.mark.parametrize("stop", ["\n", [1], None])
def test_online_invalid_stop_rejected_during_planning(stop):
    with pytest.raises(ValueError, match="stop"):
        build_plan({
            "scenario": "online_serving", "model": "mock", "stop": stop,
        })
