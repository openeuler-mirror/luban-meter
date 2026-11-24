"""Selected online metrics retain formulas, scope and report visibility."""

import copy
import json
import subprocess
import sys

import pytest

from luban_meter.core.metric import compute_metrics
from luban_meter.core.planning import bootstrap, build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.core.token_totals import (
    run_throughput,
    run_throughput_selected,
)
from luban_meter.reporting.best_summary import serving_metric
from luban_meter.reporting.html import report_data, run_summary_html
from luban_meter.reporting.terminal import render_text
from tests.test_multi_turn_serving import ChatHandler, service as chat_service
from tests.test_prefix_cache_warmup import (
    execute as execute_prefix,
    service as cache_service,
    task as cache_task,
)
from tests.test_streaming_metric import formal_facts, response, streaming

NAMES = (
    "ttft", "tpot", "itl", "e2el", "request_throughput",
    "input_token_throughput", "output_token_throughput",
    "total_token_throughput",
)


def calculate(names, facts):
    bootstrap()
    return compute_metrics(names, {"client_stream_timing"}, facts)


@pytest.mark.parametrize("selected", [
    ["ttft"], ["ttft", "stability_trends", "capacity_search"],
])
def test_step_respects_explicit_selection(selected):
    plan = build_plan({
        "scenario": "online_serving", "model": "mock", "profile": "step",
        "metrics": selected,
    })
    assert plan["tasks"][0]["cases"][0]["parameters"]["metrics"] == selected


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("condition", [
    "success", "missing_usage", "single_token", "missing_last_output",
    "partial_failed", "all_failed", "multi_turn",
])
def test_selected_statistic_matches_existing_formula(name, condition):
    rows = [response(0), response(1), response(2)]
    rows[1].update(output_tokens=7, input_tokens=200)
    if condition == "missing_usage":
        rows[1] = response(1, usage=False)
    elif condition == "single_token":
        rows[1].update(output_tokens=1, last_output_latency_ms=11)
    elif condition == "missing_last_output":
        rows[1].pop("last_output_latency_ms")
    elif condition == "partial_failed":
        rows[1] = response(1, status="failed")
    elif condition == "all_failed":
        rows = [response(i, status="failed") for i in range(3)]
    facts = formal_facts(rows, multi_turn=condition == "multi_turn")
    original = copy.deepcopy(facts)
    expected = streaming(facts)
    actual = calculate([name], facts)[name]
    view = "request_view" if name in NAMES[:4] else "service_view"
    assert actual["status"] == expected["status"]
    assert actual["values"][view][name] == expected["values"][view][name]
    if view == "request_view":
        assert set(actual["values"][view]) == {name}
    else:
        assert "request_view" not in actual["values"]
    for other in set(NAMES) - {name}:
        assert other not in actual["values"]["service_view"]
    assert "turn_view" not in actual["values"]
    assert facts == original


@pytest.mark.parametrize("scenario", [
    "online_serving", "multi_turn_serving", "prefix_cache",
])
def test_all_online_scenarios_accept_independent_metrics(tmp_path, scenario):
    dataset = tmp_path / "conversation.json"
    dataset.write_text("[]")
    parameters = {
        "scenario": scenario, "model": "mock", "tokenizer": "mock",
        "metrics": list(NAMES),
    }
    if scenario != "prefix_cache":
        parameters["dataset_path"] = str(dataset)
    else:
        parameters.update(prefix_cache_target=0.5, kv_cache_block_size=16)
    plan = build_plan(parameters)
    assert plan["tasks"][0]["cases"][0]["parameters"]["metrics"] == list(NAMES)


@pytest.mark.parametrize("scenario", [
    "offline_vllm_engine", "offline_vllm_async", "gsm8k",
])
def test_other_observation_boundaries_reject_online_metrics(
    tmp_path, scenario,
):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n')
    with pytest.raises(ValueError, match="requires.*client_stream_timing"):
        build_plan({
            "scenario": scenario, "model": "mock", "metrics": ["ttft"],
            "dataset_path": str(dataset),
        })


def saved_case(names):
    facts = formal_facts([response(0), response(1)])
    return {
        "id": "case_1", "status": "success", "collection_complete": True,
        "metrics_status": "complete", "coverage": {"complete": 2},
        "aggregate_run_throughput": bool(
            {"total_token_throughput", "streaming_performance"} & set(names)
        ),
        "parameters": {"metrics": list(names), "max_concurrency": 2},
        "metrics": calculate(names, facts),
    }


def task(case):
    return {
        "kind": "task", "name": "selected", "status": "success",
        "category": "generation_performance", "scenario": "online_serving",
        "cases": [case],
    }


def test_selected_providers_merge_without_order_or_input_mutation():
    case = saved_case(["ttft", "output_token_throughput"])
    original = copy.deepcopy(case)
    expected = serving_metric(case)
    case["metrics"] = dict(reversed(list(case["metrics"].items())))
    assert serving_metric(case)["values"] == expected["values"]
    case["metrics"] = dict(reversed(list(case["metrics"].items())))
    assert case == original


def test_group_and_individual_metric_keep_legacy_non_view_values():
    case = saved_case(["streaming_performance", "ttft"])
    combined = serving_metric(case)["values"]
    original = case["metrics"]["streaming_performance"]["values"]
    assert combined == original


def test_unavailable_selected_metric_remains_visible():
    case = saved_case(["ttft"])
    case["metrics"]["ttft"] = {
        "status": "unavailable", "reason": "no admitted requests",
    }
    text = render_text(task(case))
    assert "最低 TTFT P99: N/A" in text
    assert "最低 TPOT" not in text


@pytest.mark.parametrize("name", [
    "prefix_cache_hit_rate", "streaming_phase_timing",
])
def test_group_only_selection_omits_unselected_performance_placeholders(name):
    case = saved_case(["ttft"])
    case["parameters"]["metrics"] = [name]
    case["metrics"] = {name: {"status": "success", "values": {}}}
    text = render_text(task(case))
    assert "最低 TTFT" not in text
    assert "最低 TPOT" not in text
    assert "最大输出 Token 吞吐量" not in text


def test_unselected_metrics_absent_from_terminal_and_saved_html_rows():
    case = saved_case(["ttft"])
    payload = {
        "kind": "run", "id": "selected", "status": "success",
        "duration_s": 2, "tasks": [task(case)], "metrics": {},
    }
    text = render_text(payload).lower()
    assert "ttft" in text
    for name in NAMES[1:]:
        assert name not in text
    fragment = run_summary_html(payload)
    assert "total_token_throughput" not in fragment
    assert "run-hero" not in fragment
    paths = [row["metric"] for row in report_data(payload)["rows"]]
    assert any("ttft" in path for path in paths)
    assert not any(name in path for path in paths for name in NAMES[1:])
    assert not run_throughput_selected(payload["tasks"])


def test_mixed_run_aggregates_only_explicit_total_throughput_cases():
    chosen = saved_case(["total_token_throughput"])
    omitted = saved_case(["ttft"])
    tasks = [task(chosen), task(omitted)]
    result = run_throughput(tasks)
    assert run_throughput_selected(tasks)
    assert result["values"]["service_view"]["total_tokens"]["value"] == 206
    assert result["metadata"]["included_cases"][0]["task_index"] == 0
    assert result["metadata"]["skipped_cases"][0]["reason"] == (
        "total token throughput not selected"
    )


def test_cli_lists_and_accepts_independent_metric_names(tmp_path):
    command = [sys.executable, "-m", "luban_meter.cli", "--log-dir",
               str(tmp_path / "logs")]
    listed = subprocess.run(
        [*command, "metrics", "list"], check=True, capture_output=True,
        text=True, timeout=15,
    )
    assert set(NAMES) <= {line.split("\t")[0]
                          for line in listed.stdout.splitlines()}
    planned = subprocess.run(
        [*command, "plan", "--scenario", "online_serving", "--model", "mock",
         "--metrics", "ttft", "output_token_throughput"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    case = json.loads(planned.stdout)["tasks"][0]["cases"][0]
    parameters = case["parameters"]
    assert parameters["metrics"] == ["ttft", "output_token_throughput"]


@pytest.mark.parametrize("names,message", [
    (["ttft", "ttft"], "must be unique"),
    (["unknown_statistic"], "unknown metric"),
])
def test_invalid_metric_selection_rejected_before_execution(names, message):
    with pytest.raises(ValueError, match=message):
        build_plan({
            "scenario": "online_serving", "model": "mock", "metrics": names,
        })


@pytest.mark.parametrize("name", ["tpot", "itl", "output_token_throughput"])
def test_invalid_output_count_remains_a_computation_error(name):
    row = response()
    row["output_tokens"] = 0
    with pytest.raises(ValueError, match="output_tokens must be positive"):
        calculate([name], formal_facts([row]))


@pytest.fixture
def prefix_service(tmp_path):
    yield from cache_service.__wrapped__(tmp_path)


def test_prefix_cache_collects_with_independent_metrics(
    tmp_path, prefix_service,
):
    names = ["ttft", "request_throughput", "total_token_throughput"]
    result, _ = execute_prefix(
        tmp_path, cache_task(prefix_service, metrics=names),
    )
    assert result["status"] == "success"
    assert set(result["metrics"]) == set(names)
    assert result["metrics"]["ttft"]["values"]["request_view"]["ttft"][
        "count"
    ] == 7


@pytest.fixture
def service():
    yield from chat_service.__wrapped__()


@pytest.mark.parametrize("names,fail", [
    (["ttft"], False),
    (["ttft", "output_token_throughput"], False),
    (list(NAMES), False),
    (["ttft", "request_throughput"], True),
])
def test_real_task_worker_selection_and_unified_logs(
    tmp_path, service, names, fail,
):
    dataset = tmp_path / "conversations.json"
    dataset.write_text(json.dumps([{"conversations": [
        {"from": "human", "value": "first question"},
        {"from": "gpt", "value": "reference"},
        {"from": "human", "value": "second question"},
    ]}]))
    if fail:
        ChatHandler.fail_times = 1
    plan = build_plan({
        "scenario": "multi_turn_serving", "model": "mock",
        "api_key": "granular-selection-secret",
        "base_url": service, "dataset_path": str(dataset),
        "multi_num_sessions": 1, "multi_max_turns": 2,
        "profile": "concurrent", "max_concurrency": 1,
        "output_tokens": 3, "ignore_eos": False, "metrics": names,
    })
    result, directory = RunCoordinator().run(
        plan, tmp_path / "results", log_dir=tmp_path / "logs",
    )
    case = result["tasks"][0]["cases"][0]
    assert set(case["metrics"]) == set(names)
    assert bool(result["metrics"]) == ("total_token_throughput" in names)
    raw = json.loads((directory / "tasks/task_1/raw/case_1.json").read_text())
    assert len(raw["requests"]) == (1 if fail else 2)
    assert len(ChatHandler.posts) == len(raw["requests"])
    log_files = list((tmp_path / "logs").glob("*.log"))
    assert len(log_files) == 1
    log = log_files[0].read_text()
    assert "Run finished" in log and "Online metric samples" in log
    assert "granular-selection-secret" not in log
    for name in names:
        assert "Metric finished: " + name in log
    if fail:
        assert case["status"] == "failed"
        assert "Traceback" in log and "request=" in log
    else:
        assert case["status"] == "success"
