"""Observe policy uses real Mock HTTP usage; never GPU evidence."""

import copy
import json
from pathlib import Path

import pytest

from luban_meter.core.planning import build_plan
from luban_meter.reporting.html import report_data, write_html
from luban_meter.reporting.prefix_cache import cache_eligible
from luban_meter.reporting.terminal import render_text
from luban_meter.runtime.case_recovery import safe_setup_diagnostic
from tests.test_prefix_cache_metrics import compute, facts
from tests.test_prefix_cache_warmup import (
    Handler, PrefixHandler, assert_private, execute,
    service as _service, task,
)


@pytest.fixture
def service(tmp_path):
    yield from _service.__wrapped__(tmp_path)


def warmup_fact(result):
    return result["metrics"]["prefix_cache_hit_rate"][
        "warmup"
    ]


@pytest.mark.parametrize("policy", ["strict", "observe"])
def test_matching_warmup(service, tmp_path, policy):
    result, directory = execute(tmp_path, task(
        service, prefix_cache_warmup_policy=policy,
        prefix_cache_environment="controlled",
    ))
    warmup = warmup_fact(result)
    assert result["status"] == "success"
    assert warmup["warmup_status"] == "success"
    assert warmup["warmup_completed"] is True
    assert warmup["warmup_fully_matched"] is True
    assert warmup["warmup_under_hit_prefixes"] == 0
    assert warmup["warmup_policy"] == policy
    assert len(Handler.posts) == 16
    assert all(row["validation_state"] == "matched"
               for rows in warmup["warmup_summary"].values() for row in rows)
    assert_private(directory)


@pytest.mark.parametrize("mode, state, actual", [
    ("short", "under_hit", 16), ("zero", "under_hit", 0),
    ("over", "over_hit", 48),
])
@pytest.mark.parametrize("at, phase", [(3, "probe"), (6, "verify")])
def test_observe_keeps_valid_mismatches_and_no_retry(
    service, tmp_path, mode, state, actual, at, phase,
):
    PrefixHandler.mode, PrefixHandler.fail_at = mode, at
    result, directory = execute(tmp_path, task(
        service, prefix_cache_warmup_policy="observe",
        prefix_cache_environment="controlled",
    ))
    assert result["status"] == "success", result
    condition = warmup_fact(result)
    assert condition["warmup_completed"] is True
    assert condition["warmup_fully_matched"] is False
    assert condition["warmup_under_hit_prefixes"] == (
        1 if state == "under_hit" else 0
    )
    row = condition["warmup_summary"][phase][0]
    assert row == {
        "prefix_id": 0, "expected_cached_tokens": 32,
        "actual_cached_tokens": actual, "deviation_tokens": actual - 32,
        "validation_state": state,
    }
    assert condition["warmup_status"] == "observed_mismatch"
    assert condition["reasons"] == ["warmup_cache_mismatch"]
    assert cache_eligible(result)
    metric = result["metrics"]["prefix_cache_hit_rate"]
    assert metric["measurement"]["status"] == "valid"
    assert metric["values"]["actual_hit_rate"]["value"] == 0.5
    assert metric["target_attainment"]["status"] == "passed"
    assert metric["diagnostics"]["eviction_confirmed"] is False
    raw = json.loads(Path(result["raw"]).read_text())
    assert len(raw["requests"]) == raw["window"]["admitted_requests"] == 7
    assert len(Handler.posts) == 16  # 9 setup + 7 formal; no compensation.
    assert all(raw["metadata"]["cache_warmup"][key] == 3 for key in (
        "seed_completed", "probe_completed", "verify_completed",
    ))
    assert result["metrics"]["streaming_performance"]["values"][
        "request_view"
    ]["ttft"]["count"] == 7
    assert_private(directory)


@pytest.mark.parametrize("mode", ["missing", "invalid", "http", "error",
                                  "断流", "length"])
@pytest.mark.parametrize("at", [0, 3, 6])
def test_observe_does_not_relax_unusable_data(
    service, tmp_path, mode, at,
):
    PrefixHandler.mode, PrefixHandler.fail_at = mode, at
    result, directory = execute(tmp_path, task(
        service, prefix_cache_warmup_policy="observe",
    ))
    assert result["status"] == "discarded"
    assert result["setup_diagnostic"]["status"] == "failed"
    assert result["setup_diagnostic"]["warmup_completed"] is False
    assert len(Handler.posts) == at + 1
    assert not (directory / "raw/case_1.json").exists()
    assert_private(directory)


@pytest.mark.parametrize("policy", ["strict", "observe"])
def test_zero_target_nonzero_capability(service, tmp_path, policy):
    PrefixHandler.mode, PrefixHandler.fail_at = "over", 0
    result, directory = execute(tmp_path, task(
        service, prefix_cache_target=0,
        prefix_cache_warmup_policy=policy,
        prefix_cache_environment="controlled",
    ))
    if policy == "strict":
        assert result["status"] == "discarded"
        warmup = result["setup_diagnostic"]
        assert warmup["reason"] == "cache_mismatch"
        assert len(Handler.posts) == 1
    else:
        assert result["status"] == "success"
        warmup = warmup_fact(result)
        assert warmup["warmup_status"] == "observed_mismatch"
        assert warmup["warmup_completed"] is True
        assert result["metrics"]["prefix_cache_hit_rate"]["values"][
            "actual_hit_rate"
        ]["value"] == 0
        assert len(Handler.posts) == 8
        assert Handler.posts[0][1]["cache_salt"] != (
            Handler.posts[1][1]["cache_salt"]
        )
    assert warmup["warmup_fully_matched"] is False
    observation = warmup["warmup_summary"]["capability"][0]
    assert observation["expected_cached_tokens"] == 0
    assert observation["actual_cached_tokens"] == 16
    assert observation["validation_state"] == "over_hit"
    status = warmup.get("warmup_status", warmup.get("status"))
    assert status != "not_applicable"
    assert_private(directory)


def test_zero_observe_matched_capability(service, tmp_path):
    result, _ = execute(tmp_path, task(
        service, prefix_cache_target=0,
        prefix_cache_warmup_policy="observe",
    ))
    warmup = warmup_fact(result)
    assert warmup["warmup_status"] == "not_applicable"
    assert warmup["warmup_completed"] is True
    assert warmup["warmup_fully_matched"] is True
    assert warmup["warmup_summary"]["capability"][0][
        "actual_cached_tokens"
    ] == 0
    assert len(Handler.posts) == 8


def test_observe_formal_below_target_not_refilled(service, tmp_path):
    PrefixHandler.mode, PrefixHandler.fail_at = "short", 9
    result, _ = execute(tmp_path, task(
        service, prefix_cache_warmup_policy="observe",
        prefix_cache_environment="controlled",
    ))
    metric = result["metrics"]["prefix_cache_hit_rate"]
    assert len(Handler.posts) == 16
    assert metric["values"]["completed_requests"]["value"] == 7
    assert metric["values"]["actual_hit_rate"]["value"] == pytest.approx(
        (7 * 32 - 16) / (7 * 64)
    )
    assert metric["measurement"]["status"] == "valid"
    assert metric["target_attainment"]["status"] == "failed"


def test_observe_cancel_still_prevents_formal(service, tmp_path):
    PrefixHandler.on_request = staticmethod(
        lambda index: (tmp_path / "stop").touch() if index == 6 else None
    )
    result, directory = execute(tmp_path, task(
        service, prefix_cache_warmup_policy="observe",
    ))
    assert result["status"] == "discarded"
    assert result["setup_diagnostic"]["reason"] == "interrupted"
    assert len(Handler.posts) == 7
    assert_private(directory)


def test_observe_timeout_is_fatal(service, tmp_path, monkeypatch):
    import httpx
    from luban_meter.benchmarking.generation_performance import (
        generation_benchmarks,
    )

    def timeout(request):
        raise httpx.ReadTimeout("private failure")

    monkeypatch.setattr(
        generation_benchmarks, "create_http_client",
        lambda *args, **kwargs: httpx.Client(
            transport=httpx.MockTransport(timeout),
        ),
    )
    result, _ = execute(tmp_path, task(
        service, prefix_cache_warmup_policy="observe",
    ))
    assert result["status"] == "discarded"
    assert result["setup_diagnostic"]["reason"] == "ReadTimeout"


def test_new_setup_fields_are_sanitized():
    diagnostic = safe_setup_diagnostic({
        "status": "observed_mismatch", "warmup_policy": "observe",
        "warmup_completed": True, "warmup_fully_matched": False,
        "warmup_under_hit_prefixes": 1,
        "cache_salt": "secret", "prompt": [123],
        "warmup_summary": {"probe": [{
            "prefix_id": 0, "expected_cached_tokens": 32,
            "actual_cached_tokens": 0, "deviation_tokens": -32,
            "validation_state": "under_hit", "cache_salt": "secret",
            "prompt": [123], "reason": "unsafe reason",
        }], "unsafe phase": [{"cache_salt": "secret"}]},
    })
    assert "secret" not in json.dumps(diagnostic)
    assert "unsafe" not in json.dumps(diagnostic)
    assert diagnostic["warmup_summary"]["probe"][0][
        "deviation_tokens"
    ] == -32


def test_two_independent_metric_judgments_and_old_facts():
    data = facts([(1024, 256)] * 4,
                 prefix_cache_environment="controlled")
    before = copy.deepcopy(data)
    old, old_values = compute(data)
    assert data == before
    data["metadata"]["cache_warmup"].update({
        "status": "observed_mismatch", "warmup_policy": "observe",
        "warmup_completed": True, "warmup_fully_matched": False,
        "warmup_under_hit_prefixes": 1,
        "warmup_summary": {},
    })
    result, values = compute(data)
    assert values == old_values
    assert result["measurement"]["status"] == old["measurement"]["status"]
    assert result["measurement"]["status"] == "valid"
    assert result["warmup"]["warmup_fully_matched"] is False
    assert result["target_attainment"]["status"] == "failed"
    assert result["diagnostics"]["eviction"] == "unknown"
    assert result["diagnostics"]["eviction_confirmed"] is False


def test_reporting_observe_keeps_formal_result(service, tmp_path):
    PrefixHandler.mode, PrefixHandler.fail_at = "zero", 6
    result, directory = execute(tmp_path, task(
        service, prefix_cache_warmup_policy="observe",
        prefix_cache_environment="controlled",
    ))
    payload = {
        "kind": "task", "name": "observe-mock", "status": "success",
        "category": "generation_performance", "scenario": "prefix_cache",
        "cases": [result],
    }
    text = render_text(payload, all_cases=True, width=120, verbose=True)
    for expected in (
        "缓存预热模式: observe", "预热完全符合: 否",
        "命中不足的前缀数量: 1", "actual=0", "under_hit",
        "实际缓存命中率: 50 %", "warmup_cache_mismatch",
    ):
        assert expected in text
    assert "来源 前缀缓存评测" in text
    data = report_data(payload, include_requests=True, raw_root=directory)
    assert data["result"] == payload
    assert len(data["request_details"][0][0]["requests"]) == 7
    html = write_html(payload, tmp_path / "observe.html",
                      include_requests=True, raw_root=directory).read_text()
    assert "预热模式" in html
    assert "预热 Probe / Verify 与能力检查实测" in html
    assert "warmup_cache_mismatch" in html
    assert_private(directory)


def test_policy_default_and_config_boundary():
    base = {
        "scenario": "prefix_cache", "model": "mock",
        "prefix_cache_target": 0.5, "input_tokens": 64,
        "kv_cache_block_size": 16,
    }
    planned = build_plan(base)
    assert planned["tasks"][0]["cases"][0]["parameters"][
        "prefix_cache_warmup_policy"
    ] == "observe"
    with pytest.raises(ValueError):
        build_plan({**base, "prefix_cache_warmup_policy": "ignore"})


def test_observe_warmup_and_formal_shortfall_are_independent(
    service, tmp_path,
):
    PrefixHandler.mode, PrefixHandler.fail_at = "zero", 3
    PrefixHandler.on_request = staticmethod(
        lambda index: setattr(PrefixHandler, "fail_at", 9)
        if index == 3 else None
    )
    result, _ = execute(tmp_path, task(
        service, prefix_cache_warmup_policy="observe",
        prefix_cache_environment="controlled",
    ))
    metric = result["metrics"]["prefix_cache_hit_rate"]
    assert metric["measurement"]["status"] == "valid"
    assert metric["warmup"]["warmup_fully_matched"] is False
    assert metric["target_attainment"]["status"] == "failed"
    assert metric["values"]["actual_hit_rate"]["value"] == pytest.approx(
        6 * 32 / (7 * 64)
    )
    assert len(Handler.posts) == 16
    assert metric["diagnostics"]["eviction_confirmed"] is False


def export_fixture(destination):
    """Write an explicitly Mock report for optional browser validation."""
    from tests.test_prefix_cache_reporting import mock_case

    case, raw = mock_case(
        "observe-under-hit", prefix_cache_warmup_policy="observe",
    )
    summary = {}
    for phase in ("probe", "verify"):
        summary[phase] = []
        for prefix in range(2):
            actual = 0 if phase == "verify" and prefix == 0 else 512
            summary[phase].append({
                "prefix_id": prefix, "expected_cached_tokens": 512,
                "actual_cached_tokens": actual,
                "deviation_tokens": actual - 512,
                "validation_state": "under_hit" if actual == 0 else "matched",
            })
    raw["metadata"]["cache_warmup"].update({
        "status": "observed_mismatch", "warmup_policy": "observe",
        "warmup_completed": True, "warmup_fully_matched": False,
        "warmup_under_hit_prefixes": 1, "warmup_summary": summary,
    })
    case["metrics"]["prefix_cache_hit_rate"] = compute(raw)[0]
    payload = {
        "kind": "task", "name": "Mock Observe — 非 GPU 实测",
        "scenario": "prefix_cache", "category": "generation_performance",
        "status": "success", "cases": [case],
    }
    write_html(payload, destination)
    return destination


if __name__ == "__main__":
    import sys

    print(export_fixture(Path(sys.argv[1])))
