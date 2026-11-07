"""Display-only tests over saved values, including missing and zero data."""

import copy
import io
import json

import pytest

from luban_meter import cli
from luban_meter.reporting.prefix_cache import cache_lines
from luban_meter.reporting.progress import TerminalProgress
from luban_meter.reporting.terminal import display_width, render_text
from tests.test_prefix_cache_reporting import mock_case


def case_for(target=0.5, shared=512, pairs=None, policy="strict"):
    case, _ = mock_case(
        "case_1", pairs or [(1024, shared)] * 2, target, shared,
        prefix_cache_warmup_policy=policy,
    )
    condition = case["metrics"]["prefix_cache_hit_rate"][
        "warmup"
    ]
    condition.update(
        warmup_policy=policy, warmup_completed=True,
        warmup_fully_matched=True, warmup_under_hit_prefixes=0,
        warmup_summary={
            phase: [{
                "prefix_id": i, "expected_cached_tokens": shared,
                "actual_cached_tokens": shared, "deviation_tokens": 0,
                "validation_state": "matched",
            } for i in range(2)]
            for phase in ("probe", "verify")
        },
    )
    return case


def payload(case):
    return {
        "kind": "task", "name": "prefix-half", "scenario": "prefix_cache",
        "category": "generation_performance",
        "status": "success", "cases": [case],
    }


def compact(case):
    return "\n".join(cache_lines(
        case["metrics"]["prefix_cache_hit_rate"],
        parameters=case["parameters"],
    ))


@pytest.mark.parametrize("target, shared", [
    (0, 0), (0.25, 256), (0.5, 512), (0.75, 768),
])
@pytest.mark.parametrize("policy", ["strict", "observe"])
def test_normal_default_seven_fields(target, shared, policy):
    case = case_for(target, shared, policy=policy)
    text = compact(case)
    assert len(text.splitlines()) == 8  # Title plus seven summary fields.
    assert f"目标命中率: {target * 100:g}%" in text
    assert f"实际命中率: {target * 100:g}%" in text
    assert "目标偏差: 0 pp" in text
    assert f"预热模式: {policy.title()}" in text
    assert "统计覆盖率: 100% (2/2)" in text
    assert "评测结论: 测量有效 / 目标达标" in text
    for hidden in (
        "理论共享比例", "已测子集", "构造偏差", "运行偏差", "Token 总数",
        "S=", "M=", "prefix_id", "verify", "probe", "unknown", "条件检查",
    ):
        assert hidden not in text
    if shared:
        assert "预热状态: 完成，符合预期 (2/2)" in text


def test_observe_under_hit_preserves_formal_data():
    case = case_for(policy="observe")
    metric = case["metrics"]["prefix_cache_hit_rate"]
    condition = metric["warmup"]
    condition.update(
        status="failed", warmup_status="observed_mismatch",
        warmup_fully_matched=False, warmup_under_hit_prefixes=1,
        reasons=["warmup_cache_mismatch"],
    )
    text = compact(case)
    assert "实际命中率: 50%" in text
    assert "完成，未完全符合预期 (2/2)；1 个前缀命中不足" in text
    assert "测量有效 / 目标达标" in text
    assert "预热命中与预期不符" in text
    assert "prefix_id" not in text
    report = render_text(payload(case))
    assert "TTFT" in report and "Output Token Throughput" in report


@pytest.mark.parametrize("cached, deviation", [
    (480, "-3.125 pp"), (528, "1.5625 pp"),
])
def test_runtime_mismatch_is_visible(cached, deviation):
    case = case_for(pairs=[(1024, cached)] * 2)
    text = compact(case)
    assert "理论共享比例: 50%" in text
    assert "目标偏差: " + deviation in text
    assert "与理论偏差:" in text
    assert "测量有效 / 目标未达标" in text


@pytest.mark.parametrize("pairs, state", [
    ([(1024, None)] * 2, "不可测"),
    ([(1024, -1)] * 2, "无效"),
    ([(1024, 512), (1024, None)], "部分可测"),
])
def test_missing_and_invalid_never_show_zero_actual(pairs, state):
    text = compact(case_for(pairs=pairs))
    assert "实际命中率: 不可测" in text
    assert "实际命中率: 0%" not in text
    assert f"评测结论: {state}" in text
    assert "缓存统计" in text
    if state == "部分可测":
        assert "统计覆盖率: 50% (1/2)" in text
        assert "已测子集命中率（非完整 Case）: 50%" in text


def test_hundred_percent_construction_limit():
    text = compact(case_for(1, 1008))
    assert "目标命中率: 100%" in text
    assert "实际命中率: 98.4375%" in text
    assert "目标偏差: -1.5625 pp" in text
    assert "理论共享比例: 98.4375%" in text
    assert "受构造上限限制" in text
    assert "末块重算" in text


def test_token_coverage_and_length_anomaly():
    text = compact(case_for(pairs=[(1024, 512), (512, None)]))
    assert "统计覆盖率: 50% (1/2)" in text
    assert "Token 统计覆盖率: 66.6667%" in text
    assert "输入长度符合率: 50%" in text


def test_failed_requests_do_not_disguise_coverage_denominator():
    case = case_for()
    values = case["metrics"]["prefix_cache_hit_rate"]["values"]
    values["success_rate"]["value"] = 0.5
    text = compact(case)
    assert "正式请求成功率: 50%" in text
    assert "覆盖率分母为已提交正式请求" in text


def test_legacy_json_and_unknown_conditions():
    case = case_for()
    metric = case["metrics"]["prefix_cache_hit_rate"]
    metric["experimental_conditions"] = {
        **metric.pop("warmup"), "status": "unknown",
        "checks": {"deployment_controlled": None},
    }
    assert "预热模式: Strict" in compact(case)
    assert "完成，符合预期 (2/2)" in compact(case)
    assert "测量有效 / 目标达标" in compact(case)
    assert "条件未知" not in compact(case)


def test_verbose_keeps_diagnostics_and_input_is_immutable():
    case = case_for()
    original = payload(case)
    before = copy.deepcopy(original)
    normal = render_text(original)
    detailed = render_text(original, verbose=True)
    assert original == before
    assert "prefix-half" not in normal
    assert "前缀缓存评测（目标 50%）" in normal
    assert "任务 ID: prefix-half" in detailed
    for field in (
        "理论共享比例", "已测子集命中率", "构造偏差", "运行偏差",
        "Token 总数（S）", "Token 总数（M）", "prefix_id",
        "probe", "verify", "完整指标上下文", "denominators",
    ):
        assert field not in normal
        assert field in detailed


@pytest.mark.parametrize("width", [20, 40, 80, 120])
@pytest.mark.parametrize("verbose", [False, True])
def test_width_and_zero_values(width, verbose):
    text = render_text(payload(case_for(0, 0)), width=width, verbose=verbose)
    assert all(display_width(line) <= width for line in text.splitlines())
    case = case_for(0, 0)
    lines = "\n".join(cache_lines(
        case["metrics"]["prefix_cache_hit_rate"], verbose=verbose,
        parameters=case["parameters"],
    ))
    assert ("实际缓存命中率: 0 %" if verbose else "实际命中率: 0%") in lines


def test_cli_verbose_and_output_are_existing_mechanisms(tmp_path):
    path = tmp_path / "result.json"
    original = payload(case_for())
    path.write_text(json.dumps(original), encoding="utf-8")
    normal, detailed = tmp_path / "normal.txt", tmp_path / "verbose.txt"
    base = ["report", "--input", str(path), "--format", "text", "--all"]
    assert cli.main([*base, "--output", str(normal)]) == 0
    assert cli.main([*base, "--verbose", "--output", str(detailed)]) == 0
    assert "prefix_id" not in normal.read_text()
    assert "prefix_id" in detailed.read_text()
    assert json.loads(path.read_text()) == original


def test_progress_uses_target_numbers_and_keeps_task_identity(tmp_path):
    task = {
        "name": "prefix-half", "scenario": "prefix_cache",
        "cases": [
            {"id": "case_1", "parameters": {
                "prefix_cache_target": 0.25, "num_requests": 20,
            }},
            {"id": "case_2", "parameters": {
                "prefix_cache_target": 0.5, "num_requests": 20,
            }},
        ],
    }
    before = copy.deepcopy(task)
    plan = {"tasks": [task], "case_count": 2}
    stream = io.StringIO()
    progress = TerminalProgress(plan, stream=stream)
    progress("task_started", task=task, directory=tmp_path)
    assert "目标 25%" in stream.getvalue()
    assert "目标 50%" in stream.getvalue()
    assert "prefix-half" not in stream.getvalue()
    assert "目标 25%" in progress._case_line(0, "case_1")
    assert "目标 50%" in progress._case_line(0, "case_2")
    assert task == before


def test_strict_failure_default_shows_only_actionable_context():
    case = case_for()
    case.update(status="discarded", metrics={}, setup_diagnostic={
        "status": "failed", "phase": "verify", "prefix_id": 0,
        "warmup_policy": "strict", "warmup_completed": False,
        "warmup_fully_matched": False, "expected_cached_tokens": 512,
        "cached_tokens": 0, "reason": "cache_mismatch",
        "warmup_summary": {"probe": [{"prefix_id": 0}]},
    })
    text = render_text(payload(case))
    assert "缓存预热: Strict，未完成" in text
    assert "完整验证，前缀编号 0" in text
    assert "预期命中 512 Token，实际 0" in text
    assert '"warmup_summary"' not in text
    assert '"warmup_summary"' in render_text(payload(case), verbose=True)



def test_zero_capability_mismatch_cannot_look_passed():
    case = case_for(0, 0, policy="observe")
    condition = case["metrics"]["prefix_cache_hit_rate"][
        "warmup"
    ]
    condition.update(
        status="failed", warmup_fully_matched=False,
        warmup_status="observed_mismatch",
        warmup_summary={"capability": [{
            "prefix_id": 0, "expected_cached_tokens": 0,
            "actual_cached_tokens": 16, "deviation_tokens": 16,
            "validation_state": "over_hit",
        }]},
    )
    text = compact(case)
    assert "实际命中率: 0%" in text
    assert "0% 能力检查出现非零命中" in text
    assert "能力检查通过" not in text
    assert "测量有效 / 目标达标" in text
    condition["warmup_status"] = "not_applicable"
    assert "能力检查通过" not in compact(case)
