"""Offline report fixtures use the approved calculator, never GPU results."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from luban_meter.benchmarking.generation_performance.common import (
    cache_usage,
)
from luban_meter.reporting.best_summary import performance_lines
from luban_meter.reporting.html import report_data, write_html
from luban_meter.reporting.prefix_cache import cache_eligible, display_value
from luban_meter.reporting.terminal import display_width, render_text
from tests.test_prefix_cache_metrics import compute, facts
from tests.test_report_analysis import browser_fixture

MOCK_NOTICE = "Mock 数据 · 非真实 GPU 压测 · cached_tokens 服务验证未完成"


def mock_case(name, pairs=((1024, 512),) * 2, target=0.5, shared=512,
              **parameters):
    raw = facts(pairs, target, shared, **{
        "prefix_cache_environment": "controlled", "model": "Mock-Qwen2.5-3B",
        "profile": "concurrent", "max_concurrency": 4, "output_tokens": 32,
        "kv_cache_block_size": 16, "base_url": "http://mock.invalid/v1",
        **parameters,
    })
    metric, _ = compute(raw)
    for request in raw["requests"]:
        response = request["response"]
        usage = cache_usage.parse_cache_usage(response["usage"])
        response["prefix_cache"].update({
            "shared_prefix_tokens": shared, "usage_state": usage.state,
            "prompt_tokens": usage.prompt_tokens,
            "cached_tokens": usage.cached_tokens, "reason": usage.reason,
            "hit_rate": (
                usage.cached_tokens / usage.prompt_tokens
                if usage.state == "valid" and usage.prompt_tokens else None
            ),
        })
        response["lengths"] = {
            "condition": ("conforming" if usage.prompt_tokens == 1024
                          else "unknown" if usage.prompt_tokens is None
                          else "nonconforming"),
        }
    values = {
        "request_view": {
            key: {"mean": value, "p99": value + 2, "unit": unit, "count": 2}
            for key, value, unit in (
                ("ttft", 10, "ms"), ("itl", 2, "ms"),
                ("tpot", 2, "ms/token"), ("e2el", 74, "ms"),
            )
        },
        "service_view": {
            key: {"value": value, "unit": unit}
            for key, value, unit in (
                ("output_token_throughput", 32, "token/s"),
                ("total_token_throughput", 1056, "token/s"),
                ("request_throughput", 1, "req/s"),
                ("total_requests", 2, "request"),
                ("successful_requests", 2, "request"),
                ("failed_requests", 0, "request"),
            )
        },
    }
    case = {
        "id": name, "status": "success", "collection_complete": True,
        "metrics_status": "complete", "parameters": raw["parameters"],
        "coverage": {"selected": 2, "complete": 2, "failed": 0,
                     "incomplete": 0, "not_started": 0},
        "metrics": {"prefix_cache_hit_rate": metric,
                    "streaming_performance": {"status": "success",
                                              "values": values}},
    }
    return case, raw


def report_fixture():
    cases = [
        mock_case(f"target-{int(target * 100)}",
                  [(1024, shared)] * 2, target, shared)[0]
        for target, shared in ((0, 0), (.25, 256), (.5, 512),
                               (.75, 768), (1, 1008))
    ]
    cases.extend([
        mock_case("partial", [(1024, 512), (1024, None)])[0],
        mock_case("missing", [(1024, None)] * 2)[0],
        mock_case("invalid", [(1024, -1)] * 2)[0],
        mock_case("length-mismatch", [(512, 256)] * 2)[0],
        mock_case("below-theory", [(1024, 480)] * 2)[0],
        mock_case("above-theory", [(1024, 528)] * 2)[0],
        mock_case("unknown", prefix_cache_environment="unknown")[0],
        mock_case("other-concurrency", max_concurrency=8)[0],
    ])
    failure, raw = mock_case("request-failure")
    raw["requests"][1]["status"] = "failed"
    failure["metrics"]["prefix_cache_hit_rate"] = compute(raw)[0]
    failure["status"] = "partial_failed"
    cases.append(failure)
    failed, _ = mock_case("warmup-failed")
    failed.update(status="discarded", metrics={}, setup_diagnostic={
        "status": "failed", "phase": "verify", "prefix_id": 1,
        "reason": "cache_mismatch", "expected_cached_tokens": 512,
        "cached_tokens": 0, "seed_completed": 2, "probe_completed": 2,
        "verify_completed": 1,
    })
    cases.append(failed)
    ordinary = browser_fixture()
    ordinary["id"] = MOCK_NOTICE
    ordinary["tasks"].insert(0, {
        "kind": "task", "name": "prefix-cache-mock",
        "scenario": "prefix_cache",
        "category": "generation_performance", "status": "partial_failed",
        "cases": cases,
    })
    return ordinary


def export_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = report_fixture()
    for index, pairs, shared in (
        (0, [(1024, 0)] * 2, 0),
        (2, [(1024, 512)] * 2, 512),
        (5, [(1024, 512), (1024, None)], 512),
        (6, [(1024, None)] * 2, 512),
    ):
        case = payload["tasks"][0]["cases"][index]
        _, raw = mock_case(
            case["id"], pairs, case["parameters"]["prefix_cache_target"],
            shared,
        )
        case["raw"] = case["id"] + "-requests.json"
        (directory / case["raw"]).write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8",
        )
    (directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (directory / "terminal.txt").write_text(
        render_text(payload, cases=["target-50", "partial", "missing"],
                    all_cases=True, width=100), encoding="utf-8",
    )
    return write_html(payload, directory / "report.html",
                      include_requests=True, raw_root=directory)


@pytest.mark.parametrize("value, key, expected", [
    (0, "actual_hit_rate", "0 %"),
    (None, "actual_hit_rate", "不可测"),
    (0.984375, "theoretical_shared_ratio", "98.4375 %"),
    (-0.015625, "construction_deviation", "-1.5625 pp"),
    (0.03125, "runtime_deviation", "3.125 pp"),
])
def test_ratio_display(value, key, expected):
    assert display_value({"value": value, "unit": "ratio"}, key) == expected


@pytest.mark.parametrize("name, expected", [
    ("target-0", True), ("target-100", True), ("partial", False),
    ("missing", False), ("invalid", False), ("length-mismatch", True),
    ("below-theory", True), ("above-theory", True), ("unknown", True),
    ("request-failure", False), ("warmup-failed", False),
])
def test_best_eligibility(name, expected):
    case = next(c for c in report_fixture()["tasks"][0]["cases"]
                if c["id"] == name)
    assert cache_eligible(case) is expected
    lines = "\n".join(performance_lines([({"name": "Mock"}, case)]))
    assert ("来源 " in lines) is expected


@pytest.mark.parametrize("width", [20, 80, 120])
def test_terminal_null_subset_decisions_and_width(width):
    payload = report_fixture()
    before = copy.deepcopy(payload)
    text = render_text(payload, all_cases=True, width=width, verbose=True)
    assert payload == before
    assert all(display_width(line) <= width for line in text.splitlines())
    if width >= 80:
        assert "实际缓存命中率: 0 %" in text
        assert "实际缓存命中率: 不可测" in text
        assert "已测子集命中率（非完整 Case）: 50 %" in text
        for label in ("测量有效性", "目标达标",
                      "三阶段预热", "construction_limited", "cache_mismatch",
                      "输入 Token 总数（S）", "缓存命中 Token 总数（M）"):
            assert label in text
        assert "-1.5625 pp" in text


def test_projection_keeps_calculated_values_and_old_json(tmp_path):
    payload = report_fixture()
    before = copy.deepcopy(payload)
    data = report_data(payload)
    assert data["result"] == before == payload
    html = export_fixture(tmp_path).read_text("utf-8")
    embedded = html.split('id="report-data">')[1].split("</script>")[0]
    saved = json.loads(embedded)
    assert saved["result"]["id"] == MOCK_NOTICE
    assert saved["request_details"][0][2]["status"] == "available"
    assert '<script src=' not in html
    assert html.count('class="report-page"') == 6
    legacy = browser_fixture()
    assert report_data(legacy)["result"] == legacy


def test_no_cache_arithmetic_in_javascript():
    source = Path("src/luban_meter/reporting/case_analysis.js").read_text()
    assert "cacheValue(c,'actual_hit_rate')" in source
    assert "valueText(c.hit_rate,'cache_ratio')" in source
    assert "cached_tokens /" not in source
    assert "cached_tokens/" not in source
    assert "observed_subset_hit_rate:'已测子集命中率（非完整 Case）'" in source


def test_salt_redaction_including_request_echo(tmp_path):
    payload = report_fixture()
    case = payload["tasks"][0]["cases"][0]
    secret = "salt-PRIVATE-DO-NOT-PUBLISH"
    case["parameters"]["cache_salt"] = secret
    case["raw"] = "raw.json"
    (tmp_path / "raw.json").write_text(json.dumps({
        "requests": [{"error": "echo " + secret}], "samples": [],
    }))
    original = copy.deepcopy(payload)
    assert secret not in render_text(payload, verbose=True, all_cases=True)
    assert secret not in json.dumps(report_data(
        payload, include_requests=True, raw_root=tmp_path,
    ))
    html = write_html(payload, tmp_path / "out.html", include_requests=True,
                      raw_root=tmp_path).read_text("utf-8")
    assert secret not in html
    assert payload == original


def test_javascript_saved_values_and_grouping(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required; run the .cjs test separately")
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(report_fixture()), encoding="utf-8")
    subprocess.run([
        node, str(Path(__file__).with_name("prefix_cache_report.unit.cjs")),
        str(fixture),
    ], check=True, capture_output=True)


if __name__ == "__main__":
    print(export_fixture(sys.argv[1]))


def test_historical_unknown_is_read_only_and_ranks_in_both_reports():
    payload = report_fixture()
    case = payload["tasks"][0]["cases"][2]
    metric = case["metrics"]["prefix_cache_hit_rate"]
    metric["experimental_conditions"] = {
        **metric.pop("warmup"), "status": "unknown",
        "checks": {"deployment_controlled": None},
        "environment_source": "operator_declaration",
    }
    original = copy.deepcopy(payload)
    text = render_text(payload, cases=[case["id"]], all_cases=True)
    assert "测量有效 / 目标达标" in text and "来源 前缀缓存评测" in text
    shown = report_data(payload)["result"]["tasks"][0]["cases"][2]
    projected = shown["metrics"]["prefix_cache_hit_rate"]
    assert "experimental_conditions" not in projected
    assert cache_eligible(shown)
    assert payload == original


@pytest.mark.parametrize("issue", [
    "quota", "failed_performance", "not_started",
])
def test_incomplete_cache_performance_cannot_win_any_card(issue):
    case, _ = mock_case("candidate")
    if issue == "quota":
        case["parameters"]["num_requests"] += 1
    elif issue == "failed_performance":
        case["metrics"]["streaming_performance"]["status"] = "failed"
    else:
        case["coverage"]["not_started"] = 1
    assert "来源 " not in "\n".join(performance_lines([({"name": "T"}, case)]))
