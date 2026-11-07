"""Hand-computable formal request sets and independent cache judgments."""

import copy
import json

import pytest

from luban_meter.benchmarking.generation_performance import (
    prefix_cache_metrics,
)
from luban_meter.core.planning import build_plan
from luban_meter.core.metric import compute_metrics
from luban_meter.reporting.data import validate_result
from tests.test_prefix_cache_warmup import (
    Handler, PrefixHandler, execute, task,
    service as _service,
)


@pytest.fixture
def service(tmp_path):
    yield from _service.__wrapped__(tmp_path)


def facts(pairs=((1024, 512),), target=0.5, shared=512, **parameters):
    requests = []
    for i, (prompt, cached) in enumerate(pairs):
        usage = {"prompt_tokens": prompt}
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        requests.append({
            "request_id": f"request-{i}", "status": "success",
            "response": {
                "status": "success", "input_tokens": prompt, "usage": usage,
                "prefix_cache": {"prefix_id": i % 2},
            },
        })
    return {
        "parameters": {
            "num_requests": len(pairs), "input_tokens": 1024,
            "prefix_cache_target": target, "prefix_count": 2,
            **parameters,
        },
        "metadata": {
            "prefix_cache": {"shared_prefix_tokens": shared},
            "cache_warmup": (
                {"status": "success", "seed_completed": 2,
                 "probe_completed": 2, "verify_completed": 2}
                if shared else {"status": "not_applicable",
                                "phase": "capability", "cached_tokens": 0}
            ),
        },
        "requests": requests, "samples": [],
        "collection_complete": True,
        "window": {"admitted_requests": len(pairs)},
    }


def compute(data):
    result = prefix_cache_metrics.PrefixCacheMetric().compute(data)
    json.dumps(result, allow_nan=False)
    return result, {k: v["value"] for k, v in result["values"].items()}


@pytest.mark.parametrize("target, shared", [
    (0, 0), (0.25, 256), (0.5, 512), (0.75, 768), (1, 1008),
])
def test_complete_targets(target, shared):
    result, values = compute(facts(
        [(1024, shared)] * 4, target, shared,
        prefix_cache_environment="controlled",
    ))
    assert values["actual_hit_rate"] == shared / 1024
    assert values["runtime_deviation"] == 0
    assert values["construction_deviation"] == shared / 1024 - target
    assert result["measurement"]["status"] == "valid"
    assert result["warmup"]["warmup_fully_matched"] is True
    goal = result["target_attainment"]
    assert goal["status"] == ("construction_limited" if target == 1 else
                              "passed")
    assert goal["exact_target_reached"] is (target != 1)


def test_review_example():
    result, v = compute(facts([
        (1024, 512), (1024, 480), (1024, 512), (1024, 496),
    ]))
    assert v["total_cached_tokens"] == 2000
    assert v["total_prompt_tokens"] == 4096
    assert v["actual_hit_rate"] == 0.48828125
    assert v["runtime_deviation"] == v["target_deviation"] == -0.01171875
    assert v["below_expected_requests"] == 2
    assert result["measurement"]["status"] == "valid"
    assert "experimental_conditions" not in result
    assert result["target_attainment"]["status"] == "failed"
    assert result["diagnostics"]["eviction"] == "suspected"


def test_token_weighting_and_length_conditions():
    result, v = compute(facts([(100, 50), (300, 0)]))
    assert v["actual_hit_rate"] == 0.125
    assert v["input_length_conformance_rate"] == 0
    assert result["measurement"]["status"] == "valid"
    assert "experimental_conditions" not in result


@pytest.mark.parametrize("bad, state", [
    (None, "partial"), (-1, "invalid"), (True, "invalid"),
    (512.0, "invalid"), (1025, "invalid"),
])
def test_incomplete_pair_coverage(bad, state):
    result, v = compute(facts([(1024, 512), (1024, bad)]))
    assert v["actual_hit_rate"] is None
    assert v["runtime_deviation"] is v["target_deviation"] is None
    assert v["observed_subset_hit_rate"] == 0.5
    assert v["request_coverage"] == v["token_coverage"] == 0.5
    assert v["total_cached_tokens"] == 512
    assert v["total_prompt_tokens"] == 2048
    assert result["measurement"]["status"] == state
    assert result["target_attainment"]["status"] == "unknown"


def test_all_missing_and_unknown_prompt_denominator():
    data = facts([(1024, None), (1024, None)])
    result, v = compute(data)
    assert result["measurement"]["status"] == "unavailable"
    assert v["request_coverage"] == v["token_coverage"] == 0
    assert v["actual_hit_rate"] is v["total_cached_tokens"] is None
    data["requests"][1]["response"]["usage"] = None
    _, v = compute(data)
    assert v["total_prompt_tokens"] is v["token_coverage"] is None
    assert v["input_length_conformance_rate"] is None


def test_failed_requests_excluded_but_experiment_fails():
    data = facts([(1024, 512), (1024, 1024)])
    data["requests"][1]["status"] = "failed"
    data["requests"][1]["response"]["status"] = "failed"
    result, v = compute(data)
    assert v["actual_hit_rate"] is None
    assert v["observed_subset_hit_rate"] == 0.5
    assert v["success_rate"] == 0.5
    assert v["total_cached_tokens"] == 512
    assert v["total_prompt_tokens"] == 1024
    assert result["measurement"]["status"] == "partial"
    assert result["target_attainment"]["status"] == "unknown"
    assert v["request_coverage"] == 0.5
    assert v["token_coverage"] is None
    assert v["successful_request_coverage"] == 1
    assert "experimental_conditions" not in result
    data["requests"][0]["status"] = "failed"
    result, v = compute(data)
    assert v["actual_hit_rate"] is None
    assert v["request_coverage"] == 0
    assert v["success_rate"] == 0
    assert result["measurement"]["status"] == "unavailable"


def test_missing_requests_and_incomplete_collection():
    data = facts([(1024, 512), (1024, 512)])
    data["requests"].pop()
    result, v = compute(data)
    assert v["actual_hit_rate"] is None
    assert v["observed_subset_hit_rate"] == 0.5
    assert v["incomplete_requests"] == 1
    assert result["measurement"]["status"] == "partial"
    data["window"]["admitted_requests"] = 1
    result, v = compute(data)
    # Measurement covers submitted successes; plan completion is independent.
    assert v["actual_hit_rate"] == 0.5
    assert v["submitted_requests"] < v["planned_requests"]
    data["collection_complete"] = False
    _, v = compute(data)
    assert v["actual_hit_rate"] is None


def test_samples_and_nonformal_rows_never_counted():
    data = facts()
    data["samples"] = [{"facts": {"usage": {"cached_tokens": 100000}}}] * 10
    warmup = copy.deepcopy(data["requests"][0])
    warmup["phase"] = "cache_warmup"
    data["requests"].extend([warmup, {**warmup, "phase": "warmup"}])
    _, v = compute(data)
    assert v["total_cached_tokens"] == 512
    assert v["successful_requests"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "count", "identity"])
def test_ambiguous_facts_are_invalid(problem):
    data = facts()
    if problem == "duplicate":
        data["requests"] *= 2
    elif problem == "count":
        data["requests"][0]["response"]["input_tokens"] = 999
    else:
        data["requests"][0].pop("request_id")
    result, v = compute(data)
    assert v["actual_hit_rate"] is None
    assert result["measurement"]["status"] == "invalid"


def test_above_theory_retained_and_tolerance_applies_to_runtime_only():
    data = facts([(1024, 528)])
    result, v = compute(data)
    assert v["actual_hit_rate"] == 0.515625
    assert v["above_expected_requests"] == 1
    assert result["target_attainment"]["above_theoretical"] is True
    assert result["target_attainment"]["status"] == "failed"
    data["parameters"]["prefix_cache_runtime_tolerance"] = 0.02
    result, _ = compute(data)
    assert result["target_attainment"]["status"] == "passed"
    assert not result["target_attainment"]["exact_target_reached"]


@pytest.mark.parametrize("environment", [
    "unknown", "controlled", "interfered",
])
def test_environment_annotation_does_not_change_metrics(environment):
    data = facts(prefix_cache_environment=environment)
    result, _ = compute(data)
    assert result == compute(facts())[0]
    data["metadata"]["cache_warmup"]["verify_completed"] = 1
    result, _ = compute(data)
    assert result["measurement"]["status"] == "valid"
    assert result["warmup"]["warmup_fully_matched"] is False


def test_zero_denominator_and_group_rates():
    result, v = compute(facts([(0, 0)]))
    assert v["actual_hit_rate"] is v["token_coverage"] is None
    assert v["request_coverage"] == 1
    result, _ = compute(facts([(1024, 512), (1024, 0)]))
    groups = result["diagnostics"]["prefix_groups"]
    assert groups["0"]["observed_subset_hit_rate"] == 0.5
    assert groups["1"]["observed_subset_hit_rate"] == 0


def test_registration_compatibility_and_old_json():
    with pytest.raises(ValueError, match="requires"):
        build_plan({
            "scenario": "online_serving", "model": "mock",
            "metrics": ["prefix_cache_hit_rate"],
        })
    old = {"kind": "task", "name": "old", "category": "generation_performance",
           "status": "success", "cases": [{"id": "case_1", "status": "success",
                                         "metrics": {}}]}
    assert validate_result(old) == [old]
    data = facts()
    result = compute_metrics(
        ["prefix_cache_hit_rate"], {"prefix_cache_usage"}, data,
    )
    assert result["prefix_cache_hit_rate"]["measurement"]["status"] == "valid"


@pytest.mark.parametrize("mode", ["short", "missing", "invalid"])
def test_live_mock_persistence_and_no_refill(service, tmp_path, mode):
    PrefixHandler.mode, PrefixHandler.fail_at = mode, 9
    result, directory = execute(tmp_path, task(service))
    assert len(Handler.posts) == 16  # Nine setup plus seven formal.
    metric = result["metrics"]["prefix_cache_hit_rate"]
    assert result["metrics"]["streaming_performance"]["status"] == "success"
    assert metric["values"]["successful_requests"]["value"] == 7
    assert metric["values"]["reporting_requests"]["value"] == (
        7 if mode == "short" else 6
    )
    saved = json.loads((directory / "raw/case_1.result.json").read_text())
    assert saved["metrics"]["prefix_cache_hit_rate"] == metric
    raw = json.loads((directory / "raw/case_1.json").read_text())
    assert {r["response"]["prefix_cache"]["prefix_id"]
            for r in raw["requests"]} == {0, 1, 2}
    assert metric["measurement"]["status"] == {
        "short": "valid", "missing": "partial", "invalid": "invalid",
    }[mode]


def test_token_coverage_is_weighted_and_facts_are_immutable():
    data = facts([(100, 50), (300, None)])
    before = copy.deepcopy(data)
    _, v = compute(data)
    assert v["request_coverage"] == 0.5
    assert v["token_coverage"] == 0.25
    assert v["observed_subset_hit_rate"] == 0.5
    assert data == before


def test_metric_exception_preserves_performance(
    service, tmp_path, monkeypatch,
):
    def broken(self, data):
        raise ValueError("test computation failure")

    monkeypatch.setattr(
        prefix_cache_metrics.PrefixCacheMetric, "compute", broken,
    )
    result, _ = execute(tmp_path, task(service))
    assert len(Handler.posts) == 16
    assert result["metrics"]["prefix_cache_hit_rate"]["status"] == "failed"
    assert result["metrics"]["streaming_performance"]["status"] == "success"


@pytest.mark.parametrize("updates", [
    {"prefix_cache_runtime_tolerance": -0.01},
    {"prefix_cache_runtime_tolerance": True},
    {"prefix_cache_runtime_tolerance": float("nan")},
    {"prefix_cache_runtime_tolerance": 1.1},
    {"prefix_cache_environment": "assumed"},
])
def test_invalid_metric_configuration(updates):
    with pytest.raises(ValueError):
        build_plan({
            "scenario": "prefix_cache", "model": "mock",
            "prefix_cache_target": 0.5, "kv_cache_block_size": 16,
            **updates,
        })
