"""Controlled token IDs, reproducibility and the worker's HTTP payload."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from luban_meter.benchmarking.generation_performance import prefix_cache
from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from tests.test_unified_runtime import Handler
from tests.test_prefix_cache_warmup import service as mock_service
from tests.tokenizer_fixture import make_tokenizer


def config(**updates):
    return {
        "scenario": "prefix_cache", "model": "mock", "input_tokens": 1024,
        "output_tokens": 2, "prefix_cache_target": 0.5, "prefix_count": 5,
        "num_requests": 100, "kv_cache_block_size": 16, "seed": 42,
        **updates,
    }


def parameters(**updates):
    return build_plan(config(**updates))["tasks"][0]["cases"][0]["parameters"]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    directory = tmp_path_factory.mktemp("prefix-tokenizer")
    wrapper = make_tokenizer(directory)
    wrapper.add_special_tokens({"pad_token": "A"})
    wrapper.save_pretrained(directory)
    return str(directory)


def prepare(tokenizer, **updates):
    context = SimpleNamespace(parameters=parameters(
        tokenizer=tokenizer, **updates,
    ))
    scenario = benchmarks.get("prefix_cache")()
    try:
        scenario.prepare(context)
    finally:
        scenario.close()
    return scenario


@pytest.mark.parametrize("rate, shared", [
    (0, 0), (0.25, 256), (0.5, 512), (0.75, 768), (1, 1008),
])
def test_full_workload(tokenizer, rate, shared):
    scene = prepare(tokenizer, prefix_cache_target=rate)
    rows = scene.generated["formal"]
    assert len(rows) == 100
    assert len(scene.prefixes) == 5
    layout = scene.input_metadata["prefix_cache"]
    assert layout == {
        "target_rate": rate, "shared_prefix_tokens": shared,
        "suffix_tokens": 1024 - shared,
        "theoretical_shared_ratio": shared / 1024,
        "construction_deviation": shared / 1024 - rate,
    }
    roots = {tuple(prefix[:16]) for prefix in scene.prefixes}
    assert len(roots) == (5 if shared else 1)
    special = set(scene.tokenizer.all_special_ids)
    vocab = set(scene.tokenizer.get_vocab().values()) - special
    suffix_heads = set()
    for i, row in enumerate(rows):
        group = i % 5
        ids = row["prompt"]
        assert row["prefix_id"] == group
        assert len(ids) == row["target_input_tokens"] == 1024
        assert set(ids) <= vocab
        assert ids[:shared] == scene.prefixes[group]
        suffix_heads.add(tuple(ids[shared:shared + 16]))
    assert len(suffix_heads) == len(rows)


def test_inputs_independent_of_scheduling_and_warmups(tokenizer):
    opts = {"num_requests": 13, "input_tokens": 64}
    original = prepare(tokenizer, **opts)
    other = prepare(tokenizer, **opts, profile="concurrent",
                    max_concurrency=4, num_warmups=3)
    rate = prepare(tokenizer, **opts, request_rate=8)
    rows = original.generated["formal"]
    assert rows == other.generated["formal"] == rate.generated["formal"]
    assert [r["prefix_id"] for r in rows] == list(range(5)) * 2 + [0, 1, 2]
    assert rows != prepare(tokenizer, **opts, seed=43).generated["formal"]
    roots = {tuple(r["prompt"][:16]) for r in rows}
    assert all(tuple(r["prompt"][:16]) not in roots
               for r in other.generated["warmup"])


@pytest.mark.parametrize("total, rate, expected", [
    (17, 1, 16), (16, 0, 0), (3, 0, 0), (49, 0.5, 16),
])
def test_short_suffix_or_partial_block(tokenizer, total, rate, expected):
    scene = prepare(tokenizer, input_tokens=total, prefix_cache_target=rate,
                    num_requests=7)
    suffixes = set()
    for row in scene.generated["formal"]:
        assert row["shared_prefix_tokens"] == expected
        assert len(row["prompt"]) == total
        suffixes.add(tuple(row["prompt"][expected:]))
    assert len(suffixes) == 7


@pytest.mark.parametrize("overrides", [
    {"prefix_cache_target": -0.1}, {"prefix_cache_target": 1.1},
    {"prefix_cache_target": True}, {"prefix_cache_target": 0.001},
    {"input_tokens": 16, "prefix_cache_target": 1},
    {"prefix_cache_alignment": "strict", "prefix_cache_target": 1},
    {"prefix_cache_alignment": "strict", "prefix_cache_target": 0.333},
    {"prefix_cache_alignment": "ceil"}, {"prefix_count": 101},
    {"prefix_count": 0}, {"kv_cache_block_size": 0}, {"input_tokens": 0},
    {"kv_cache_block_size": 1.5}, {"endpoint": "/v1/chat/completions"},
    {"max_model_len": 1024}, {"data": "sharegpt"}, {"input_tokens_std": 2},
    {"request_params": {"truncate_prompt_tokens": 4}},
])
def test_invalid_plan(overrides):
    with pytest.raises(ValueError):
        parameters(**overrides)


def test_bounded_failure(monkeypatch, tokenizer):
    vocab = SimpleNamespace(
        all_special_ids=[0], vocab_size=3,
        get_vocab=lambda: {"special": 0, "only": 2, "outside": 5},
    )
    monkeypatch.setattr(prefix_cache, "load_tokenizer", lambda _: vocab)
    with pytest.raises(ValueError, match="64 attempts"):
        prepare(tokenizer)
    vocab.all_special_ids.append(2)
    with pytest.raises(ValueError, match="no usable"):
        prepare(tokenizer)


def test_matrix(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [config(
        prefix_cache_alignment="strict",
        cases={"matrix": {"prefix_cache_target": [0.25, 0.5, 0.75]}},
    )]}))
    plan = build_plan({}, path)
    assert [c["parameters"]["prefix_cache_target"]
            for c in plan["tasks"][0]["cases"]] == [0.25, 0.5, 0.75]


@pytest.fixture
def service(tmp_path):
    yield from mock_service.__wrapped__(tmp_path)


def test_worker_sends_ids_and_persists_source_once(service, tmp_path):
    plan = build_plan(config(
        tokenizer=Handler.tokenizer_path, base_url=service,
        input_tokens=64, num_requests=7, profile="synchronous",
    ))
    result, _ = RunCoordinator().run(plan, tmp_path / "runs")
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    raw = json.loads(Path(case["raw"]).read_text())
    assert raw["metadata"]["prefix_cache"]["shared_prefix_tokens"] == 32
    assert len(raw["samples"]) == len(Handler.posts) - 15 == 7
    for sample, (endpoint, body) in zip(raw["samples"], Handler.posts[15:]):
        assert endpoint == "/v1/completions"
        assert body["add_special_tokens"] is False
        assert body["prompt"] == sample["source"]["prompt"]
        assert all(type(token) is int for token in body["prompt"])
        assert "prompt" not in sample["facts"]
        assert sample["facts"]["input_tokens"] == 64


def test_decimal_target_and_last_token_limit_without_tokenizer():
    p = {
        "input_tokens": 100, "kv_cache_block_size": 1,
        "prefix_cache_target": 0.29, "prefix_cache_alignment": "strict",
    }
    assert prefix_cache.prefix_layout(p)["shared_prefix_tokens"] == 29
    layout = prefix_cache.prefix_layout({
        **p, "prefix_cache_target": 1, "prefix_cache_alignment": "floor",
    })
    assert (layout["shared_prefix_tokens"], layout["suffix_tokens"]) == (99, 1)
