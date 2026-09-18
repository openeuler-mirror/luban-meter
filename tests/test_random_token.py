"""Random text, workload acceptance and independently seeded scheduling."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from luban_meter.benchmarking.generation_performance.common import tokenization
from luban_meter.benchmarking.generation_performance.common.lengths import (
    assess_lengths,
)
from luban_meter.core.planning import build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from tests.test_unified_runtime import Handler, serving
from tests.test_unified_runtime import service as _service
from tests.tokenizer_fixture import make_tokenizer


@pytest.fixture
def service(tmp_path):
    yield from _service.__wrapped__(tmp_path)


def parameters(**changes):
    return build_plan(
        {
            "scenario": "online_serving",
            "model": "mock",
            **changes,
        }
    )["tasks"][0]["cases"][0]["parameters"]


def test_lengths_fixed_uniform_normal():
    fixed = tokenization.LengthSampler.from_parameters(parameters(), "input")
    assert {fixed.sample(i) for i in range(100)} == {128}
    uniform = tokenization.LengthSampler.from_parameters(
        parameters(
            input_tokens_min=8,
            input_tokens_max=12,
        ),
        "input",
    )
    assert {uniform.sample(i) for i in range(200)} == set(range(8, 13))
    normal = tokenization.LengthSampler.from_parameters(
        parameters(
            input_tokens=100,
            input_tokens_std=2,
        ),
        "input",
    )
    assert (normal.lower, normal.upper) == (90, 110)
    values = [normal.sample(i) for i in range(500)]
    assert min(values) >= 90 and max(values) <= 110
    assert 99 < sum(values) / len(values) < 101
    assert values == [normal.sample(i) for i in range(500)]


@pytest.mark.parametrize(
    "changes",
    [
        {"input_tokens_min": 20, "input_tokens_max": 10},
        {"output_tokens_std": -1},
        {"input_tokens_std": 2, "input_tokens_max": 2},
        {"input_tokens_min": 0},
        {"input_tokens": 100, "output_tokens": 100, "max_model_len": 199},
        {"input_tokens": 100, "input_tokens_std": 10, "max_model_len": 270},
    ],
)
def test_invalid_lengths_rejected_before_execution(changes):
    with pytest.raises(ValueError):
        parameters(**changes)


def test_sample_identity_independent_of_scheduling_and_warmup(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    p = parameters(
        input_tokens=32,
        input_tokens_std=3,
        output_tokens_min=2,
        output_tokens_max=8,
        num_requests=5,
    )
    first = tokenization.TextGenerator(tokenizer, p)
    records = [first.sample(i, "formal") for i in range(5)]
    other = tokenization.TextGenerator(
        tokenizer,
        {
            **p,
            "max_concurrency": 16,
            "request_rate": 123,
            "arrival_process": "poisson",
            "num_warmups": 20,
        },
    )
    for i in reversed(range(5)):
        assert other.sample(i, "formal") == records[i]
        assert other.sample(i, "warmup")["prompt"] != records[i]["prompt"]
    for row in records:
        assert (
            len(tokenizer.encode(row["prompt"], add_special_tokens=False))
            == row["target_input_tokens"]
        )
    assert len({r["prompt"] for r in records}) == 5
    new_seed = tokenization.TextGenerator(tokenizer, {**p, "data_seed": 7})
    assert new_seed.sample(0, "formal") != records[0]
    changed_output = tokenization.TextGenerator(
        tokenizer, {**p, "output_tokens": 5, "output_tokens_std": 2}
    )
    assert changed_output.sample(0, "formal")["prompt"] == records[0]["prompt"]


def test_prepare_keeps_formal_data_when_warmup_count_changes(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    p = parameters(input_tokens=16, num_requests=3, num_warmups=1)
    context = SimpleNamespace(stopped=lambda: False)
    before = tokenization.TextGenerator(tokenizer, p).prepare(context)
    after = tokenization.TextGenerator(
        tokenizer, {**p, "num_warmups": 4}
    ).prepare(context)
    assert before["formal"] == after["formal"]
    assert len(after["warmup"]) == 4


def test_unstable_tokenizer_fails_bounded_generation():
    tokenizer = SimpleNamespace(
        encode=lambda *a, **kw: [1] * 10,
        decode=lambda *a, **kw: "text",
    )
    generator = tokenization.TextGenerator(
        tokenizer, parameters(input_tokens=5)
    )
    with pytest.raises(ValueError, match="8 attempts"):
        generator.sample(0, "formal")


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("observe", "conforming"),
        ("strict", "nonconforming"),
    ],
)
def test_input_deviation_is_not_request_failure(
    service, tmp_path, policy, expected
):
    Handler.input_delta = 2
    p = {**serving(service), "length_policy": policy, "num_warmups": 1}
    result, directory = RunCoordinator().run(build_plan(p), tmp_path / policy)
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    facts = json.loads(Path(case["raw"]).read_text())
    for sample in facts["samples"]:
        row = sample["facts"]
        assert row["status"] == "success"
        assert row["lengths"]["condition"] == expected
        assert row["lengths"]["input_delta"] == 2
    metric = case["metrics"]["streaming_performance"]
    assert metric["values"]["request_view"]["input_tokens"]["mean"] == 10
    assert metric["lengths"]["counts"][expected] == 4
    from luban_meter.reporting.html import write_html

    html = write_html(result, directory / "report.html")
    assert "conformance_rate" in html.read_text()


def test_missing_usage_does_not_invent_token_counts(service, tmp_path):
    Handler.omit_usage = True
    result, _ = RunCoordinator().run(build_plan(serving(service)), tmp_path)
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    metric = case["metrics"]["streaming_performance"]
    assert metric["lengths"]["counts"]["unknown"] == 4
    service_metrics = metric["values"]["service_view"]
    assert service_metrics["output_token_throughput"]["value"] is None
    assert service_metrics["successful_requests"]["value"] == 4


@pytest.mark.parametrize(
    "ignore_eos,condition",
    [
        (True, "nonconforming"),
        (False, "conforming"),
    ],
)
def test_output_budget_and_early_eos(service, tmp_path, ignore_eos, condition):
    Handler.output_shortfall = 1
    p = {**serving(service), "ignore_eos": ignore_eos}
    result, _ = RunCoordinator().run(build_plan(p), tmp_path)
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    posts = [body for _, body in Handler.posts if body.get("stream")]
    assert all((body.get("min_tokens") == 3) == ignore_eos for body in posts)
    metric = case["metrics"]["streaming_performance"]
    assert metric["lengths"]["counts"][condition] == 4
    assert metric["values"]["request_view"]["output_tokens"]["mean"] == 2


def test_chat_accepts_synthetic_text(service, tmp_path):
    p = {**serving(service), "endpoint": "/v1/chat/completions"}
    result, _ = RunCoordinator().run(build_plan(p), tmp_path)
    assert result["status"] == "success", result
    assert all("messages" in body for _, body in Handler.posts)


def test_sharegpt_preserved_and_range_checked(service, tmp_path):
    text = "  Real input stays unchanged!  "
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "conversations": [{"from": "human", "value": text}],
                }
            ]
        )
    )
    p = {
        **serving(service),
        "data": "sharegpt",
        "dataset_path": str(dataset),
        "input_tokens_max": 10,
    }
    result, _ = RunCoordinator().run(build_plan(p), tmp_path / "run")
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    assert Handler.posts[0][1]["prompt"] == text
    metric = case["metrics"]["streaming_performance"]
    assert metric["lengths"]["counts"]["nonconforming"] == 1
    assert metric["lengths"]["target_input_tokens"]["count"] == 0


def test_conformance_unknown_without_actual_and_boundaries():
    p = parameters(length_policy="strict")
    sample = {"target_input_tokens": 128, "target_output_tokens": 128}
    assert assess_lengths(sample, None, None, p)["condition"] == "unknown"
    assert assess_lengths(sample, 128, 128, p)["condition"] == "conforming"


@pytest.mark.parametrize("source", ["random_token", "burstgpt"])
def test_offline_mixed_lengths_per_request_budgets(
    monkeypatch, tmp_path, source
):
    import sys

    from luban_meter.benchmarking.generation_performance import (
        generation_benchmarks,
    )
    from luban_meter.benchmarking.generation_performance import (
        vllm_engine as engine,
    )
    from luban_meter.runtime.evaluation_session import run_case

    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    batches = []

    def generate(prompts, settings, use_tqdm):
        batches.append((prompts, settings))
        result = []
        for prompt, setting in zip(prompts, settings, strict=True):
            # A valid Engine response shorter than its strict output budget
            # remains successful; its workload condition is nonconforming.
            count = setting["max_tokens"] - 1
            result.append(
                SimpleNamespace(
                    prompt_token_ids=prompt["prompt_token_ids"],
                    outputs=[SimpleNamespace(token_ids=[1] * count)],
                    metrics=SimpleNamespace(
                        first_token_latency=0.1,
                        scheduled_ts=10.0,
                        first_token_ts=10.1,
                        last_token_ts=10.1 if count == 1 else 10.3,
                    ),
                )
            )
        return result

    implementation = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        generate=generate,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            SamplingParams=lambda **kwargs: kwargs,
        ),
    )
    monkeypatch.setattr(engine, "build_engine", lambda *args: implementation)
    monkeypatch.setattr(engine, "engine_max_model_len", lambda _: 256)
    monkeypatch.setattr(
        engine,
        "kv_cache_environment",
        lambda _: {
            "num_gpu_blocks": 8,
            "block_size": 16,
            "kv_cache_size_tokens": 128,
            "kv_cache_max_concurrency": 4,
        },
    )
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "timestamp,input_token,output_token\n"
        + "".join(f"{i},{i + 8},{i + 2}\n" for i in range(7))
    )
    plan = build_plan(
        {
            "scenario": "offline_vllm_engine",
            "data": source,
            "dataset_path": str(trace) if source == "burstgpt" else "",
            "model": "mock",
            "input_tokens_min": 8,
            "input_tokens_max": 20,
            "output_tokens_min": 2,
            "output_tokens_max": 8,
            "num_requests": 7,
            "num_warmups": 1,
            "batch_size": 3,
            "length_policy": "strict",
        }
    )
    case_plan = plan["tasks"][0]["cases"][0]
    (tmp_path / "raw").mkdir()
    result = run_case(
        generation_benchmarks.OfflineEngineBenchmark,
        case_plan,
        tmp_path,
        tmp_path / "stop",
    )
    assert result["status"] == "success", result
    assert [len(prompts) for prompts, _ in batches] == [3, 3, 3, 1]
    assert result["window"]["admitted_requests"] == 7
    facts = json.loads(Path(result["raw"]).read_text())
    requests = [r for s in facts["samples"] for r in s["facts"]["requests"]]
    assert len({r["actual_prompt_tokens"] for r in requests}) > 1
    assert len({r["actual_output_tokens"] for r in requests}) > 1
    assert all(r["lengths"]["condition"] == "nonconforming" for r in requests)
    summary = result["metrics"]["engine_performance"]["lengths"]
    assert summary["output_delta"]["mean"] == -1
    assert summary["actual_output_tokens"]["count"] == 7
    generator = tokenization.TextGenerator(tokenizer, case_plan["parameters"])
    inputs = [r for s in facts["samples"] for r in s["source"]["inputs"]]
    if source == "random_token":
        assert inputs == [generator.sample(i, "formal") for i in range(7)]
    else:
        assert [
            (r["target_input_tokens"], r["target_output_tokens"])
            for r in inputs
        ] == [(i + 8, i + 2) for i in range(7)]


def test_chat_context_overflow_detected_before_any_request(
    service,
    tmp_path,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Handler.tokenizer_path)
    tokenizer.chat_template = (
        "{{ 'a very long chat template prefix' }}"
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    tokenizer.save_pretrained(Handler.tokenizer_path)
    p = {
        **serving(service),
        "endpoint": "/v1/chat/completions",
        "max_model_len": 12,
    }
    result, _ = RunCoordinator().run(build_plan(p), tmp_path / "run")
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "discarded"
    assert "exceeds model limit" in case["reason"]["message"]
    assert not Handler.posts


def test_online_formal_inputs_survive_profile_and_warmup_changes(
    service,
    tmp_path,
):
    base = {
        **serving(service),
        "input_tokens_min": 8,
        "input_tokens_max": 20,
        "output_tokens_min": 2,
        "output_tokens_max": 6,
        "data_seed": 123,
    }
    formal = []
    for index, changes in enumerate(
        (
            {"profile": "concurrent", "num_warmups": 0},
            {
                "profile": "rate",
                "request_rate": 100,
                "arrival_process": "poisson",
                "num_warmups": 2,
                "max_concurrency": 1,
                "seed": 999,
            },
        )
    ):
        result, _ = RunCoordinator().run(
            build_plan({**base, **changes}),
            tmp_path / str(index),
        )
        case = result["tasks"][0]["cases"][0]
        assert case["status"] == "success", case
        facts = json.loads(Path(case["raw"]).read_text())
        formal.append([row["source"] for row in facts["samples"]])
        for row in facts["samples"]:
            assert (
                row["facts"]["output_tokens"]
                == (row["source"]["target_output_tokens"])
            )
        assert case["window"]["admitted_requests"] == 4
    assert formal[0] == formal[1]
