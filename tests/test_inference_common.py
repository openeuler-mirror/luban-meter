"""Unit tests for inference/common pure functions."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from luban_meter.benchmark.inference.common import metrics, parsers, prompts
from luban_meter.benchmark.inference.common.dataset import (
    load_records,
    select_few_shot,
    select_records,
)
from luban_meter.benchmark.inference.common.parameters import (
    enum_value,
    fixed_value,
    positive_integer,
)


def test_accuracy() -> None:
    assert metrics.accuracy(7, 10) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        metrics.accuracy(1, 0)
    with pytest.raises(ValueError):
        metrics.accuracy(2, 1)


def test_token_f1() -> None:
    assert metrics.token_f1([], []) == 1.0
    assert metrics.token_f1(["a"], []) == 0.0
    f1 = metrics.token_f1(["the", "cat"], ["the", "cat", "sat"])
    assert f1 == pytest.approx(0.8)


def test_rouge_n_f1() -> None:
    score = metrics.rouge_n_f1(list("abc"), list("abd"), 1)
    assert score == pytest.approx(2 / 3)
    assert metrics.rouge_n_f1([], list("a"), 1) == 0.0


def test_rouge_l_f1() -> None:
    score = metrics.rouge_l_f1(list("abcd"), list("acd"))
    assert score == pytest.approx(6 / 7)


def test_pass_at_k() -> None:
    assert metrics.pass_at_k(1, 1, 1) == 1.0
    assert metrics.pass_at_k(1, 0, 1) == 0.0
    assert metrics.pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)
    with pytest.raises(ValueError):
        metrics.pass_at_k(4, 5, 2)


def test_mean_loss_and_perplexity() -> None:
    total_logprob = -math.log(2.0) * 10
    assert metrics.mean_loss(total_logprob, 10) == pytest.approx(math.log(2.0))
    assert metrics.perplexity(total_logprob, 10) == pytest.approx(2.0)


def test_extract_choice() -> None:
    assert parsers.extract_choice("答案是 B。") == "B"
    assert parsers.extract_choice("B") == "B"
    assert parsers.extract_choice("选 C") == "C"
    assert parsers.extract_choice("Answer: D") == "D"
    assert parsers.extract_choice("无法确定") is None
    assert parsers.extract_choice("") is None


def test_extract_number() -> None:
    assert parsers.extract_number("推理过程……\n#### 42") == 42.0
    assert parsers.extract_number("The answer is 1,234") == 1234.0
    assert parsers.extract_number("没有数字") is None


def test_normalize_answer() -> None:
    assert parsers.normalize_answer("The  Cat!") == "cat"


def test_extract_code() -> None:
    text = "```python\ndef f():\n    return 1\n```"
    assert parsers.extract_code(text) == "def f():\n    return 1\n"
    assert parsers.extract_code("plain") == "plain"


def test_render_choice_prompt() -> None:
    sample = {
        "id": "1",
        "question": "题目",
        "choices": ["甲", "乙", "丙", "丁"],
        "answer": "B",
        "subject": "计算机网络",
    }
    example = {
        "id": "0",
        "question": "示例题",
        "choices": ["一", "二", "三", "四"],
        "answer": "A",
        "subject": "计算机网络",
    }
    prompt = prompts.render_choice_prompt(sample, few_shot_samples=[example])
    assert prompt.startswith("以下是中国关于计算机网络的单项选择题")
    assert "A. 甲" in prompt
    assert "答案：A" in prompt
    assert prompt.endswith("答案：")


def test_validate_prompt_version() -> None:
    prompts.validate_prompt_version("ceval", "ceval-v1")
    with pytest.raises(ValueError):
        prompts.validate_prompt_version("ceval", "unknown")


def test_dataset_loading_and_sampling(tmp_path: Path) -> None:
    file = tmp_path / "data.jsonl"
    lines = [
        '{"id": "1", "subject": "s1"}',
        '{"id": "2", "subject": "s2"}',
        '{"id": "3", "subject": "s1"}',
    ]
    file.write_text("\n".join(lines), encoding="utf-8")
    records = load_records(file)
    assert len(records) == 3
    assert [record["id"] for record in select_records(records, max_samples=2)] == [
        "1",
        "2",
    ]
    shuffled = select_records(records, shuffle=True, seed=7)
    assert [record["id"] for record in shuffled] != [
        record["id"] for record in records
    ] or len(records) <= 1
    assert select_records(records, shuffle=True, seed=7) == shuffled
    few_shot = select_few_shot(records, "subject", "s1", 5)
    assert [record["id"] for record in few_shot] == ["1", "3"]


def test_resolve_data_path_falls_back_to_bundled_package_data(tmp_path: Path) -> None:
    """Default relative dataset paths resolve to bundled package data."""
    from luban_meter.benchmark.inference.common.dataset import resolve_data_path

    # Absolute paths are returned verbatim.
    absolute = tmp_path / "abs.jsonl"
    absolute.write_text("{}", encoding="utf-8")
    assert resolve_data_path(absolute) == absolute

    # A default YAML relative path resolves to the bundled package data dir.
    bundled = resolve_data_path("data/ceval/val.jsonl")
    assert bundled.is_file()
    assert resolve_data_path(bundled) == bundled

    # A missing relative path keeps its original form so load_records raises.
    missing = resolve_data_path("data/does/not/exist.jsonl")
    assert missing == Path("data/does/not/exist.jsonl")


def test_parameter_validation() -> None:
    assert positive_integer({"a": 3}, "a", 1) == 3
    with pytest.raises(ValueError):
        positive_integer({"a": True}, "a", 1)
    assert enum_value({"mode": "ppl"}, "mode", ("ppl", "gen"), "gen") == "ppl"
    with pytest.raises(ValueError):
        enum_value({"mode": "x"}, "mode", ("ppl", "gen"), "gen")
    with pytest.raises(ValueError):
        fixed_value({"temperature": 1.0}, "temperature", 0.0)
