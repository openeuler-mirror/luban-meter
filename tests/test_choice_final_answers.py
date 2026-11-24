"""Final-answer boundaries shared by C-Eval and CMMLU generation."""

from __future__ import annotations

import pytest

from luban_meter.benchmarking.model_service_quality.common import (
    choice,
    parsers,
    template_adapters,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("D", "D"),
        ("答案：D", "D"),
        ("答案是 B。", "B"),
        ("最终答案：C", "C"),
        ("Answer: D", "D"),
        ("Answer is B.", "B"),
        ("<think>A</think>\n**答案：B**", "B"),
        ("**A. 达尔文**", None),
        ("<think>答案是A</think>\n答案：D", "D"),
        ("<think>选项是A到D四个选项", None),
        ("<think>", None),
        ("<think>答案是A</think>", None),
        ("<think>A</think>\n<think>B</think>\nD", "D"),
        ("<think><think>A</think>\n答案：D", None),
        ("</think>\n答案：D", None),
        ("答案：D\n<think>未结束", None),
        ("选项是A到D四个选项", None),
        ("A. 甲\nB. 乙\nC. 丙\nD. 丁", None),
        ("选 C", None),
        ("答案：A\n最终答案：D", None),
        ("A\n答案：D", None),
        ("A\nD", None),
        ("答案：D\n答案：D", "D"),
        ("答案：A或D", None),
        ("答案：D，因为其他不对", None),
        ("分析A不合适，D更合理。", None),
        ("<think>A</think>\n**A. 达尔文**不符合题意", None),
        ("分析A不合适。\n答案：D", "D"),
        (r"\boxed{A}", "A"),
        (r"$\boxed{b}$", "B"),
        (r"$$\boxed{ C }$$", "C"),
        (r"\(\boxed{D}\)", "D"),
        (r"\[\boxed{A}\]", "A"),
        ("$$\n\\boxed{A}\n$$", "A"),
        ("\\[\n\\boxed{B}\n\\]", "B"),
        (r"答案：\boxed{C}", "C"),
        (r"**最终答案：$\boxed{D}$**", "D"),
        ("答案：A\n\\boxed{A}", "A"),
        ("答案：D\n\\boxed{A}", None),
        ("\\boxed{A}\n\\boxed{D}", None),
        (r"\boxed{AB}", None),
        (r"\boxed{A,B}", None),
        (r"\boxed{a+b}", None),
        (r"推导中的\boxed{A}并非最终答案", None),
        ("<think>\\boxed{A}</think>\n答案：D", "D"),
        ("<think>\\boxed{A}", None),
    ],
)
def test_strict_choice_parser(text, expected):
    assert parsers.extract_choice(text) == expected
    assert template_adapters.ChoiceParser().parse({"raw_output": text}) == {
        "prediction": expected
    }


@pytest.mark.parametrize(
    ("text", "expected", "status"),
    [
        ("<think>选项是A到D四个选项", None, "parse_failed"),
        ("<think>答案是A</think>\n答案：D", "D", "success"),
        ("答案：A\n答案：D", None, "parse_failed"),
    ],
)
def test_choice_collection_keeps_raw_output(text, expected, status):
    class Client:
        def chat(self, messages, **kwargs):
            assert "只输出一行，格式为“答案：X”" in messages[0]["content"]
            assert kwargs["stop"] == []
            return {"text": text, "input_tokens": 100, "output_tokens": 50}

    config = choice.validate_choice_parameters(
        {
            "dataset_path": "unused.jsonl",
            "few_shot": 0,
            "eval_mode": "gen",
        },
    )
    sample = {
        "id": "anatomy-43",
        "question": "全身最复杂的关节是",
        "choices": ["肘关节", "髋关节", "肩关节", "膝关节"],
        "answer": "D",
    }
    result = choice.collect_sample(Client(), sample, [], config)
    assert result["raw_output"] == text
    assert result["prediction"] == expected
    assert result["status"] == status
    assert result["correct"] is (True if expected else None)


def test_choice_ppl_scores_each_continuation():
    sample = {
        "question": "题目", "choices": ["甲", "乙", "丙", "丁"], "answer": "B",
    }
    example = {**sample, "question": "示例"}
    expected = (
        "以下是中国关于综合知识的单项选择题，请选出其中的正确答案。\n"
        "问题：示例\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n答案：B\n\n"
        "问题：题目\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n答案："
    )
    calls = []

    class Client:
        def tokenize(self, text):
            return list(text)

        def completion_logprobs(self, text):
            calls.append(text)
            logprob = -0.1 if text.endswith(" B. 乙") else -2.0
            return {"token_logprobs": [logprob] * len(text)}

    config = choice.validate_choice_parameters(
        {
            "dataset_path": "unused.jsonl", "few_shot": 0,
            "eval_mode": "ppl", "prompt_format": "base",
        },
    )
    result = choice.collect_sample(Client(), sample, [example], config)
    assert result["prompt"] == expected
    assert calls == [expected + f" {letter}. {option}"
                     for letter, option in zip("ABCD", sample["choices"])]
    assert result["prediction"] == "B"
    assert result["correct"] is True


def test_choice_collection_preserves_generation_budget():
    class Client:
        def chat(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 8192
            assert "只输出一行，格式为“答案：X”" in messages[0]["content"]
            return {"text": "答案：D", "output_tokens": 4}

    config = choice.validate_choice_parameters(
        {
            "dataset_path": "unused.jsonl",
            "few_shot": 0,
            "eval_mode": "gen",
            "max_tokens": 8192,
        },
    )
    sample = {
        "id": "q1",
        "question": "题目",
        "choices": ["甲", "乙", "丙", "丁"],
        "answer": "D",
    }
    assert choice.collect_sample(Client(), sample, [], config)["correct"]
