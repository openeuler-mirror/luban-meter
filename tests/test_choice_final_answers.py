"""Final-answer boundaries shared by C-Eval and CMMLU generation."""

from __future__ import annotations

import json

import httpx
import pytest

from luban_meter.benchmarking.model_service_quality.common import (
    choice,
    parsers,
    prompts,
)
from luban_meter.benchmarking.model_service_quality.common import (
    template_adapters,
)
from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
)
from luban_meter.core.planning import build_plan


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


@pytest.mark.parametrize("name", ["ceval", "cmmlu"])
@pytest.mark.parametrize(
    ("text", "expected", "status"),
    [
        ("<think>选项是A到D四个选项", None, "parse_failed"),
        ("<think>答案是A</think>\n答案：D", "D", "success"),
        ("答案：A\n答案：D", None, "parse_failed"),
    ],
)
def test_choice_collection_keeps_raw_output(name, text, expected, status):
    class Client:
        def chat(self, messages, **kwargs):
            assert "只输出一行答案" in messages[0]["content"]
            assert kwargs["stop"] == []
            return {"text": text, "input_tokens": 100, "output_tokens": 50}

    definition = choice.ChoiceBenchmarkDefinition(name, name, name)
    config = choice.validate_choice_parameters(
        {
            "dataset_path": "unused.jsonl",
            "few_shot": 0,
            "eval_mode": "gen",
        },
        definition,
    )
    assert config["prompt_version"] == f"{name}-v3"
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


def test_choice_prompt_requires_final_answer_and_preserves_ppl():
    sample = {
        "question": "题目",
        "choices": ["甲", "乙", "丙", "丁"],
        "answer": "B",
    }
    text = prompts.render_choice_prompt(sample, few_shot_samples=[sample])
    assert "只能是A、B、C、D之一" in text
    assert "只输出一行答案" in text
    assert "不要输出思考过程、分析、解释或其他文字" in text
    assert "答案：B\n\n问题：题目" in text
    assert text.endswith("答案：")
    legacy = prompts.render_choice_prompt(sample, strict_final_answer=False)
    assert "最终答案" not in legacy


@pytest.mark.parametrize("name", ["ceval", "cmmlu"])
def test_choice_prompt_v2_remains_reproducible(name):
    config = choice.validate_choice_parameters(
        {
            "dataset_path": "unused.jsonl",
            "few_shot": 0,
            "eval_mode": "gen",
            "prompt_version": f"{name}-v2",
        },
        choice.ChoiceBenchmarkDefinition(name, name, name),
    )
    sample = {"question": "题目", "choices": ["甲", "乙", "丙", "丁"]}
    text = template_adapters.ChoiceTemplate().render(sample, [], config)
    assert "最终答案单独写在最后一行" in text["prompt"]
    assert "只输出一行答案" not in text["prompt"]


@pytest.mark.parametrize("name", ["ceval", "cmmlu"])
def test_choice_collection_forwards_thinking_switch(name):
    class Client:
        def chat(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["max_tokens"] == 8192
            assert "只输出一行答案" in messages[0]["content"]
            return {"text": "答案：D", "output_tokens": 4}

    config = choice.validate_choice_parameters(
        {
            "dataset_path": "unused.jsonl",
            "few_shot": 0,
            "eval_mode": "gen",
            "max_tokens": 8192,
            "enable_thinking": False,
        },
        choice.ChoiceBenchmarkDefinition(name, name, name),
    )
    sample = {
        "id": "q1",
        "question": "题目",
        "choices": ["甲", "乙", "丙", "丁"],
        "answer": "D",
    }
    assert choice.collect_sample(Client(), sample, [], config)["correct"]


@pytest.mark.parametrize("enabled", [None, False, True])
def test_thinking_switch_is_sent_as_chat_template_kwarg(enabled):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "答案：A"}}]}
        )

    client = OpenAIClient("http://mock", "model")
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        result = client.chat(
            [{"role": "user", "content": "题目"}],
            enable_thinking=enabled,
        )
    finally:
        client.close()
    assert result["text"] == "答案：A"
    if enabled is None:
        assert "chat_template_kwargs" not in payloads[0]
    else:
        assert payloads[0]["chat_template_kwargs"] == {
            "enable_thinking": enabled
        }


@pytest.mark.parametrize("name", ["ceval", "cmmlu"])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_public_thinking_switch(name, enabled, tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {"question": "题目", "choices": ["甲", "乙", "丙", "丁"],
             "answer": "A"}
        ) + "\n"
    )
    plan = build_plan(
        {
            "scenario": name,
            "model": "mock",
            "dataset_path": str(dataset),
            "endpoint": "/v1/chat/completions",
            "few_shot": 0,
            "enable_thinking": enabled,
        }
    )
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    assert parameters["enable_thinking"] is enabled


@pytest.mark.parametrize("enabled", ["false", 0, 1, []])
def test_thinking_switch_rejects_non_booleans(enabled):
    with pytest.raises(ValueError, match="enable_thinking"):
        choice.validate_choice_parameters(
            {
                "dataset_path": "unused.jsonl",
                "few_shot": 0,
                "eval_mode": "gen",
                "enable_thinking": enabled,
            },
            choice.ChoiceBenchmarkDefinition("ceval", "ceval", "ceval"),
        )


@pytest.mark.parametrize(
    ("mode", "prompt_format"), [("ppl", "base"), ("gen", "base")]
)
def test_thinking_switch_requires_chat_generation(mode, prompt_format):
    with pytest.raises(ValueError, match="enable_thinking"):
        choice.validate_choice_parameters(
            {
                "dataset_path": "unused.jsonl",
                "few_shot": 0,
                "eval_mode": mode,
                "prompt_format": prompt_format,
                "enable_thinking": False,
            },
            choice.ChoiceBenchmarkDefinition("ceval", "ceval", "ceval"),
        )
