"""Prompt templates for inference benchmarks.

Templates belong to the task content layer only. Model chat-template tokens are
applied by the serving side, never rendered here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CHOICE_LETTERS = ("A", "B", "C", "D")

SUPPORTED_PROMPT_VERSIONS = {
    "ceval": ("ceval-v1",),
    "cmmlu": ("cmmlu-v1",),
    "gsm8k": ("gsm8k-v1",),
}


def validate_prompt_version(benchmark: str, prompt_version: str) -> None:
    allowed = SUPPORTED_PROMPT_VERSIONS.get(benchmark, ())
    if prompt_version not in allowed:
        raise ValueError(
            f"prompt_version must be one of {list(allowed)} for {benchmark}"
        )


def render_choice_question(sample: Mapping[str, Any]) -> str:
    """Render question and choices lines for a four-choice sample."""
    question = str(sample["question"]).strip()
    choices = sample["choices"]
    if not isinstance(choices, Sequence) or len(choices) != 4:
        raise ValueError("choices must be a list of four entries")
    lines = [f"问题：{question}"]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def render_choice_prompt(
    sample: Mapping[str, Any],
    *,
    few_shot_samples: Sequence[Mapping[str, Any]] = (),
    instruction_prefix: str = "以下是中国关于",
    instruction_suffix: str = "的单项选择题，请选出其中的正确答案。",
) -> str:
    """Render instruction + few-shot examples + question, ending with 答案：."""
    subject = str(sample.get("subject") or "综合知识")
    header = f"{instruction_prefix}{subject}{instruction_suffix}\n"
    blocks: list[str] = []
    for example in few_shot_samples:
        blocks.append(
            f"{render_choice_question(example)}\n答案：{example['answer']}"
        )
    blocks.append(render_choice_question(sample))
    return f"{header}" + "\n\n".join(blocks) + "\n答案："


def render_math_prompt(
    sample: Mapping[str, Any],
    *,
    few_shot_samples: Sequence[Mapping[str, Any]] = (),
    instruction: str = (
        "以下是数学应用题，请逐步推理，最后用 \"#### 数字\" 给出最终答案。"
    ),
) -> str:
    """Render GSM8K-style few-shot chain-of-thought prompt, ending with 答案：."""
    blocks: list[str] = []
    for example in few_shot_samples:
        question = str(example["question"]).strip()
        answer = str(example["answer"]).strip()
        blocks.append(f"问题：{question}\n答案：{answer}")
    blocks.append(f"问题：{str(sample['question']).strip()}\n答案：")
    return f"{instruction}\n\n" + "\n\n".join(blocks)
