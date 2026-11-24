"""Prompt templates for model service quality benchmarking.

Templates belong to the task content layer only. Model chat-template
tokens are applied by the serving side, never rendered here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CHOICE_LETTERS = ("A", "B", "C", "D")


def render_choice_generation(sample, examples=()):
    subject = sample.get("subject") or "综合知识"
    instruction = (
        f"以下是关于{subject}的单项选择题。每题只能选择一个正确答案，"
        "合法选项为A、B、C、D。\n"
        "只回答待答题。示例仅展示格式，不要复述示例、题目或选项。\n"
        "只输出一行，格式为“答案：X”，X为一个大写选项字母。"
        "不要输出分析、解释或其他文字。"
    )
    ending = "请输出待答题的答案，格式为“答案：X”。"
    return _render_bounded(sample, examples, instruction, ending, choice=True)


def _render_summary(sample, examples):
    instruction = (
        "阅读正文并生成简短摘要，概括其中的主要信息。\n"
        "摘要只能依据给出的正文，不补充正文之外的信息。\n"
        "只返回一行摘要正文，不添加标题、“摘要：”前缀、解释、"
        "列表或其他内容。\n"
        "示例仅展示任务形式；只为待摘要文本生成摘要。"
    )
    ending = "请输出待摘要文本的一行摘要正文。"
    return _render_bounded(sample, examples, instruction, ending)


def _render_bounded(sample, examples, instruction, ending, *, choice=False):
    """Keep dataset text and selected examples outside template boundaries."""
    input_fields = ("question", "choices", "subject", "content", "title",
                    "context")
    sources = [{key: sample[key] for key in input_fields if key in sample},
               *examples]
    source_text = "\n".join(
        str(value) for row in sources for value in row.values()
    )
    suffix = ""
    counter = 0
    while f"[LBM{suffix}:" in source_text:
        counter += 1
        suffix = f"_{counter}"
    markers = []

    def mark(label):
        value = f"[LBM{suffix}:{label}]"
        markers.append(value)
        return value

    def field(label, value):
        return f"{mark(label + '_BEGIN')}\n{value}\n{mark(label + '_END')}"

    def body(row):
        if not choice:
            return field("CONTENT", row["content"])
        # Reuse validation but keep the question itself unchanged.
        render_choice_question(row)
        options = "\n".join(
            f"{letter}. {value}"
            for letter, value in zip(CHOICE_LETTERS, row["choices"])
        )
        return field("QUESTION", row["question"]) + "\n" + field(
            "OPTIONS", options
        )

    blocks = [instruction]
    for index, example in enumerate(examples, 1):
        answer = (
            f"答案：{example['answer']}" if choice
            else field("SUMMARY", example["abst"])
        )
        blocks.append(
            f"{mark(f'EXAMPLE_{index}_BEGIN')}\n{body(example)}\n"
            f"{answer}\n{mark(f'EXAMPLE_{index}_END')}"
        )
    blocks.append(
        f"{mark('TARGET_BEGIN')}\n{body(sample)}\n{mark('TARGET_END')}"
    )
    return {
        "prompt": "\n\n".join([*blocks, ending]),
        "boundary_markers": list(dict.fromkeys(markers)),
    }


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


def render_choice_ppl(
    sample: Mapping[str, Any],
    *,
    few_shot_samples: Sequence[Mapping[str, Any]] = (),
    instruction_prefix: str = "以下是中国关于",
    instruction_suffix: str = "的单项选择题，请选出其中的正确答案。",
) -> str:
    """Render the choice continuation for logprob scoring."""
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
        '以下是数学应用题，请逐步推理，最后用 "#### 数字" 给出最终答案。'
    ),
) -> str:
    """Render GSM8K-style few-shot chain-of-thought prompt, ending with
    答案：.
    """
    blocks: list[str] = []
    for example in few_shot_samples:
        question = str(example["question"]).strip()
        answer = str(example["answer"]).strip()
        blocks.append(f"问题：{question}\n答案：{answer}")
    blocks.append(f"问题：{str(sample['question']).strip()}\n答案：")
    return f"{instruction}\n\n" + "\n\n".join(blocks)


def render_wikitext_prompt(sample: Mapping[str, Any]) -> str:
    """Return the detokenized text for loss scoring (no template)."""
    text = sample.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("sample text must be a non-empty string")
    return text


def render_lcsts_prompt(sample, examples=()):
    content = sample["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("sample content must be a non-empty string")
    return _render_summary(sample, examples)


def render_lcsts_override(
    sample: Mapping[str, Any],
    *,
    few_shot_samples: Sequence[Mapping[str, Any]] = (),
    template: str,
) -> str:
    """Render an explicitly supplied LCSTS format string."""
    content = sample.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("sample content must be a non-empty string")
    blocks: list[str] = []
    for example in few_shot_samples:
        ex_content = str(example["content"]).strip()
        ex_abst = str(example["abst"]).strip()
        blocks.append(f"{template.format(content=ex_content)}{ex_abst}")
    blocks.append(template.format(content=content.strip()))
    return "\n\n".join(blocks)


def render_squad_prompt(sample: Mapping[str, Any]) -> str:
    """Render zero-shot SQuAD 2.0 without truncating the source
    context.
    """
    return (
        "Answer the question using only the passage below. "
        "Return only the shortest answer span copied from the passage. "
        "If the passage does not support an answer, return exactly "
        "unanswerable. Do not explain your answer.\n\n"
        f"Title: {sample.get('title', '')}\n\n"
        f"Passage: {sample['context']}\n\n"
        f"Question: {sample['question']}\nAnswer:"
    )
