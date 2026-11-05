"""Extract and normalize model service quality answers."""

from __future__ import annotations

import re
import string

from luban_meter.log import get_logger

logger = get_logger(__name__)


_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
_CHOICE_PREFIX = (
    r"(?:(?:最终答案|正确答案|答案)(?:是)?\s*[:：]?\s*"
    r"|Answer\s*(?:is\s*)?[:：]?\s*)"
)
_CHOICE_LINE = re.compile(
    _CHOICE_PREFIX + r"([A-D])\s*[。.!！]?",
    re.IGNORECASE,
)
_BOXED_CHOICE = re.compile(r"\\boxed\{\s*([A-D])\s*\}", re.IGNORECASE)
_BOXED_TOKEN = r"\\boxed\{\s*[A-D]\s*\}"
_BOXED_DISPLAY = re.compile(
    r"^[ \t]*(?:\$\$\s*(" + _BOXED_TOKEN + r")\s*\$\$"
    r"|\\\[\s*(" + _BOXED_TOKEN + r")\s*\\\])[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_GSM8K_DELIMITED = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")
_NUMBER = re.compile(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation + "。，、；：？！“”‘’（）《》【】…—·")
_CODE_FENCE = re.compile(
    r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def _extract_boxed_choice(line: str) -> str | None:
    prefix = re.match(_CHOICE_PREFIX, line, re.IGNORECASE)
    if prefix:
        line = line[prefix.end():].strip()
        if line.endswith(("。", ".", "!", "！")):
            line = line[:-1].rstrip()
    for opening, closing in (
        ("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"),
    ):
        if line.startswith(opening) and line.endswith(closing):
            line = line[len(opening):-len(closing)].strip()
            break
    match = _BOXED_CHOICE.fullmatch(line)
    return match.group(1).upper() if match else None


def extract_final_output(text: str, *, code: bool = False) -> str:
    """Remove thinking before grading while retaining answer whitespace.

    Code responses only lose complete leading thinking blocks, so tags in
    Python strings and comments are preserved. Empty or malformed answers
    fail closed rather than allowing reasoning to become a prediction.
    """
    if not isinstance(text, str):
        raise ValueError("model output must be text")
    if code:
        answer = _code_after_thinking(text)
    else:
        answer = _text_outside_thinking(text)
    if not answer.strip():
        raise ValueError("no final answer outside thinking blocks")
    return answer


def _code_after_thinking(text: str) -> str:
    start = 0
    while True:
        position = start + len(text[start:]) - len(text[start:].lstrip())
        opening = _THINK_TAG.match(text, position)
        if opening is None:
            return text[start:]
        closing = _THINK_TAG.search(text, opening.end())
        if (
            opening.group().startswith("</")
            or closing is None
            or not closing.group().startswith("</")
        ):
            raise ValueError("unclosed or malformed <think> tags")
        start = closing.end()


def _text_outside_thinking(text: str) -> str:
    parts = []
    start = 0
    thinking = False
    for tag in _THINK_TAG.finditer(text):
        closing = tag.group().startswith("</")
        if closing != thinking:
            raise ValueError("malformed <think> tags")
        if not thinking:
            parts.append(text[start:tag.start()])
        thinking = not closing
        start = tag.end()
    if thinking:
        raise ValueError("unclosed <think> block")
    parts.append(text[start:])
    return "\n".join(parts)


def extract_choice(text: str) -> str | None:
    """Parse an unambiguous final answer outside complete think blocks.

    Unclosed/malformed thinking and conflicting answer lines fail closed.
    Option letters in explanations are never used as a fallback.
    """
    try:
        answer = extract_final_output(text).strip()
    except ValueError:
        logger.warning("Choice answer parsing failed", exc_info=True)
        return None
    answer = _BOXED_DISPLAY.sub(
        lambda match: match.group(1) or match.group(2), answer
    )
    if re.fullmatch(r"[A-D]", answer, re.IGNORECASE):
        return answer.upper()
    candidates = set()
    for line in answer.splitlines():
        line = line.strip()
        if len(line) >= 4 and line.startswith("**") and line.endswith("**"):
            line = line[2:-2].strip()
        match = _CHOICE_LINE.fullmatch(line) or re.fullmatch(
            r"([A-D])", line, re.IGNORECASE
        )
        if match:
            candidates.add(match.group(1).upper())
        else:
            boxed = _extract_boxed_choice(line)
            if boxed:
                candidates.add(boxed)
    return candidates.pop() if len(candidates) == 1 else None


def extract_number(text: str) -> float | None:
    """Extract the final numeric answer, preferring the #### delimiter."""
    if not text:
        return None
    match = _GSM8K_DELIMITED.search(text)
    if match:
        candidate = match.group(1)
    else:
        matches = _NUMBER.findall(text)
        if not matches:
            return None
        candidate = matches[-1]
    try:
        return float(candidate.replace(",", ""))
    except ValueError:
        logger.warning("Numeric answer parsing failed", exc_info=True)
        return None


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip
    punctuation/articles/space.
    """
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = "".join(char for char in text if char not in _PUNCTUATION)
    return " ".join(text.split())


def split_tokens(text: str) -> list[str]:
    """Whitespace tokens of the normalized answer."""
    return normalize_answer(text).split()


def extract_code(text: str) -> str:
    """Strip markdown code fences when present."""
    match = _CODE_FENCE.search(text or "")
    if match:
        return match.group(1)
    return text or ""


def extract_humaneval_completion(text: str, prompt: str) -> str:
    """Extract a completion while preserving function-body indentation.

    HumanEval's canonical contract is completion-only: the returned
    text is appended directly to the prompt. An exact repeated
    prompt is stripped, but Markdown fences and other generated text
    are deliberately preserved so the sandbox judges the model's
    actual base completion.
    """
    candidate = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    candidate = candidate.removeprefix(prompt)
    while candidate.startswith("\n"):
        candidate = candidate[1:]
    if not candidate.strip():
        return ""
    return candidate.rstrip() + "\n"


def lcsts_postprocess(text: str) -> str:
    """First-layer post-processing for LCSTS predictions.

    Matches OpenCompass ``lcsts_postprocess`` in
    ``opencompass/datasets/lcsts.py``: take the first line, strip
    numbered / bulleted prefixes, strip surrounding Chinese
    punctuation.  Applied only to the prediction (not the reference).
    """
    text = text.strip().split("\n")[0].strip()
    if text.startswith("1. "):
        text = text[3:]
    if text.startswith("- "):
        text = text[2:]
    text = text.strip("\u201c\u201d\uff0c\u3002\uff01\u201d")
    return text


def general_postprocess(text: str) -> str:
    """Second-layer post-processing before ROUGE scoring.

    Matches OpenCompass ``general_postprocess`` in
    ``opencompass/utils/text_postprocessors.py``: truncate at the
    first newline / period / comma, remove punctuation, remove
    English articles, collapse whitespace.  Applied to both
    prediction and reference.

    Note: for Chinese text the truncation only fires on ASCII
    ``.``, ``,``, or ``\\n``; Chinese punctuation is removed by the
    ``[^\\w\\s]`` substitution (``\\w`` matches CJK characters).
    """
    truncated = re.split(r"[\n.,]", text, 1)[0]
    no_punct = re.sub(r"[^\w\s]", "", truncated)
    no_articles = re.sub(r"\b(a|an|the)\b", "", no_punct, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", no_articles).strip()
