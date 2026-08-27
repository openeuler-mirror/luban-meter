"""Answer extraction and normalization for inference benchmarks."""

from __future__ import annotations

import re
import string

_CHOICE_PATTERNS = (
    re.compile(r"答案是\s*[:：]?\s*([A-D])"),
    re.compile(r"选\s*[:：]?\s*([A-D])"),
    re.compile(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*([A-D])", re.IGNORECASE),
    re.compile(r"^([A-D])(?![a-zA-Z])"),
    re.compile(r"(?<![A-Za-z])([A-D])(?![a-z])"),
)

_GSM8K_DELIMITED = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")
_NUMBER = re.compile(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation + "。，、；：？！“”‘’（）《》【】…—·")
_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_choice(text: str) -> str | None:
    """Extract an A-D letter from generated text; None when unparseable."""
    if not text:
        return None
    stripped = text.strip()
    for pattern in _CHOICE_PATTERNS:
        match = pattern.search(stripped)
        if match:
            return match.group(1).upper()
    return None


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
        return None


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/space."""
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
