"""SQuAD 2.0 text scoring, without probability threshold optimization.

Normalization follows the official evaluate-v2.0.py algorithm. Keep this
separate from the broader multilingual common.normalize_answer helper.
"""

import re
import string

from luban_meter.benchmarking.model_service_quality.common.metrics import (
    token_f1,
)

NO_ANSWER = "unanswerable"
SCORER_VERSION = "squad2-official-text-v1"


def normalize_answer(text: str) -> str:
    """Lowercase, remove ASCII punctuation, articles, then extra
    spaces.
    """
    text = "".join(
        character
        for character in text.lower()
        if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_answer(text: str) -> str:
    """Accept an exact refusal marker; otherwise preserve answer
    content.

    Empty output is a parse failure, not a deliberate abstention. No
    substring matching, explanation extraction, punctuation
    truncation or case folding is applied before recognizing the
    marker.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty or non-string model output")
    answer = text.strip()
    return "" if answer == NO_ANSWER else answer


def score_answer(prediction: str, answers: list[str]) -> tuple[float, float]:
    """Return EM/F1 in [0, 1], maximizing independently over
    references.
    """
    golds = [
        normalize_answer(answer)
        for answer in answers
        if normalize_answer(answer)
    ]
    golds = golds or [""]
    predicted = normalize_answer(prediction)
    return (
        float(any(predicted == gold for gold in golds)),
        max(token_f1(predicted.split(), gold.split()) for gold in golds),
    )
