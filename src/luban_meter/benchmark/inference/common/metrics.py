"""Metric math for inference benchmarks (standard library only)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def accuracy(correct_count: int, total_count: int) -> float:
    if total_count <= 0:
        raise ValueError("total count must be positive")
    if correct_count < 0 or correct_count > total_count:
        raise ValueError("correct count out of range")
    return correct_count / total_count


def token_f1(prediction_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    """Token-level F1 between one prediction and one reference."""
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(
    prediction_tokens: Sequence[str], reference_tokens: Sequence[str], n: int
) -> float:
    """F-measure of n-gram overlap between prediction and reference."""
    pred_ngrams = ngrams(prediction_tokens, n)
    ref_ngrams = ngrams(reference_tokens, n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_ngrams)
    recall = overlap / len(ref_ngrams)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for item_a in a:
        current = [0] * (len(b) + 1)
        for index_b, item_b in enumerate(b, start=1):
            if item_a == item_b:
                current[index_b] = previous[index_b - 1] + 1
            else:
                current[index_b] = max(previous[index_b], current[index_b - 1])
        previous = current
    return previous[-1]


def rouge_l_f1(prediction_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    """F-measure based on the longest common subsequence."""
    if not prediction_tokens or not reference_tokens:
        return 0.0
    lcs = lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator for n samples with c successes."""
    if n <= 0 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"invalid pass_at_k parameters: n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_loss(total_logprob: float, token_count: int) -> float:
    """Average negative log likelihood in nats per token."""
    if token_count <= 0:
        raise ValueError("token count must be positive")
    return -total_logprob / token_count


def perplexity(total_logprob: float, token_count: int) -> float:
    return math.exp(mean_loss(total_logprob, token_count))
