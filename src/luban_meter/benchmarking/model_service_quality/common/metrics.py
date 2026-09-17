"""Metric math for model service quality benchmarks (standard library only)."""

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


def token_f1(
    prediction_tokens: Sequence[str], reference_tokens: Sequence[str]
) -> float:
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


def ngrams(tokens: Sequence[str], ngram_size: int) -> list[tuple[str, ...]]:
    if ngram_size <= 0:
        raise ValueError("n must be positive")
    return [
        tuple(tokens[token_index : token_index + ngram_size])
        for token_index in range(len(tokens) - ngram_size + 1)
    ]


def rouge_n_f1(
    prediction_tokens: Sequence[str],
    reference_tokens: Sequence[str],
    ngram_size: int,
) -> float:
    """F-measure of n-gram overlap between prediction and reference."""
    pred_ngrams = ngrams(prediction_tokens, ngram_size)
    ref_ngrams = ngrams(reference_tokens, ngram_size)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_ngrams)
    recall = overlap / len(ref_ngrams)
    return 2 * precision * recall / (precision + recall)


def longest_common_subsequence_length(
    prediction_tokens: Sequence[str], reference_tokens: Sequence[str]
) -> int:
    if not prediction_tokens or not reference_tokens:
        return 0
    previous = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        current = [0] * (len(reference_tokens) + 1)
        for reference_index, reference_token in enumerate(
            reference_tokens, start=1
        ):
            if prediction_token == reference_token:
                current[reference_index] = previous[reference_index - 1] + 1
            else:
                current[reference_index] = max(
                    previous[reference_index], current[reference_index - 1]
                )
        previous = current
    return previous[-1]


def rouge_l_f1(
    prediction_tokens: Sequence[str], reference_tokens: Sequence[str]
) -> float:
    """F-measure based on the longest common subsequence."""
    if not prediction_tokens or not reference_tokens:
        return 0.0
    lcs = longest_common_subsequence_length(
        prediction_tokens, reference_tokens
    )
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def pass_at_k(
    sample_count: int, success_count: int, selection_count: int
) -> float:
    """Unbiased pass@k estimator for n samples with c successes."""
    if (
        sample_count <= 0
        or not 0 <= success_count <= sample_count
        or not 1 <= selection_count <= sample_count
    ):
        raise ValueError(
            f"invalid pass_at_k parameters: n={sample_count}, "
            f"c={success_count}, k={selection_count}"
        )
    if sample_count - success_count < selection_count:
        return 1.0
    return 1.0 - math.comb(
        sample_count - success_count, selection_count
    ) / math.comb(sample_count, selection_count)


def mean_loss(total_logprob: float, token_count: int) -> float:
    """Average negative log likelihood in nats per token."""
    if token_count <= 0:
        raise ValueError("token count must be positive")
    return -total_logprob / token_count


def perplexity(total_logprob: float, token_count: int) -> float:
    return math.exp(mean_loss(total_logprob, token_count))


def bits_per_byte(total_logprob: float, byte_count: int) -> float:
    """Bits per byte (tokenizer-independent, cross-model comparable).

    bpb = -sum_logprob / (byte_count * ln(2))
    """
    if byte_count <= 0:
        raise ValueError("byte count must be positive")
    return -total_logprob / byte_count / math.log(2)
