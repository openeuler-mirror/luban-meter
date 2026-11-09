"""Supplement missing reasoning usage from requested Token IDs."""

import json
from dataclasses import dataclass, field
from typing import Any

from luban_meter.log import get_logger

from .tokenization import load_tokenizer

logger = get_logger(__name__)


def _saved_ids(value):
    """Keep malformed IDs readable as ASCII JSON diagnostics."""
    if value is None or (
        isinstance(value, list)
        and all(isinstance(token, int) for token in value)
    ):
        return value
    # Keep escapes so unpaired surrogates cannot reach UTF-8 writers.
    return json.dumps(value, ensure_ascii=True)


@dataclass(frozen=True)
class ReasoningTokenCounter:
    """Read-only tokenizer information shared within one Case."""

    tokenizer: Any = None
    tokenizer_name: str = ""
    start_id: int | None = None
    end_id: int | None = None
    vocabulary_ids: frozenset[int] = field(default_factory=frozenset)
    unavailable_reason: str | None = None

    def valid_ids(self, value):
        return isinstance(value, list) and all(
            isinstance(token, int) and not isinstance(token, bool)
            and token in self.vocabulary_ids for token in value
        )

    def count(self, record, observation):
        """Return (count, reason, initial state) from actual IDs."""
        if self.unavailable_reason:
            return None, self.unavailable_reason, None
        if observation.token_ids_error:
            return None, observation.token_ids_error, None
        chunks = observation.output_token_id_chunks
        if not chunks:
            return None, "missing_output_token_ids", None
        if not all(self.valid_ids(chunk) for chunk in chunks):
            return None, "invalid_output_token_ids", None
        output = [token for chunk in chunks for token in chunk]
        if record["output_tokens"] is None:
            return None, "missing_completion_tokens", None
        if len(output) != record["output_tokens"]:
            return None, "output_token_ids_count_mismatch", None
        prompt = observation.prompt_token_ids
        initial = False
        if prompt is not None:
            if not self.valid_ids(prompt):
                return None, "invalid_prompt_token_ids", None
            if len(prompt) != record["input_tokens"]:
                return None, "prompt_token_ids_count_mismatch", None
            last_start = max(
                (i for i, token in enumerate(prompt)
                 if token == self.start_id), default=-1,
            )
            last_end = max(
                (i for i, token in enumerate(prompt)
                 if token == self.end_id), default=-1,
            )
            # R1 prompts may end with <think> and whitespace.
            # An arbitrary unclosed marker earlier in history is unsafe.
            initial = last_start > last_end and not self.tokenizer.decode(
                prompt[last_start + 1:], skip_special_tokens=False,
            ).strip()
        if not initial:
            try:
                start = output.index(self.start_id)
            except ValueError:
                logger.debug("Generated thinking start marker absent")
                return None, "thinking_start_unavailable", False
            if self.tokenizer.decode(
                output[:start], skip_special_tokens=False,
            ).strip():
                return None, "ambiguous_thinking_prefix", False
        inside = initial
        started = initial
        closed = False
        count = 0
        for token in output:
            if token == self.start_id:
                if started:
                    return None, "multiple_or_nested_thinking_blocks", initial
                started = inside = True
            elif token == self.end_id:
                if not inside:
                    return None, "unexpected_thinking_end", initial
                inside = False
                closed = True
            elif inside:
                count += 1
        if not closed:
            return None, "incomplete_thinking_block", initial
        return count, None, initial


def prepare_reasoning_counter(parameters, tokenizer=None):
    """Prepare optional counting resources before sending requests."""
    enabled = bool(parameters.get("thinking_metrics")) and parameters.get(
        "endpoint"
    ) == "/v1/chat/completions"
    logger.info("Phase timing configured: enabled=%s", enabled)
    if not enabled:
        return None
    name = parameters.get("tokenizer") or parameters["model"]
    reason = None
    requested = parameters.get("request_params", {}).get("return_token_ids")
    if requested is not True:
        reason = "token_ids_not_requested"
    elif tokenizer is None and not parameters.get("tokenizer"):
        reason = "tokenizer_not_configured"
    if reason:
        logger.info(
            "Thinking Token ID fallback unavailable: reason=%s", reason,
        )
        return ReasoningTokenCounter(unavailable_reason=reason)
    try:
        if tokenizer is None:
            tokenizer = load_tokenizer(parameters)
        vocabulary = tokenizer.get_vocab()
        start, end = vocabulary.get("<think>"), vocabulary.get("</think>")
        if start is None or end is None or start == end or any(
            tokenizer.encode(text, add_special_tokens=False) != [token]
            for text, token in (("<think>", start), ("</think>", end))
        ):
            reason = "unsupported_thinking_markers"
    except Exception:  # Supplemental setup must not change request failures.
        logger.warning("Thinking tokenizer preparation failed", exc_info=True)
        return ReasoningTokenCounter(
            tokenizer_name=name, unavailable_reason="tokenizer_unavailable",
        )
    logger.info(
        "Thinking Token ID fallback prepared: tokenizer=%s available=%s "
        "reason=%s", name, reason is None, reason,
    )
    return ReasoningTokenCounter(
        tokenizer=tokenizer, tokenizer_name=name, start_id=start, end_id=end,
        vocabulary_ids=frozenset(vocabulary.values()),
        unavailable_reason=reason,
    )


def supplement_reasoning_tokens(record, observation, counter):
    """Add provenance and count after E2EL is fixed; keep usage."""
    record["reasoning_tokens_source"] = None
    record["reasoning_tokens_reason"] = None
    if record["status"] != "success":
        record["reasoning_tokens_reason"] = "request_failed"
        return
    if observation.output_token_id_chunks or (
        observation.prompt_token_ids is not None
    ):
        record["token_id_observation"] = {
            "prompt_token_ids": _saved_ids(observation.prompt_token_ids),
            "output_token_id_chunks": [
                _saved_ids(chunk)
                for chunk in observation.output_token_id_chunks
            ],
            "error": observation.token_ids_error,
        }
    # The collector validates service usage, including valid zero.
    if record["reasoning_tokens"] is not None:
        record["reasoning_tokens_source"] = "service_usage"
        return
    if counter is None:
        record["reasoning_tokens_reason"] = "token_ids_not_requested"
        return
    try:
        count, reason, initial = counter.count(record, observation)
    except Exception:
        logger.warning(
            "Thinking Token ID counting failed: index=%s",
            record["request_index"], exc_info=True,
        )
        count, reason, initial = None, "tokenizer_counting_failed", None
    record["reasoning_tokens"] = count
    record["reasoning_tokens_source"] = (
        "token_ids" if count is not None else None
    )
    record["reasoning_tokens_reason"] = reason
    record["reasoning_token_counting"] = {
        "tokenizer": counter.tokenizer_name,
        "start_token_id": counter.start_id, "end_token_id": counter.end_id,
        "starts_in_thinking": initial,
        "method": "single_think_block_excluding_markers",
    }
