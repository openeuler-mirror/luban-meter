"""Cache counts are measured pairs; missing evidence is never zero."""

from copy import deepcopy

import pytest

from luban_meter.benchmarking.generation_performance.common import (
    cache_usage,
)


def usage(prompt=64, cached=32):
    return {
        "prompt_tokens": prompt,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


@pytest.mark.parametrize("prompt, cached", [
    (64, 0), (64, 32), (64, 64), (0, 0),
])
def test_valid_pairs_preserve_zero_without_computing_a_rate(prompt, cached):
    result = cache_usage.parse_cache_usage(usage(prompt, cached))
    assert result.state == "valid"
    assert result.prompt_tokens == prompt
    assert result.cached_tokens == cached
    assert result.reason is None


@pytest.mark.parametrize("value, prompt, reason", [
    (None, None, "missing_usage"),
    ({}, None, "missing_prompt_tokens"),
    ({"prompt_tokens": None}, None, "missing_prompt_tokens"),
    ({"prompt_tokens_details": {"cached_tokens": 0}},
     None, "missing_prompt_tokens"),
    ({"prompt_tokens": 64}, 64, "missing_prompt_tokens_details"),
    ({"prompt_tokens": 64, "prompt_tokens_details": None},
     64, "missing_prompt_tokens_details"),
    ({"prompt_tokens": 64, "prompt_tokens_details": {}},
     64, "missing_cached_tokens"),
    (usage(cached=None), 64, "missing_cached_tokens"),
    ({"prompt_tokens": 64, "cached_tokens": 0},
     64, "missing_prompt_tokens_details"),
])
def test_missing_is_not_zero(value, prompt, reason):
    result = cache_usage.parse_cache_usage(value)
    assert result.state == "missing"
    assert result.prompt_tokens == prompt
    assert result.cached_tokens is None
    assert result.reason == reason


@pytest.mark.parametrize("value", [[], "", 0, False])
def test_usage_must_be_an_object(value):
    result = cache_usage.parse_cache_usage(value)
    assert result.state == "invalid"
    assert result.reason == "invalid_usage"
    assert result.prompt_tokens is None
    assert result.cached_tokens is None


@pytest.mark.parametrize("value", [
    -1, 0.0, 1.5, "0", True, False, [], {}, float("nan"), float("inf"),
])
@pytest.mark.parametrize("field", ["prompt_tokens", "cached_tokens"])
def test_counts_reject_invalid_values(field, value):
    data = usage()
    if field == "prompt_tokens":
        data[field] = value
    else:
        data["prompt_tokens_details"][field] = value
    result = cache_usage.parse_cache_usage(data)
    assert result.state == "invalid"
    assert result.reason == f"invalid_{field}"
    assert result.prompt_tokens == (None if field == "prompt_tokens" else 64)
    assert result.cached_tokens is None


@pytest.mark.parametrize("details", [[], "", 0, False])
def test_details_must_be_an_object(details):
    result = cache_usage.parse_cache_usage({
        "prompt_tokens": 64, "prompt_tokens_details": details,
    })
    assert result.state == "invalid"
    assert result.reason == "invalid_prompt_tokens_details"
    assert result.prompt_tokens == 64
    assert result.cached_tokens is None


@pytest.mark.parametrize("prompt, cached", [(64, 65), (0, 1)])
def test_cached_count_is_not_clamped(prompt, cached):
    data = usage(prompt, cached)
    result = cache_usage.parse_cache_usage(data)
    assert result.state == "invalid"
    assert result.reason == "cached_tokens_exceed_prompt_tokens"
    assert result.cached_tokens is None
    assert data["prompt_tokens_details"]["cached_tokens"] == cached


@pytest.mark.parametrize("data, reason", [
    ({"prompt_tokens": -1}, "invalid_prompt_tokens"),
    ({"prompt_tokens_details": []}, "invalid_prompt_tokens_details"),
    ({"prompt_tokens_details": {"cached_tokens": True}},
     "invalid_cached_tokens"),
])
def test_invalid_fields_take_precedence_over_missing_fields(data, reason):
    result = cache_usage.parse_cache_usage(data)
    assert result.state == "invalid"
    assert result.reason == reason


def test_parsing_does_not_modify_usage_or_consume_other_token_counts():
    data = {
        **usage(), "completion_tokens": 999,
        "prompt_tokens_details": {"cached_tokens": 32, "audio_tokens": 7},
        "vendor": {"items": [1, 2]},
    }
    original = deepcopy(data)
    result = cache_usage.parse_cache_usage(data)
    assert data == original
    assert (result.prompt_tokens, result.cached_tokens) == (64, 32)
