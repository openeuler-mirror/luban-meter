"""Thinking controls and response facts shared by HTTP benchmarks."""

from collections.abc import Mapping


def thinking_defaults(supplied, *, endpoint, eval_mode="gen"):
    """Default chat generation to non-thinking; preserve explicit null."""
    result = dict(supplied)
    if (
        "enable_thinking" not in result
        and result.get("endpoint", endpoint) == "/v1/chat/completions"
        and result.get("eval_mode", eval_mode) == "gen"
    ):
        result["enable_thinking"] = False
    return result


def thinking_payload(enable_thinking=None, thinking_protocol="chat_template"):
    """Build an explicit protocol extension; null leaves service defaults."""
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise ValueError("enable_thinking must be a boolean or null")
    if thinking_protocol not in ("chat_template", "deepseek"):
        raise ValueError("thinking_protocol must be chat_template or deepseek")
    if enable_thinking is None:
        return {}
    if thinking_protocol == "deepseek":
        return {
            "thinking": {"type": "enabled" if enable_thinking else "disabled"}
        }
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


def validate_thinking(parameters):
    """Reject explicit controls outside chat generation before execution."""
    enabled = parameters.get("enable_thinking")
    thinking_payload(
        enabled, parameters.get("thinking_protocol", "chat_template")
    )
    if enabled is not None and (
        parameters["endpoint"] != "/v1/chat/completions"
        or parameters.get("eval_mode", "gen") != "gen"
    ):
        raise ValueError("enable_thinking requires chat endpoint and gen mode")


def reasoning_text(message):
    """Read either service alias without silently duplicating reasoning."""
    values = []
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string or null")
        values.append(value)
    if len(set(values)) > 1:
        raise ValueError("conflicting reasoning and reasoning_content fields")
    return values[0] if values else ""


def reasoning_token_count(usage):
    """Preserve reported reasoning counts without estimating missing ones."""
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("completion_tokens_details")
    if details is None:
        return None
    if not isinstance(details, Mapping):
        raise ValueError("completion_tokens_details must be an object")
    count = details.get("reasoning_tokens")
    if count is not None and (
        not isinstance(count, int) or isinstance(count, bool) or count < 0
    ):
        raise ValueError("reasoning_tokens must be a non-negative integer")
    return count


def response_facts(response):
    """Copy chat diagnostics into quality facts without changing scoring."""
    return {
        name: response[name]
        for name in (
            "reasoning_text", "reasoning_tokens", "has_content",
            "finish_reason", "usage",
        )
        if name in response
    }
