"""Append service extensions without changing benchmark request contracts."""

import json

CORE_REQUEST_FIELDS = frozenset({
    "model", "messages", "prompt", "stream", "stream_options",
    "max_tokens", "max_completion_tokens", "max_output_tokens", "min_tokens",
    "temperature", "seed", "stop", "ignore_eos", "add_special_tokens",
    "n", "best_of", "echo", "logprobs",
})


def validate_request_params(value):
    """Validate JSON extensions and return an independent nested object."""
    if not isinstance(value, dict):
        raise ValueError("request_params must be a JSON object")
    conflicts = CORE_REQUEST_FIELDS.intersection(value)
    if conflicts:
        raise ValueError(
            "request_params cannot override core fields: "
            + ", ".join(sorted(conflicts))
        )
    try:
        decoded = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "request_params must contain finite JSON values"
        ) from exc
    if decoded != value:
        raise ValueError(
            "request_params must contain JSON values with string object keys"
        )
    return decoded


def append_request_params(payload, extensions):
    """Merge at the top level; no overrides or implicit deep merging."""
    extra = validate_request_params(extensions)
    conflicts = payload.keys() & extra.keys()
    if conflicts:
        raise ValueError(
            "request_params conflicts with request fields: "
            + ", ".join(sorted(conflicts))
        )
    return {**payload, **extra}
