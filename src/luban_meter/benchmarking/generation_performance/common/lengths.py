"""Workload conformance is independent of request success and metrics."""

from .statistics import summarize


def assess_lengths(sample, actual_input, actual_output, parameters):
    target_input = sample.get("target_input_tokens")
    target_output = sample["target_output_tokens"]
    strict = parameters["length_policy"] == "strict"
    lower = (
        parameters.get("input_tokens_min") if target_input is None else None
    )
    upper = (
        parameters.get("input_tokens_max") if target_input is None else None
    )
    checks = [None] if actual_input is None else []
    if strict and target_input is not None:
        checks.append(
            None if actual_input is None else actual_input == target_input
        )
    if lower is not None or upper is not None:
        checks.append(
            None
            if actual_input is None
            else (
                (lower is None or actual_input >= lower)
                and (upper is None or actual_input <= upper)
            )
        )
    checks.append(
        None
        if actual_output is None
        else (
            actual_output == target_output
            if parameters["ignore_eos"]
            else actual_output <= target_output
        )
    )
    state = (
        "nonconforming"
        if False in checks
        else "unknown"
        if None in checks
        else "conforming"
    )
    return {
        "policy": parameters["length_policy"],
        "target_input_tokens": target_input,
        "local_input_tokens": sample.get("local_input_tokens"),
        "target_output_tokens": target_output,
        "actual_input_tokens": actual_input,
        "actual_output_tokens": actual_output,
        "input_delta": (
            actual_input - target_input
            if actual_input is not None and target_input is not None
            else None
        ),
        "output_delta": (
            actual_output - target_output
            if actual_output is not None
            else None
        ),
        "input_min": lower,
        "input_max": upper,
        "condition": state,
    }


def length_summary(records):
    rows = [r["lengths"] for r in records if "lengths" in r]
    counts = {
        state: sum(r["condition"] == state for r in rows)
        for state in ("conforming", "nonconforming", "unknown")
    }
    known = counts["conforming"] + counts["nonconforming"]
    return {
        "counts": counts,
        "conformance_rate": {
            "value": counts["conforming"] / known if known else None,
            "unit": "ratio",
            "sample_count": known,
            "condition": "known conditions; independent of request state",
        },
        **{
            name: summarize(
                [r[name] for r in rows if r.get(name) is not None], "token"
            )
            for name in (
                "target_input_tokens",
                "target_output_tokens",
                "local_input_tokens",
                "actual_input_tokens",
                "actual_output_tokens",
                "input_delta",
                "output_delta",
            )
        },
    }
