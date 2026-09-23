"""Shared trace text construction for HTTP and Engine backends."""

from luban_meter.core.parameters import Parameter
from luban_meter.data import DatasetLoader
from luban_meter.data.trace_adapter import select_trace

TRACE_PARAMETERS = {
    "trace_start": Parameter(int, 0, minimum=0),
    "skip_invalid": Parameter(bool, False),
}


def validate_trace_parameters(parameters):
    """Reject conflicting trace limits before loading resources."""
    if parameters["data"] != "burstgpt":
        return
    maximum = parameters.get("max_samples")
    requested = parameters["num_requests"]
    if maximum is not None and maximum < requested:
        raise ValueError(
            f"burstgpt max_samples ({maximum}) must be at least "
            f"num_requests ({requested})"
        )


def prepare_trace(context, generator, batch_size=1):
    p = context.parameters
    errors = []
    records = DatasetLoader(
        p["dataset_path"],
        format="burstgpt",
        skip_invalid=p["skip_invalid"],
        diagnostics=errors,
    ).load()
    selected = select_trace(records, p)
    prepared = {}
    for phase, count in (
        ("formal", len(selected)),
        ("warmup", p["num_warmups"] * batch_size),
    ):
        prepared[phase] = []
        for index in range(count):
            if context.stopped():
                raise InterruptedError("trace preparation interrupted")
            source = selected[index % len(selected)]
            sample = generator.sample(
                index,
                phase,
                input_tokens=source["input_tokens"],
                output_tokens=source["output_tokens"],
            )
            sample["record"] = source
            if phase == "formal" and "scheduling" in source:
                sample["scheduling"] = source["scheduling"]
            prepared[phase].append(sample)
    context.total_samples = len(selected)
    return prepared, {
        "trace_mode": "replay" if p["profile"] == "replay" else "lengths",
        "selected_records": len(selected),
        "skipped_records": len(errors),
        "skip_reasons": errors,
        "replay_speed": p.get("replay_speed"),
        "original_duration_s": (
            max(r["offset_s"] for r in selected)
            - min(r["offset_s"] for r in selected)
        ),
        "input_tokens_range": [
            min(r["input_tokens"] for r in selected),
            max(r["input_tokens"] for r in selected),
        ],
        "output_tokens_range": [
            min(r["output_tokens"] for r in selected),
            max(r["output_tokens"] for r in selected),
        ],
    }
