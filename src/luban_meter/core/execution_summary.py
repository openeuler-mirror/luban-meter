"""Public Case summaries assembled from extension declarations."""

from luban_meter.core.parameters import public_parameters
from luban_meter.core.planning import schema_for
from luban_meter.runtime.load_profiles import profiles


def case_summary(scenario, parameters):
    public = public_parameters(parameters, schema_for(scenario, parameters))
    profile = profiles.get(parameters["profile"])
    fields = set(scenario.summary_fields) | set(profile.summary_fields)
    return {name: public[name] for name in sorted(fields) if name in public}


def prepared_summary(context, scenario):
    metadata = dict(scenario.metadata(context))
    # Keep full diagnostics in Case metadata and bound console output.
    if "skip_reasons" in metadata:
        metadata["skip_reason_examples"] = metadata.pop("skip_reasons")[:20]
    return {"selected_samples": context.total_samples, **metadata}
