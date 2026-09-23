"""Independent metric extensions over persisted facts."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping

from luban_meter.core.registry import Registry
from luban_meter.core.template_contracts import validate_fields


class Metric(ABC):
    """Metrics never execute a model or a candidate program."""

    requires: frozenset[str] = frozenset()
    required_fields: dict = {}
    description = ""

    @abstractmethod
    def compute(self, facts: Mapping) -> Mapping:
        """Return named values with units and sample counts."""


metrics = Registry("metric", Metric)


def compute_metrics(names, capabilities, facts):
    results = {}
    for name in names:
        metric = metrics.get(name)()
        missing = metric.requires - set(capabilities)
        if missing:
            raise ValueError(f"{name} requires facts: {sorted(missing)}")
        for sample in facts["samples"]:
            if sample["state"] == "complete":
                validate_fields(
                    sample["facts"], metric.required_fields,
                    f"metric {name} sample "
                    f"{sample.get('sample_id', 'unknown')}",
                )
        result = metric.compute(facts)
        if not isinstance(result, Mapping):
            raise TypeError(f"{name}.compute must return a mapping")
        json.dumps(result, allow_nan=False)
        results[name] = dict(result)
    return results
