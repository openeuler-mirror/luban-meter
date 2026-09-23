"""Run existing domain fixtures through the new public scenario runtime.

Only tests translate the former fixture vocabulary. Production has no
legacy config loader or script adapter.
"""

import json
import tempfile
from pathlib import Path

from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import bootstrap, build_plan, schema_for
from luban_meter.runtime.evaluation_session import run_case


def collect_domain(name, request, parameters):
    bootstrap()
    implementation = benchmarks.get(name)
    domain = implementation.domain_module
    if name in ("ceval", "cmmlu"):
        spec = domain.ChoiceBenchmarkDefinition(
            name, f"{name}_choice_accuracy_online_service", name
        )
        parameters = domain.validate_choice_parameters(parameters, spec)
    else:
        parameters = domain.validate_parameters(parameters)
    supplied = dict(parameters)
    supplied.update(
        model=request.get("model_name", "test-model"),
        base_url=parameters.get("service_url", "http://127.0.0.1:8000"),
        endpoint="/v1/chat/completions"
        if parameters.get("prompt_format", "base") == "chat"
        else "/v1/completions",
    )
    if "max_tokens" in parameters:
        supplied["output_tokens"] = parameters["max_tokens"]
    schema = schema_for(implementation, supplied)
    supplied = {k: v for k, v in supplied.items() if k in schema}
    plan = build_plan({"scenario": name, **supplied})
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "raw").mkdir()
        result = run_case(
            implementation,
            plan["tasks"][0]["cases"][0],
            directory,
            directory / "stop",
        )
        assert "raw" in result, result
        facts = json.loads(Path(result["raw"]).read_text())
    samples = [row["facts"] for row in facts["samples"]]
    counts = {"total": len(samples)}
    if name in ("ceval", "cmmlu", "gsm8k"):
        counts["correct"] = sum(bool(r.get("correct")) for r in samples)
    if name == "humaneval":
        counts["passed"] = sum(bool(r.get("passed")) for r in samples)
    for status in ("parse_failed", "service_failed"):
        if status == "parse_failed" and name in ("lcsts", "wikitext"):
            continue
        counts[status] = sum(r["status"] == status for r in samples)
    if name == "humaneval":
        counts["sandbox_failed"] = sum(
            r["status"] == "sandbox_failed" for r in samples
        )
    metadata = facts["metadata"]
    module = domain
    if hasattr(module, "MEASUREMENT"):
        metadata["measurement"] = module.MEASUREMENT
    return {
        "status": "success",
        "metrics": {"samples": samples, "counts": counts},
        "metadata": metadata,
    }
