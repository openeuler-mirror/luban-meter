"""Stage-four evidence checks, without network or model dependencies."""

import copy

import pytest

from luban_meter.benchmarking.benchmark_metrics import AnswerOutputQuality
from luban_meter.benchmarking.model_service_quality.common import (
    prompt_protocol as protocol,
)
from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import build_plan
from luban_meter.reporting.html import report_data
from luban_meter.reporting.terminal import render_text


def diagnostic_record(status, output, *, version="ceval-v4", mode="gen"):
    sample = {"question": "q", "choices": list("abcd"), "answer": "A"}
    record = {"status": status, "raw_output": output,
              "final_output": output, "prediction": "A"
              if status == "success" else None, "correct": status == "success"}
    config = {"_protocol": {
        "input_template": "choice", "prompt_source": "builtin",
        "prompt_version": version, "answer_parser": "choice",
    }, "eval_mode": mode, "prompt_format": "chat"}
    return protocol.annotate_record(
        record, sample, [], config, {"prompt": "q"}
    )


def test_diagnostic_denominators_and_task_facts_are_independent():
    pairs = [("success", "答案：A"), ("success", "解释\n答案：A"),
             ("parse_failed", ""), ("parse_failed", " \n"),
             ("parse_failed", "无法回答"), ("service_failed", None)]
    rows = [diagnostic_record(status, text) for status, text in pairs]
    before = copy.deepcopy(rows)
    facts = {"samples": [{"state": "complete", "facts": row} for row in rows],
             "metadata": {"evaluation": {"generation": {"max_tokens": 128}}}}
    result = AnswerOutputQuality().compute(facts)
    values = result["values"]
    assert values["answer_parse_rate"]["value"] == 2 / 5
    assert values["parse_failure_rate"]["value"] == 3 / 5
    assert values["invalid_output_rate"]["value"] == 3 / 5
    assert values["format_compliance_rate"]["value"] == 1 / 5
    assert values["format_compliance_rate"]["sample_count"] == 5
    assert values["parse_failures"]["empty_output"]["value"] == 2
    assert rows == before
    case = {"id": "case_1", "status": "success", "parameters": {},
            "metrics": {"answer_output_quality": result}}
    payload = {"kind": "task", "name": "diagnostics", "scenario": "ceval",
               "category": "model_service_quality", "status": "success",
               "cases": [case]}
    rendered = report_data(payload)
    assert not any("/metadata/" in row["metric"] for row in rendered["rows"])
    for key in ("answer_parse_rate", "format_compliance_rate",
                "parse_failure_rate", "invalid_output_rate"):
        assert sum(row["metric"].endswith("/" + key)
                   for row in rendered["rows"]) == 1
    text = render_text(payload, verbose=True, width=140)
    assert "0.4" in text and "0.6" in text and "ratio" in text
    for key in ("answer_parse_rate", "format_compliance_rate",
                "parse_failure_rate", "invalid_output_rate"):
        assert text.count(key) == 1


@pytest.mark.parametrize(("version", "mode"), [
    ("humaneval-completion-v1", "gen"), ("wikitext-v1", "loss"),
    ("ceval-v1", "ppl"),
])
def test_nonapplicable_format_has_no_denominator(version, mode):
    row = diagnostic_record("success", "answer", version=version, mode=mode)
    assert row["format_compliant"] is None
    result = AnswerOutputQuality().compute({
        "samples": [{"state": "complete", "facts": row}], "metadata": {},
    })
    assert result["values"]["format_compliance_rate"]["value"] is None
    assert result["values"]["format_compliance_rate"]["sample_count"] == 0
    if mode == "loss":
        assert result["status"] == "unavailable"
        assert result["values"]["answer_parse_rate"]["value"] is None


def test_service_failures_do_not_become_empty_or_parsed_answers():
    row = diagnostic_record("service_failed", None)
    assert row["parse_diagnostic"]["status"] == "service_failed"
    result = AnswerOutputQuality().compute({
        "samples": [{"state": "failed", "facts": row}], "metadata": {},
    })
    assert result["status"] == "unavailable"
    assert result["values"]["parse_failures"] == {}
    assert result["values"]["answer_parse_rate"]["sample_count"] == 0
    assert result["values"]["answer_parse_rate"]["value"] is None


@pytest.mark.parametrize(("scenario", "version", "parser"), [
    ("ceval", "ceval-v4", "choice"),
    ("cmmlu", "cmmlu-v4", "choice"),
    ("gsm8k", "gsm8k-v1", "number"),
    ("lcsts", "lcsts-v2", "text"),
    ("squad", "squad2-gen-v1", "reading"),
    ("humaneval", "humaneval-completion-v1", "humaneval_completion"),
    ("wikitext", "wikitext-v1", None),
])
def test_internal_identity_matches_each_default(
    tmp_path, scenario, version, parser
):
    path = tmp_path / "unused.jsonl"
    path.write_text('{}\n')
    plan = build_plan({"scenario": scenario, "model": "unused",
                       "dataset_path": str(path)})
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    config = benchmarks.get(scenario).domain_parameters(parameters)
    assert config["prompt_version"] == version
    assert config["answer_parser"] == parser


@pytest.mark.parametrize("scenario", ["ceval", "cmmlu"])
def test_ppl_omitted_version_stays_v1(tmp_path, scenario):
    path = tmp_path / "unused.jsonl"
    path.write_text('{}\n')
    plan = build_plan({"scenario": scenario, "model": "unused",
                       "dataset_path": str(path), "eval_mode": "ppl"})
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    config = benchmarks.get(scenario).domain_parameters(parameters)
    assert config["prompt_version"] == f"{scenario}-v1"
    assert config["answer_parser"] is None
