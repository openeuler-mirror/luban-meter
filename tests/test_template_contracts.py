"""Composable templates reject incompatible fields before model calls."""

import pytest

from luban_meter.api import AnswerParser, answer_parsers
from luban_meter.core.planning import bootstrap, build_plan
from luban_meter.core.template_contracts import TemplatePipeline
from tests.test_unified_runtime import service as _service


@pytest.fixture
def service(tmp_path):
    yield from _service.__wrapped__(tmp_path)


bootstrap()


def test_template_parser_contract_mismatch():
    with pytest.raises(ValueError, match="prediction"):
        TemplatePipeline("numeric_qa", "text", {"question": str}, float)
    with pytest.raises(ValueError, match="choices"):
        TemplatePipeline("choice", "choice", {"question": str}, str)


def test_custom_parser_and_field_mapping():
    @answer_parsers.register("test_integer_answer")
    class IntegerParser(AnswerParser):
        provides = {"prediction": float}

        def parse(self, fields):
            return {"prediction": float(fields["raw_output"].split(":")[1])}

    pipeline = TemplatePipeline(
        "format_string", "test_integer_answer", {}, float
    )
    p = {"input_format": "Solve {question}", "field_mapping": ["question=q"]}
    rendered = pipeline.render({"q": "2+2"}, [], p)
    assert rendered == {"prompt": "Solve 2+2"}
    assert pipeline.parse({}, rendered, "answer:4") == {"prediction": 4.0}
    with pytest.raises(ValueError, match="missing field"):
        pipeline.render({}, [], p)


def test_registered_template_output_is_checked():
    from luban_meter.api import InputTemplate, input_templates

    @input_templates.register("test_broken_output")
    class BrokenTemplate(InputTemplate):
        def render(self, record, examples, parameters):
            return {"prompt": 42}

    pipeline = TemplatePipeline("test_broken_output", "text", {}, str)
    with pytest.raises(TypeError, match="template output"):
        pipeline.render({}, [], {})


def test_plan_rejects_incompatible_parser_before_http(tmp_path):
    path = tmp_path / "math.jsonl"
    path.write_text('{"question":"2+2?", "answer":"#### 4"}\n')
    with pytest.raises(ValueError, match="prediction"):
        build_plan(
            {
                "scenario": "gsm8k",
                "model": "unused",
                "dataset_path": str(path),
                "answer_parser": "text",
            }
        )


def test_metric_field_contract_rejected_during_planning(tmp_path):
    from luban_meter.api import Metric, metrics

    @metrics.register("test_missing_metric_field")
    class ExtraMetric(Metric):
        required_fields = {"missing_field": float}

        def compute(self, facts):
            raise AssertionError("must fail before execution")

    path = tmp_path / "records.jsonl"
    path.write_text('{"question":"2+2", "answer":"#### 4"}\n')
    with pytest.raises(ValueError, match="missing_field"):
        build_plan(
            {
                "scenario": "gsm8k",
                "model": "unused",
                "dataset_path": str(path),
                "metrics": ["test_missing_metric_field"],
            }
        )


def test_quality_template_configuration_runs_in_worker(service, tmp_path):
    import json
    from pathlib import Path

    from luban_meter.core.run_coordinator import RunCoordinator
    from tests.test_unified_runtime import Handler

    path = tmp_path / "choice.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "pick",
                "choices": ["a", "b", "c", "d"],
                "answer": "A",
                "subject": "test",
                "id": "1",
            }
        )
        + "\n"
    )
    result, _ = RunCoordinator().run(
        build_plan(
            {
                "scenario": "ceval",
                "base_url": service,
                "model": "mock",
                "dataset_path": str(path),
                "input_template": "format_string",
                "input_format": "Question: {q}",
                "field_mapping": ["q=question"],
                "answer_parser": "choice",
            }
        ),
        tmp_path / "out",
    )
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    raw = json.loads(Path(case["raw"]).read_text())
    assert raw["samples"][0]["facts"]["correct"]
    assert Handler.posts[-1][1]["prompt"] == "Question: pick"
