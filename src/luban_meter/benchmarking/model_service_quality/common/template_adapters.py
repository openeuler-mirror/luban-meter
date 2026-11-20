"""Registered adapters around existing domain prompt and answer functions."""

from luban_meter.core.template_contracts import (
    AnswerParser,
    InputTemplate,
    answer_parsers,
    input_templates,
)

from .parsers import extract_choice, extract_number
from .prompts import (
    render_choice_generation,
    render_choice_ppl,
    render_lcsts_override,
    render_lcsts_prompt,
    render_math_prompt,
    render_squad_prompt,
)


@input_templates.register("choice")
class ChoiceTemplate(InputTemplate):
    requires = {"question": str, "choices": list}

    def render(self, record, examples, parameters):
        return render_choice_input(record, examples, parameters)


@input_templates.register("numeric_qa")
class NumericTemplate(InputTemplate):
    requires = {"question": str}

    def render(self, record, examples, parameters):
        return {
            "prompt": render_math_prompt(record, few_shot_samples=examples)
        }


@input_templates.register("summarization")
class SummaryTemplate(InputTemplate):
    requires = {"content": str}

    def render(self, record, examples, parameters):
        return render_summary_input(record, examples, parameters)


@input_templates.register("reading")
class ReadingTemplate(InputTemplate):
    requires = {"question": str, "context": str}

    def render(self, record, examples, parameters):
        return {"prompt": render_squad_prompt(record)}


@answer_parsers.register("choice")
class ChoiceParser(AnswerParser):
    provides = {"prediction": (str, type(None))}

    def parse(self, fields):
        return {"prediction": extract_choice(fields["raw_output"])}


@answer_parsers.register("number")
class NumberParser(AnswerParser):
    provides = {"prediction": (float, type(None))}

    def parse(self, fields):
        return {"prediction": extract_number(fields["raw_output"])}


@answer_parsers.register("text")
class TextParser(AnswerParser):
    def parse(self, fields):
        return {"prediction": fields["raw_output"]}


@answer_parsers.register("reading")
class ReadingParser(AnswerParser):
    def parse(self, fields):
        from ..reading_comprehension import parse_answer

        return {"prediction": parse_answer(fields["raw_output"])}


def render_choice_input(record, examples, config):
    if config.get("eval_mode") == "ppl":
        return {"prompt": render_choice_ppl(
            record, few_shot_samples=examples
        )}
    return render_choice_generation(record, examples)


def render_summary_input(record, examples, config):
    if config.get("prompt_template"):
        return {"prompt": render_lcsts_override(
            record, few_shot_samples=examples,
            template=config["prompt_template"],
        )}
    return render_lcsts_prompt(record, examples)


def render_input(config, sample, examples, fallback):
    pipeline = config.get("_pipeline")
    if pipeline:
        return pipeline.render(sample, examples, config)
    rendered = fallback()
    return {"prompt": rendered} if isinstance(rendered, str) else rendered


def parse_output(config, sample, rendered, text, fallback):
    pipeline = config.get("_pipeline")
    if pipeline:
        return pipeline.parse(sample, rendered, text)["prediction"]
    return fallback(text)
