"""Typed contracts for independently registered input and answer adapters."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from string import Formatter

from luban_meter.core.registry import Registry


def validate_fields(value, fields, label):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label}: expected object")
    for name, expected in fields.items():
        if name not in value:
            raise ValueError(f"{label}: missing field {name}")
        field = value[name]
        if not isinstance(field, expected) or (
            isinstance(field, bool) and expected in (int, float)
        ):
            raise TypeError(f"{label}: invalid type for {name}")


def compatible(provided, required):
    for name, expected in required.items():
        actual = provided.get(name)
        types = actual if isinstance(actual, tuple) else (actual,)
        if actual is None or not all(issubclass(t, expected) for t in types):
            raise ValueError(f"incompatible field contract: {name}")


class InputTemplate(ABC):
    requires = {}
    provides = {"prompt": str}

    @abstractmethod
    def render(self, record, examples, parameters):
        """Return prompt and any declared parser context fields."""


class AnswerParser(ABC):
    requires = {"raw_output": str}
    provides = {"prediction": str}

    @abstractmethod
    def parse(self, fields):
        """Return declared answer fields without calculating metrics."""


input_templates = Registry("input template", InputTemplate)
answer_parsers = Registry("answer parser", AnswerParser)


class TemplatePipeline:
    def __init__(self, template, parser, record_fields, prediction_type):
        self.template = input_templates.get(template)()
        self.parser = answer_parsers.get(parser)()
        self.parser_name = parser
        compatible(self.template.provides, {"prompt": str})
        compatible(record_fields, self.template.requires)
        compatible(
            {**record_fields, **self.template.provides, "raw_output": str},
            self.parser.requires,
        )
        compatible(self.parser.provides, {"prediction": prediction_type})

    def render(self, record, examples, parameters):
        validate_fields(record, self.template.requires, "input record")
        rendered = self.template.render(record, examples, parameters)
        validate_fields(rendered, self.template.provides, "template output")
        return rendered

    def parse(self, record, rendered, text):
        fields = {**record, **rendered, "raw_output": text}
        validate_fields(fields, self.parser.requires, "parser input")
        result = self.parser.parse(fields)
        validate_fields(result, self.parser.provides, "parser output")
        return result


@input_templates.register("format_string")
class FormatStringTemplate(InputTemplate):
    def render(self, record, examples, parameters):
        template = parameters["input_format"]
        if not template:
            raise ValueError("format_string requires input_format")
        mapping = {}
        for entry in parameters.get("field_mapping", []):
            key, separator, source = entry.partition("=")
            if not separator or not key or not source or key in mapping:
                raise ValueError("field_mapping expects unique name=field")
            mapping[key] = source
        values = {}
        for _, name, spec, conversion in Formatter().parse(template):
            if name is None:
                continue
            if not name.isidentifier() or spec or conversion:
                raise ValueError("input_format only allows {field} entries")
            source = mapping.get(name, name)
            if source not in record:
                raise ValueError(f"input template missing field: {source}")
            values[name] = record[source]
        return {"prompt": template.format_map(values)}


@input_templates.register("prompt_record")
class PromptRecordTemplate(InputTemplate):
    def render(self, record, examples, parameters):
        from luban_meter.data.projections import extract_prompt

        return {"prompt": extract_prompt(
            record, parameters.get("prompt_field", "prompt"), strip=False
        )}
