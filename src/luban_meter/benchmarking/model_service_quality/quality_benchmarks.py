"""Quality scenarios share input, scheduling and domain rules."""

from __future__ import annotations

import threading

from luban_meter.api import (
    ModelServiceQualityBenchmark,
    Parameter,
    benchmarks,
)
from luban_meter.benchmarking.benchmark_parameters import (
    DATA,
    FEW_SHOT,
    GENERATION,
    HTTP,
    SANDBOX,
)
from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
)
from luban_meter.core.template_contracts import TemplatePipeline
from luban_meter.data import DatasetLoader
from luban_meter.data.engine import select_few_shot
from luban_meter.data.sources import sources
from luban_meter.runtime.http_client import http_settings
from luban_meter.runtime.request_params import validate_request_params

from . import (
    arithmetic_qa,
    code_generation,
    language_modeling,
    reading_comprehension,
    text_summarization,
)
from .common import choice


class QualityBenchmark(ModelServiceQualityBenchmark):
    parameters = {
        **HTTP,
        **DATA,
        **GENERATION,
        "profile": Parameter(str, "concurrent"),
        "eval_mode": Parameter(str, "gen", choices=("gen",)),
        "template": Parameter(str, ""),
        "input_template": Parameter(str, ""),
        "answer_parser": Parameter(str, ""),
        "input_format": Parameter(str, ""),
        "field_mapping": Parameter(list, [], item_type=str),
    }
    domain_module = None
    default_input_template = ""
    default_answer_parser = ""
    record_fields = {}
    prediction_type = str

    @classmethod
    def pipeline(cls, parameters):
        template = (
            parameters.get("input_template") or cls.default_input_template
        )
        parser = parameters.get("answer_parser") or cls.default_answer_parser
        if not cls.default_input_template:
            if template or parser:
                raise ValueError(
                    "this specialized workflow owns its templates"
                )
            return None
        if parameters.get("eval_mode") == "ppl" and parameters.get(
            "answer_parser"
        ):
            raise ValueError(
                "ppl workflow does not use a generated answer parser"
            )
        return TemplatePipeline(
            template, parser, cls.record_fields, cls.prediction_type
        )

    @classmethod
    def provided_fields(cls, parameters):
        pipeline = cls.pipeline(parameters)
        return {
            **cls.fact_fields,
            **({"prediction": cls.prediction_type} if pipeline else {}),
        }

    @classmethod
    def _prompt_format(cls, parameters):
        endpoint = cls.parameter_schema()["endpoint"].validate(
            "endpoint", parameters["endpoint"],
        )
        return {"chat": "chat", "completion": "base"}[endpoint]

    @classmethod
    def domain_parameters(cls, parameters):
        values = dict(parameters)
        values.update(
            service_url=parameters["base_url"],
            max_tokens=parameters["output_tokens"],
            prompt_format=cls._prompt_format(parameters),
            max_concurrency=parameters.get("max_concurrency", 8),
        )
        return cls.domain_module.validate_parameters(values)

    @classmethod
    def validate(cls, parameters):
        http_settings(parameters)
        validate_request_params(parameters["request_params"])
        sources.get(parameters["data"])
        cls.domain_parameters(parameters)
        cls.pipeline(parameters)

    def prepare_records(self, records):
        module = self.domain_module
        if hasattr(module, "prepare_samples"):
            return module.prepare_samples(records)
        if hasattr(module, "validate_records"):
            module.validate_records(records)
        return records

    def prepare(self, context):
        self.config = self.domain_parameters(context.parameters)
        self.config.update(
            {
                key: context.parameters[key]
                for key in ("input_format", "field_mapping", "answer_parser")
            }
        )
        self.config["_pipeline"] = self.pipeline(context.parameters)
        self.records = self.prepare_records(
            DatasetLoader(
                context.parameters["dataset_path"],
                format=context.parameters["data"],
                max_samples=context.parameters.get("max_samples"),
                shuffle=context.parameters["shuffle"],
                seed=context.parameters["seed"],
            ).load()
        )
        if not self.records:
            raise ValueError("dataset contains no samples")
        context.total_samples = len(self.records)
        self.http_settings = http_settings(context.parameters)
        self.client = OpenAIClient(
            context.parameters["base_url"],
            served_model_name=context.parameters["model"],
            api_key=context.parameters["api_key"],
            timeout=context.parameters["request_timeout"],
            request_runner=context.request,
            request_params=context.parameters["request_params"],
            max_connections=self.http_settings["max_connections"],
            keepalive_expiry=self.http_settings["keepalive_expiry"],
        )
        self.few_shot = []
        if self.config.get("few_shot", 0):
            self.few_shot = self.prepare_records(
                DatasetLoader(self.config["few_shot_path"]).load()
            )

        if self.config["_pipeline"]:
            for record in self.records:
                self.config["_pipeline"].render(
                    record, self.few_shot, self.config
                )

    def close(self):
        if getattr(self, "client", None) is not None:
            self.client.close()

    def samples(self, context):
        return iter(self.records)

    def evaluate(self, sample, context):
        return self.domain_module.collect_sample(
            self.client, sample, self.config
        )

    def metadata(self, context):
        return {
            "http_client": self.http_settings,
            "dataset_path": context.parameters["dataset_path"],
            "model": context.parameters["model"],
            "eval_mode": self.config["eval_mode"],
            "template": context.parameters.get("template"),
        }


class ChoiceBenchmark(QualityBenchmark):
    default_input_template = "choice"
    default_answer_parser = "choice"
    record_fields = {"question": str, "choices": list, "answer": str}
    prediction_type = (str, type(None))
    parameters = {
        **FEW_SHOT,
        "max_samples": Parameter(int, 200, minimum=1),
        "output_tokens": Parameter(int, 8, minimum=1, matrix=True),
        "stop": Parameter(list, [], item_type=str),
        "eval_mode": Parameter(str, "gen", choices=("gen", "ppl")),
    }
    domain_module = choice
    capabilities = frozenset({"choice_answers", "correctness"})
    default_metrics = ("choice_accuracy",)

    @classmethod
    def domain_parameters(cls, parameters):
        if (parameters["eval_mode"] == "ppl"
                and parameters["endpoint"] != "completion"):
            raise ValueError(
                "eval_mode=ppl supports only endpoint='completion'"
            )
        module = cls.domain_module
        template = parameters["template"]
        definition = module.ChoiceBenchmarkDefinition(
            template, f"{template}_choice_accuracy", template
        )
        values = dict(parameters)
        values.update(
            service_url=parameters["base_url"],
            max_tokens=parameters["output_tokens"],
            prompt_format=cls._prompt_format(parameters),
        )
        return module.validate_choice_parameters(values, definition)

    def evaluate(self, sample, context):
        examples = select_few_shot(
            self.few_shot,
            "subject",
            sample.get("subject"),
            self.config["few_shot"],
        )
        return self.domain_module.collect_sample(
            self.client, sample, examples, self.config
        )


def choice_definition(name):
    """Bind a dataset template to one shared implementation."""
    implementation = type(
        f"{name.title()}Benchmark",
        (ChoiceBenchmark,),
        {"parameters": {"template": Parameter(str, name, choices=(name,))}},
    )
    return benchmarks.register(name)(implementation)


CEvalBenchmark = choice_definition("ceval")
CMMLUBenchmark = choice_definition("cmmlu")


class FewShotQualityBenchmark(QualityBenchmark):
    parameters = {
        **FEW_SHOT,
        "max_samples": Parameter(int, 200, minimum=1),
        "output_tokens": Parameter(int, 512, minimum=1, matrix=True),
    }

    def evaluate(self, sample, context):
        return self.domain_module.collect_sample(
            self.client,
            sample,
            self.few_shot[: self.config["few_shot"]],
            self.config,
        )


@benchmarks.register("gsm8k")
class GSM8KBenchmark(FewShotQualityBenchmark):
    default_input_template = "numeric_qa"
    default_answer_parser = "number"
    record_fields = {"question": str, "answer": str, "reference_number": float}
    prediction_type = (float, type(None))
    domain_module = arithmetic_qa
    capabilities = frozenset({"numeric_answers", "correctness"})
    default_metrics = ("numeric_accuracy",)
    parameters = {"template": Parameter(str, "gsm8k", choices=("gsm8k",))}


@benchmarks.register("lcsts")
class LCSTSBenchmark(FewShotQualityBenchmark):
    default_input_template = "summarization"
    default_answer_parser = "text"
    record_fields = {"content": str, "abst": str}
    domain_module = text_summarization
    capabilities = frozenset({"summary_answers"})
    default_metrics = ("chinese_rouge",)
    parameters = {"template": Parameter(str, "lcsts", choices=("lcsts",))}


@benchmarks.register("squad")
class SquadBenchmark(QualityBenchmark):
    default_input_template = "reading"
    default_answer_parser = "reading"
    record_fields = {"question": str, "context": str, "answers": list}
    domain_module = reading_comprehension
    capabilities = frozenset({"reading_answers"})
    default_metrics = ("reading_accuracy",)
    parameters = {
        "template": Parameter(str, "squad", choices=("squad",)),
        "data": Parameter(str, "squad", choices=("squad", "jsonl")),
        "output_tokens": Parameter(int, 64, minimum=1, matrix=True),
    }

    def prepare_records(self, records):
        return reading_comprehension.validate_records(records)


@benchmarks.register("wikitext")
class WikiTextBenchmark(QualityBenchmark):
    domain_module = language_modeling
    capabilities = frozenset({"token_logprobs"})
    default_metrics = ("language_model_loss",)
    parameters = {
        "template": Parameter(str, "wikitext", choices=("wikitext",)),
        "eval_mode": Parameter(str, "loss", choices=("loss",)),
        "endpoint": Parameter(
            str, "completion", choices=("completion",),
            help=HTTP["endpoint"].help,
        ),
        "max_context_length": Parameter(int, 2048, minimum=2, matrix=True),
        "stride": Parameter(int, None, minimum=1, matrix=True),
    }


@benchmarks.register("humaneval")
class HumanEvalBenchmark(QualityBenchmark):
    domain_module = code_generation
    capabilities = frozenset({"code_execution"})
    default_metrics = ("code_pass_rate",)
    resources = ("docker_sandbox",)
    parameters = {
        **SANDBOX,
        "template": Parameter(str, "humaneval", choices=("humaneval",)),
        "endpoint": Parameter(
            str, "completion", choices=("completion",),
            help=HTTP["endpoint"].help,
        ),
        "max_samples": Parameter(int, 164, minimum=1),
        "output_tokens": Parameter(int, 1024, minimum=1, matrix=True),
        "samples_per_task": Parameter(int, 1, choices=(1,)),
        "pass_k": Parameter(list, [1], item_type=int),
    }

    def prepare(self, context):
        super().prepare(context)
        module = self.domain_module
        self.sandbox = module.DockerSandboxExecutor(
            module._sandbox_config(self.config)
        )
        self.sandbox_environment = self.sandbox.preflight()
        self.semaphore = threading.BoundedSemaphore(
            self.config["sandbox_concurrency"]
        )

    def evaluate(self, sample, context):
        return self.domain_module.collect_sample(
            self.client, self.sandbox, self.semaphore, sample, self.config
        )

    def metadata(self, context):
        return {
            **super().metadata(context),
            "sandbox": self.sandbox_environment,
            "dataset_sha256": self.domain_module._file_sha256(
                context.dataset_path
            ),
        }
