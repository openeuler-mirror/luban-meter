"""Generation scenarios share admission and preserve timing views."""

from __future__ import annotations

import itertools
import math

from luban_meter.api import (
    GenerationPerformanceBenchmark,
    Parameter,
    benchmarks,
)
from luban_meter.benchmarking.benchmark_parameters import DATA, HTTP
from luban_meter.core.template_contracts import (
    compatible,
    input_templates,
    validate_fields,
)
from luban_meter.data import DatasetLoader
from luban_meter.data.sources import sources
from luban_meter.runtime.http_client import create_http_client, http_settings
from luban_meter.runtime.request_params import (
    append_request_params,
    validate_request_params,
)

from .common.lengths import assess_lengths
from .common.tokenization import (
    LENGTH_PARAMETERS,
    LengthSampler,
    TextGenerator,
    check_context,
    derived_seed,
    generated_summary,
    load_tokenizer,
    validate_lengths,
)
from .common.trace_inputs import (
    TRACE_PARAMETERS,
    prepare_trace,
    validate_trace_parameters,
)


def thresholds(entries, allowed):
    result = {}
    for entry in entries:
        name, separator, value = entry.partition(":")
        if not separator or name not in allowed:
            raise ValueError(f"invalid goodput threshold: {entry}")
        threshold = float(value)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"threshold must be positive: {entry}")
        if name in result:
            raise ValueError(f"duplicate threshold: {name}")
        result[name] = threshold
    return result


@benchmarks.register("online_serving")
class OnlineServingBenchmark(GenerationPerformanceBenchmark):
    profile_modes = frozenset({"requests", "trace"})
    capabilities = frozenset({"client_stream_timing"})
    default_metrics = ("streaming_performance",)
    parameters = {
        **HTTP,
        **DATA,
        **TRACE_PARAMETERS,
        "data": Parameter(str, "random_token"),
        "dataset_path": Parameter(str, "", path=True),
        "profile": Parameter(str, "rate"),
        **LENGTH_PARAMETERS,
        "num_requests": Parameter(int, 100, minimum=1),
        "tokenizer": Parameter(str, ""),
        "prompt_field": Parameter(str, "prompt"),
        "input_template": Parameter(str, "prompt_record"),
        "input_format": Parameter(str, ""),
        "field_mapping": Parameter(list, [], item_type=str),
        "repeat_data": Parameter(bool, False),
        "stop": Parameter(list, [], item_type=str),
        "goodput": Parameter(list, [], item_type=str),
        "circuit_breaker": Parameter(float, None, minimum=0.001),
        "max_error_rate": Parameter(float, None, minimum=0.0, maximum=1.0),
    }

    @classmethod
    def validate(cls, parameters):
        http_settings(parameters)
        validate_request_params(parameters["request_params"])
        validate_trace_parameters(parameters)
        thresholds(parameters["goodput"], ("ttft", "tpot", "e2el"))
        if parameters["profile"] == "replay":
            if parameters["data"] != "burstgpt" or parameters["shuffle"]:
                raise ValueError("replay requires unshuffled burstgpt input")
        if parameters["data"] == "burstgpt" and parameters["repeat_data"]:
            raise ValueError("trace records cannot be repeated")
        compatible(
            input_templates.get(parameters["input_template"]).provides,
            {"prompt": str},
        )
        if parameters["skip_invalid"] and parameters["data"] != "burstgpt":
            raise ValueError("skip_invalid requires burstgpt")
        synthetic = parameters["data"] == "random_token"
        validate_lengths(parameters, synthetic=synthetic)
        if not synthetic:
            sources.get(parameters["data"])
            if not parameters["dataset_path"]:
                raise ValueError("local input requires dataset_path")

    def prepare(self, context):
        self.p = context.parameters
        self.records = None
        self.tokenizer = None
        self.generator = None
        self.trace_metadata = {}
        self.input_template = input_templates.get(self.p["input_template"])()
        if self.p["data"] in ("random_token", "burstgpt"):
            self.tokenizer = load_tokenizer(self.p)
            self.generator = TextGenerator(self.tokenizer, self.p)
            if self.p["data"] == "burstgpt":
                self.generated, self.trace_metadata = prepare_trace(
                    context, self.generator
                )
            else:
                self.generated = self.generator.prepare(context)
            if self.p["max_model_len"]:
                for rows in self.generated.values():
                    for row in rows:
                        self.check_input(row["prompt"], row)
            context.total_samples = self.p["num_requests"]
        else:
            self.records = DatasetLoader(
                self.p["dataset_path"],
                format=self.p["data"],
                max_samples=self.p.get("max_samples"),
                shuffle=self.p["shuffle"],
                seed=(
                    self.p["data_seed"]
                    if self.p["data_seed"] is not None
                    else self.p["seed"]
                ),
            ).load()
            if not self.records:
                raise ValueError("dataset contains no samples")
            for record in self.records:
                self.render_record(record)
            if self.p["max_model_len"]:
                self.tokenizer = load_tokenizer(self.p)
            context.total_samples = (
                self.p["num_requests"]
                if self.p["repeat_data"]
                else min(len(self.records), self.p["num_requests"])
            )
        from .online_serving import ActiveRequestTracker

        self.tracker = ActiveRequestTracker()
        self.http_settings = http_settings(self.p)
        self.client = create_http_client(
            self.p["request_timeout"],
            max_connections=self.http_settings["max_connections"],
            keepalive_expiry=self.http_settings["keepalive_expiry"],
        )

    def close(self):
        if getattr(self, "client", None) is not None:
            self.client.close()

    def samples(self, context):
        phase = context.phase
        if self.records is None:
            yield from self.generated[phase]
            return
        records = (
            itertools.cycle(self.records)
            if self.p["repeat_data"]
            else iter(self.records)
        )
        seed = self.p["data_seed"]
        seed = self.p["seed"] if seed is None else seed
        sampler = LengthSampler.from_parameters(self.p, "output")
        for index, record in enumerate(
            itertools.islice(records, self.p["num_requests"])
        ):
            yield {
                "record": record,
                "target_output_tokens": sampler.sample(
                    derived_seed(seed, phase, index, "output_length")
                ),
            }

    def render_record(self, record):
        validate_fields(record, self.input_template.requires, "input record")
        rendered = self.input_template.render(record, [], self.p)
        validate_fields(
            rendered, self.input_template.provides, "template output"
        )
        if not rendered["prompt"].strip():
            raise ValueError("sample has no non-empty prompt field")
        return rendered["prompt"]

    def check_input(self, text, sample):
        limit = self.p["max_model_len"]
        if not limit:
            return
        if self.p["endpoint"] == "/v1/chat/completions":
            tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        else:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        check_context(len(tokens), sample["target_output_tokens"], limit)

    def evaluate(self, sample, context):
        from .common.streaming import (
            collect_chat_stream,
            collect_completion_stream,
        )
        from .online_serving import execute_request

        if self.records is None:
            text = sample["prompt"]
        else:
            text = self.render_record(sample["record"])
        budget = sample["target_output_tokens"]
        payload = {
            "model": self.p["model"],
            "max_tokens": budget,
            "temperature": self.p["temperature"],
            "seed": self.p["seed"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "stop": self.p["stop"],
        }
        if self.p["ignore_eos"]:
            payload.update(min_tokens=budget, ignore_eos=True)
        if self.p["endpoint"] == "/v1/chat/completions":
            payload["messages"] = [{"role": "user", "content": text}]
        else:
            payload.update(prompt=text, add_special_tokens=False)
        payload = append_request_params(payload, self.p["request_params"])
        if self.records is not None:
            self.check_input(text, sample)

        def send(ticket):
            record = execute_request(
                client=self.client,
                request_index=ticket.index,
                payload=payload,
                endpoint=self.p["endpoint"],
                service_url=self.p["base_url"],
                api_key=self.p["api_key"],
                timeout=self.p["request_timeout"],
                benchmark_start=context.scheduler.started,
                scheduled_time=ticket.scheduled,
                tracker=self.tracker,
                stream_collector=(
                    collect_chat_stream
                    if self.p["endpoint"] == "/v1/chat/completions"
                    else collect_completion_stream
                ),
            )
            record["lengths"] = assess_lengths(
                sample,
                record.get("input_tokens"),
                record.get("output_tokens"),
                self.p,
            )
            return record

        return context.request(send)

    def stop_after(self, facts):
        from .common.circuit_breaker import (
            case_p99_e2el_ms,
            check_stop_constraints,
        )

        case = {"requests": [row["facts"] for row in facts["samples"]]}
        p99 = case_p99_e2el_ms(case)
        threshold = self.p["circuit_breaker"]
        if threshold is not None and p99 is not None and p99 > threshold:
            return "circuit_breaker_exceeded"
        return check_stop_constraints(
            case, max_duration=None, max_error_rate=self.p["max_error_rate"]
        )

    def metadata(self, context):
        return {
            "http_client": self.http_settings,
            **(
                generated_summary(self.generated["formal"])
                if hasattr(self, "generated")
                else {}
            ),
            **self.trace_metadata,
            "model": self.p["model"],
            "slo_config": {
                f"{name}_ms": value
                for name, value in thresholds(
                    self.p["goodput"], ("ttft", "tpot", "e2el")
                ).items()
            }
            or None,
        }


@benchmarks.register("offline_vllm_engine")
class OfflineEngineBenchmark(GenerationPerformanceBenchmark):
    profile_modes = frozenset({"batch"})
    capabilities = frozenset({"engine_timing"})
    default_metrics = ("engine_performance",)
    parameters = {
        "profile": Parameter(str, "batch"),
        "backend": Parameter(str, "vllm_python", choices=("vllm_python",)),
        "model": Parameter(str, "", required=True),
        **DATA,
        **TRACE_PARAMETERS,
        "data": Parameter(
            str, "random_token", choices=("random_token", "burstgpt")
        ),
        "dataset_path": Parameter(str, "", path=True),
        **LENGTH_PARAMETERS,
        "num_requests": Parameter(int, 100, minimum=1),
        "temperature": Parameter(float, 0.0, choices=(0.0,)),
        "tensor_parallel_size": Parameter(int, 1, minimum=1),
        "gpu_memory_utilization": Parameter(
            float, 0.9, minimum=0.01, maximum=1
        ),
        "dtype": Parameter(str, "auto"),
        "trust_remote_code": Parameter(bool, False),
        "enforce_eager": Parameter(bool, False),
        "max_model_len": Parameter(int, None, minimum=1),
        "goodput": Parameter(list, [], item_type=str),
    }

    @classmethod
    def validate(cls, parameters):
        from .vllm_engine import ENGINE_SLO_DIMENSIONS

        validate_trace_parameters(parameters)
        thresholds(parameters["goodput"], ENGINE_SLO_DIMENSIONS)
        if parameters["data"] == "random_token":
            validate_lengths(parameters)
        if parameters["data"] == "burstgpt" and not parameters["dataset_path"]:
            raise ValueError("burstgpt requires dataset_path")

    def prepare(self, context):
        from . import vllm_engine as sampling

        self.p = context.parameters
        self.engine = sampling.build_engine(
            {"model_name": self.p["model"]}, self.p
        )
        self.limit = sampling.engine_max_model_len(self.engine)
        if self.p["data"] == "random_token":
            validate_lengths(self.p, limit=self.limit)
        self.generator = TextGenerator(self.engine.get_tokenizer(), self.p)
        self.trace_metadata = {}
        if self.p["data"] == "burstgpt":
            self.generated, self.trace_metadata = prepare_trace(
                context, self.generator, self.p["batch_size"]
            )
        else:
            self.generated = self.generator.prepare(
                context, self.p["batch_size"]
            )
        for rows in self.generated.values():
            for row in rows:
                check_context(
                    row["local_input_tokens"],
                    row["target_output_tokens"],
                    self.limit,
                )
        self.environment = {
            "kv_cache": sampling.kv_cache_environment(self.engine)
        }
        context.total_samples = self.p["num_requests"]

    def samples(self, context):
        inputs = self.generated[context.phase]
        size = self.p["batch_size"]
        for offset in range(0, len(inputs), size):
            rows = inputs[offset : offset + size]
            yield {"offset": offset, "size": len(rows), "inputs": rows}

    def evaluate(self, sample, context):
        from vllm import SamplingParams

        from .vllm_engine import generate_batch

        prompts = []
        settings = []
        for row in sample["inputs"]:
            # Send the exact tokenization of the generated text, without
            # Engine-specific automatic special-token insertion.
            tokens = self.engine.get_tokenizer().encode(
                row["prompt"],
                add_special_tokens=False,
            )
            budget = row["target_output_tokens"]
            check_context(len(tokens), budget, self.limit)
            prompts.append({"prompt_token_ids": tokens})
            settings.append(
                SamplingParams(
                    max_tokens=budget,
                    min_tokens=budget if self.p["ignore_eos"] else 0,
                    ignore_eos=self.p["ignore_eos"],
                    temperature=self.p["temperature"],
                    detokenize=False,
                    seed=self.p["seed"],
                )
            )
        records = context.request(
            lambda ticket: generate_batch(self.engine, settings, prompts),
            cost=sample["size"],
        )
        for row, record in zip(sample["inputs"], records, strict=True):
            record.update(request_batch_size=sample["size"])
            record["lengths"] = assess_lengths(
                row,
                record.get("actual_prompt_tokens"),
                record.get("actual_output_tokens"),
                self.p,
            )
        return {"status": "success", "requests": records, **sample}

    def metadata(self, context):
        from .vllm_engine import ENGINE_SLO_DIMENSIONS

        return {
            **(
                generated_summary(self.generated["formal"])
                if hasattr(self, "generated")
                else {}
            ),
            **self.trace_metadata,
            "model": self.p["model"],
            "environment": self.environment,
            "engine_slo_config": thresholds(
                self.p["goodput"], ENGINE_SLO_DIMENSIONS
            )
            or None,
        }


@benchmarks.register("offline_vllm_async")
class AsyncEngineBenchmark(OfflineEngineBenchmark):
    profile_modes = frozenset({"requests", "trace"})
    capabilities = frozenset({"async_engine_timing"})
    default_metrics = ("async_engine_performance",)
    parameters = {
        "profile": Parameter(str, "concurrent"),
        "backend": Parameter(
            str, "vllm_python_async", choices=("vllm_python_async",)
        ),
    }

    @classmethod
    def validate(cls, parameters):
        super().validate(parameters)
        if parameters["goodput"]:
            raise ValueError("async Engine does not expose batch goodput")
        if parameters["profile"] == "replay" and (
            parameters["data"] != "burstgpt" or parameters["shuffle"]
        ):
            raise ValueError("replay requires unshuffled burstgpt input")

    def prepare(self, context):
        from .async_engine import AsyncEngineBackend

        self.p = context.parameters
        self.engine = AsyncEngineBackend(self.p)
        self.limit = self.engine.max_model_len
        self.generator = TextGenerator(self.engine.tokenizer, self.p)
        self.trace_metadata = {}
        if self.p["data"] == "burstgpt":
            self.generated, self.trace_metadata = prepare_trace(
                context, self.generator
            )
        else:
            self.generated = self.generator.prepare(context)
        for rows in self.generated.values():
            for row in rows:
                check_context(
                    row["local_input_tokens"],
                    row["target_output_tokens"],
                    self.limit,
                )
        context.total_samples = len(self.generated["formal"])

    def samples(self, context):
        return iter(self.generated[context.phase])

    def evaluate(self, sample, context):
        from vllm import SamplingParams

        budget = sample["target_output_tokens"]
        tokens = self.engine.tokenizer.encode(
            sample["prompt"], add_special_tokens=False
        )
        settings = SamplingParams(
            max_tokens=budget,
            min_tokens=budget if self.p["ignore_eos"] else 0,
            ignore_eos=self.p["ignore_eos"],
            temperature=0,
            seed=self.p["seed"],
            detokenize=False,
        )

        def submit(ticket):
            record = self.engine.generate(
                {"prompt_token_ids": tokens},
                settings,
                self.p["request_timeout"],
            )
            record["lengths"] = assess_lengths(
                sample, record["input_tokens"], record["output_tokens"], self.p
            )
            return record

        return context.request(submit)

    def metadata(self, context):
        return {
            **(
                generated_summary(self.generated["formal"])
                if hasattr(self, "generated")
                else {}
            ),
            **self.trace_metadata,
            "model": self.p["model"],
            "observation_boundary": "engine_submission",
        }

    def close(self):
        if hasattr(self, "engine"):
            self.engine.close()
