"""Registered metrics reuse domain formulas, not old file protocols."""

from __future__ import annotations

import math

from luban_meter.api import Metric, metrics

from .model_service_quality import (
    arithmetic_qa,
    code_generation,
    language_modeling,
    reading_comprehension,
    text_summarization,
)


def completed_samples(facts):
    return [
        sample["facts"]
        for sample in facts["samples"]
        if sample["state"] not in ("incomplete", "not_started")
    ]


class DomainMetric(Metric):
    processor = None

    def compute(self, facts):
        samples = completed_samples(facts)
        if not samples:
            return {"status": "unavailable", "reason": "no complete samples"}
        processed = self.processor(
            {
                "status": "success",
                "metrics": {"samples": samples},
                "metadata": facts["metadata"],
            }
        )
        return {
            "status": processed.get("status", "success"),
            "values": processed["metrics"],
        }


def domain_metric(name, implementation, capability):
    metric = type(
        name.title().replace("_", ""),
        (DomainMetric,),
        {
            "processor": staticmethod(implementation.process),
            "requires": frozenset({capability}),
        },
    )
    metrics.register(name)(metric)


domain_metric("numeric_accuracy", arithmetic_qa, "numeric_answers")
domain_metric("chinese_rouge", text_summarization, "summary_answers")
domain_metric("reading_accuracy", reading_comprehension, "reading_answers")
domain_metric("language_model_loss", language_modeling, "token_logprobs")
domain_metric("code_pass_rate", code_generation, "code_execution")


@metrics.register("choice_accuracy")
class ChoiceAccuracy(Metric):
    requires = frozenset({"choice_answers"})

    def compute(self, facts):
        from .model_service_quality.common.choice_result import (
            process_choice_result,
        )

        samples = completed_samples(facts)
        if not samples:
            return {"status": "unavailable", "reason": "no complete samples"}
        result = process_choice_result(
            {"status": "success", "metrics": {"samples": samples}},
            facts["metadata"]["template"],
        )
        return {"status": result["status"], "values": result["metrics"]}


@metrics.register("sample_accuracy")
class SampleAccuracy(Metric):
    """A reusable optional metric, independent of a dataset class."""

    requires = frozenset({"correctness"})

    def compute(self, facts):
        samples = completed_samples(facts)
        correct = sum(bool(sample.get("correct")) for sample in samples)
        return {
            "accuracy": {
                "value": correct / len(samples) if samples else None,
                "unit": "ratio",
                "sample_count": len(samples),
                "condition": (
                    "complete samples; failed samples count as incorrect"
                ),
            }
        }


@metrics.register("streaming_performance")
class StreamingPerformance(Metric):
    requires = frozenset({"client_stream_timing"})

    def compute(self, facts):
        from .generation_performance.common.lengths import length_summary
        from .generation_performance.online_serving import process_case

        samples = completed_samples(facts)
        if not samples:
            return {"status": "unavailable", "reason": "no admitted requests"}
        p = facts["parameters"]
        rate = p.get("request_rate")
        if rate is not None and not math.isfinite(float(rate)):
            rate = None
        case = {
            "request_rate": rate,
            "benchmark_duration_seconds": facts["window"]["duration_s"],
            "maximum_request_concurrency": facts["window"]["request_capacity"],
            "peak_concurrent_requests": facts["window"]["peak_requests"],
            "requests": samples,
        }
        values, success, failed = process_case(
            case, facts["metadata"].get("slo_config")
        )
        return {
            "status": "partial_failed"
            if failed and success
            else ("failed" if failed else "success"),
            "values": values,
            "observation_boundary": "client",
            "lengths": length_summary(samples),
        }


@metrics.register("engine_performance")
class EnginePerformance(Metric):
    requires = frozenset({"engine_timing"})

    def compute(self, facts):
        from .generation_performance.common.lengths import length_summary
        from .generation_performance.vllm_engine import process

        samples = completed_samples(facts)
        if not samples:
            return {"status": "unavailable", "reason": "no complete batches"}
        # Group the final smaller batch by its actual size.
        groups = {}
        for index, sample in enumerate(samples):
            size = sample["size"]
            groups.setdefault(size, []).append(
                {
                    "round_index": index,
                    "requests": sample["requests"],
                }
            )
        result = process(
            {
                "metadata": facts["metadata"],
                "environment": facts["metadata"]["environment"],
                "metrics": {
                    "cases": [
                        {
                            "request_batch_size": size,
                            "rounds": rounds,
                        }
                        for size, rounds in groups.items()
                    ]
                },
            }
        )
        return {
            "status": "success",
            "values": result["metrics"],
            "environment": result["environment"],
            "observation_boundary": "engine",
            "lengths": length_summary(
                [row for sample in samples for row in sample["requests"]]
            ),
        }


@metrics.register("async_engine_performance")
class AsyncEnginePerformance(Metric):
    requires = frozenset({"async_engine_timing"})

    def compute(self, facts):
        from .generation_performance.common.lengths import length_summary
        from .generation_performance.common.statistics import summarize

        rows = [
            row["facts"] for row in facts["samples"]
            if row["state"] == "complete"
        ]
        duration = facts["window"]["duration_s"]
        return {
            "observation_boundary": "engine_submission",
            "condition": "completed requests; actual collection window",
            "e2el": summarize(
                [row["submission_e2el_s"] * 1000 for row in rows], "ms"
            ),
            "ttft": summarize([
                row["submission_ttft_s"] * 1000 for row in rows
                if row["submission_ttft_s"] is not None
            ], "ms"),
            "output_throughput": {
                "value": sum(r["output_tokens"] for r in rows) / duration
                if duration > 0 else None,
                "unit": "token/s", "sample_count": len(rows),
            },
            "engine_internal_timing": {
                "available_requests": sum(
                    row["engine_timestamps_available"] for row in rows
                ),
                "condition": "raw Engine timestamps only; no substitution",
            },
            "lengths": length_summary(rows),
        }
