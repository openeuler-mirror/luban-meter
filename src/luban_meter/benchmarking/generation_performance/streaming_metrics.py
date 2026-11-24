"""Individually selectable online statistics over formal request facts."""

from collections.abc import Mapping

from luban_meter.api import Metric, metrics
from luban_meter.core.token_totals import formal_throughput
from luban_meter.log import get_logger

from .common.statistics import scalar, summarize, summarize_weighted
from .online_serving import (
    guidellm_token_latencies,
    numeric,
    positive_numeric,
    positive_token_count,
    token_count,
)

logger = get_logger(__name__)
LATENCIES = ("ttft", "tpot", "itl", "e2el")
THROUGHPUTS = (
    "request_throughput", "input_token_throughput",
    "output_token_throughput", "total_token_throughput",
)


def latency_summary(name, responses):
    samples = []
    for response in responses:
        ttft = numeric(response, "ttft_ms")
        e2el = numeric(response, "e2el_ms")
        if e2el < ttft:
            raise ValueError(
                "request e2el_ms must not be smaller than ttft_ms"
            )
        if name in ("ttft", "e2el"):
            samples.append(ttft if name == "ttft" else e2el)
        elif response.get("output_tokens") is not None:
            tokens = positive_token_count(response, "output_tokens")
            tpot, itl = guidellm_token_latencies(response)
            if name == "tpot":
                samples.append((tpot, tokens))
            elif itl is not None:
                samples.append((itl, tokens - 1))
    if name in ("tpot", "itl"):
        return summarize_weighted(samples, "ms/token")
    return summarize(samples, "ms")


class StreamingStatistic(Metric):
    requires = frozenset({"client_stream_timing"})
    name = ""

    def compute(self, facts):
        responses = [
            row["response"] if isinstance(row.get("response"), Mapping)
            else {} for row in facts["requests"]
        ]
        if not responses:
            return {"status": "unavailable", "reason": "no admitted requests"}
        successful = [r for r in responses if r.get("status") == "success"]
        failed = len(responses) - len(successful)
        duration = positive_numeric(facts["window"], "duration_s")
        service = {
            "benchmark_duration": scalar(duration, "s", precision=6),
            "total_requests": scalar(len(responses), "request"),
            "successful_requests": scalar(len(successful), "request"),
            "failed_requests": scalar(failed, "request"),
        }
        values = {"service_view": service}
        if self.name in LATENCIES:
            node = latency_summary(self.name, successful)
            values["request_view"] = {self.name: node}
        elif self.name == "total_token_throughput":
            saved = formal_throughput(facts)
            service.update(saved["values"]["service_view"])
            node = service[self.name]
        elif self.name == "request_throughput":
            node = scalar(len(successful) / duration, "req/s")
            service[self.name] = node
        else:
            field = ("input_tokens" if self.name == "input_token_throughput"
                     else "output_tokens")
            counter = (token_count if field == "input_tokens"
                       else positive_token_count)
            counts = [counter(row, field) for row in successful
                      if row.get(field) is not None]
            missing = len(counts) != len(successful)
            node = scalar(None if missing else sum(counts) / duration,
                          "token/s")
            total = scalar(None if missing else sum(counts), "token")
            if missing:
                node["reason"] = total["reason"] = "missing usage"
            service[self.name] = node
            service["total_" + field] = total
        logger.info(
            "Online metric samples: metric=%s successful=%d failed=%d "
            "eligible=%d", self.name, len(successful), failed,
            node.get("count", len(successful)
                     if node.get("value") is not None else 0),
        )
        return {
            "status": ("partial_failed" if successful and failed
                       else "failed" if failed else "success"),
            "observation_boundary": "client",
            "values": values,
        }


for _name in (*LATENCIES, *THROUGHPUTS):
    metrics.register(_name)(type(
        _name.title().replace("_", ""), (StreamingStatistic,),
        {"name": _name, "description": "Online " + _name,
         "aggregate_run_throughput": _name == "total_token_throughput"},
    ))
