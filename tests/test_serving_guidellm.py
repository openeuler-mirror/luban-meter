"""Fixed-timeline checks for GuideLLM token metrics and script
execution.
"""

import importlib.util
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from luban_meter.benchmarking.generation_performance.common.statistics import (
    summarize_weighted,
)
from luban_meter.benchmarking.generation_performance.common.streaming import (
    StreamObservation,
)

ROOT = Path(__file__).parents[1]
SERVING = (
    ROOT / "src/luban_meter/benchmarking/generation_performance"
    / "online_serving.py"
)


def load_module(filename):
    spec = importlib.util.spec_from_file_location(
        f"guidellm_test_{filename}", SERVING
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def processor():
    return load_module("metrics")


def record(*, tokens=2, first=900, last=1000, end=1500):
    return {
        "status": "success",
        "ttft_ms": first,
        "last_output_latency_ms": last,
        "e2el_ms": end,
        "duration_ms": end,
        "start_offset_ms": 0,
        "dispatch_delay_ms": 0,
        "itl_samples_ms": [last - first] if last is not None else [],
        "input_tokens": 4,
        "output_tokens": tokens,
    }


def raw_result(records, *, slo=None):
    metadata = {}
    if slo is not None:
        metadata["slo_config"] = slo
    return {
        "metadata": metadata,
        "metrics": {
            "cases": [
                {
                    "request_rate": 1,
                    "benchmark_duration_seconds": 10,
                    "maximum_request_concurrency": 2,
                    "peak_concurrent_requests": 2,
                    "requests": records,
                }
            ]
        },
    }


def case_result(processor, records, **kwargs):
    return processor.process(raw_result(records, **kwargs))["metrics"][
        "cases"
    ][0]


def test_tail_events_do_not_inflate_token_metrics(processor):
    case = case_result(processor, [record()], slo={"tpot_ms": 200})
    view = case["request_view"]
    assert view["tpot"]["mean"] == 500
    assert view["itl"]["mean"] == 100
    assert view["e2el"]["mean"] == 1500
    goodput = case["service_view"]["goodput"]
    assert goodput["slo_satisfied_count"]["value"] == 1
    assert goodput["tpot_metric"] == "inter_token_latency"


def test_variable_lengths_use_token_weighted_cdf(processor):
    case = case_result(
        processor,
        [
            record(tokens=2, first=100, last=200, end=300),
            record(tokens=8, first=10, last=80, end=200),
        ],
    )
    tpot = case["request_view"]["tpot"]
    assert tpot["mean"] == 28
    assert tpot["stddev"] == 36
    assert tpot["p50"] == 10
    assert tpot["p90"] == 100
    assert tpot["count"] == 2
    assert tpot["weight_sum"] == 10
    itl = case["request_view"]["itl"]
    assert itl["weight_sum"] == 8
    assert itl["mean"] == 21.25
    assert itl["p50"] == 10
    assert itl["p90"] == 100
    assert itl["p99"] == 100
    # Two SSE gaps remain available, independently of the Token count.
    assert case["request_view"]["stream_event_itl"]["count"] == 2


def test_weighted_quantiles_do_not_interpolate():
    summary = summarize_weighted([(10, 1), (30, 1)], "ms/token")
    assert summary["p50"] == 10
    assert summary["p90"] == 30
    assert summary["mean"] == 20
    assert summarize_weighted([], "ms/token")["mean"] is None


@pytest.mark.parametrize(
    "sample",
    [
        (float("nan"), 1),
        (float("inf"), 1),
        (-1, 1),
        (1, 0),
        (1, -1),
        (1, True),
    ],
)
def test_invalid_weighted_samples_rejected(sample):
    with pytest.raises(ValueError):
        summarize_weighted([sample], "ms/token")


def test_single_token_is_measured_but_slo_is_undetermined(processor):
    case = case_result(
        processor,
        [record(tokens=1, first=1000, last=1000)],
        slo={"tpot_ms": 50, "ttft_ms": 10},
    )
    assert case["request_view"]["tpot"]["mean"] == 1000
    assert case["request_view"]["itl"]["count"] == 0
    goodput = case["service_view"]["goodput"]
    assert goodput["status"] == "not_applicable"
    assert goodput["undetermined_count"]["value"] == 1
    assert goodput["slo_violated_count"]["value"] == 0
    assert goodput["slo_satisfied_rate"]["value"] is None
    assert goodput["goodput_request_throughput"]["value"] is None


def test_single_token_can_satisfy_ttft_only_objective(processor):
    case = case_result(
        processor,
        [record(tokens=1, first=1000, last=1000)],
        slo={"ttft_ms": 1000},
    )
    assert case["service_view"]["goodput"]["slo_satisfied_count"]["value"] == 1


def test_failures_lower_attainment_and_unknowns_are_excluded(processor):
    failed = record()
    failed["status"] = "failed"
    case = case_result(
        processor,
        [
            record(),
            failed,
            record(tokens=1, first=1000, last=1000),
        ],
        slo={"tpot_ms": 100},
    )
    goodput = case["service_view"]["goodput"]
    assert goodput["determined_count"]["value"] == 2
    assert goodput["undetermined_count"]["value"] == 1
    assert goodput["failed_count"]["value"] == 1
    assert goodput["slo_satisfied_rate"]["value"] == 0.5
    assert goodput["goodput_request_throughput"]["value"] == 0.1


def test_all_failures_have_zero_goodput(processor):
    failed = record()
    failed["status"] = "failed"
    goodput = case_result(processor, [failed], slo={"tpot_ms": 100})[
        "service_view"
    ]["goodput"]
    assert goodput["status"] == "applicable"
    assert goodput["slo_satisfied_rate"]["value"] == 0


def test_missing_last_event_falls_back_only_for_report_tpot(processor):
    case = case_result(processor, [record(last=None)], slo={"tpot_ms": 100})
    assert case["request_view"]["tpot"]["mean"] == 750
    assert case["request_view"]["itl"]["mean"] is None
    assert case["service_view"]["goodput"]["undetermined_count"]["value"] == 1


@pytest.mark.parametrize("last", [800, 1600, float("nan")])
def test_invalid_event_timeline_is_rejected(processor, last):
    with pytest.raises(ValueError):
        case_result(processor, [record(last=last)])


def test_raw_uses_guidellm_without_semantics_marker(processor):
    raw = raw_result([record()], slo={"tpot_ms": 200})
    result = processor.process(raw)
    assert "token_metrics_semantics" not in raw["metadata"]
    assert "token_metrics_semantics" not in result["metadata"]
    case = result["metrics"]["cases"][0]
    assert case["request_view"]["tpot"]["mean"] == 500
    assert case["request_view"]["itl"]["mean"] == 100
    assert case["service_view"]["goodput"]["slo_satisfied_count"]["value"] == 1


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_invalid_slo_in_raw_result_is_rejected(processor, threshold):
    with pytest.raises(ValueError):
        case_result(processor, [record()], slo={"tpot_ms": threshold})


def test_collector_retains_last_content_time():
    benchmark = load_module("sampling")
    collected = benchmark._build_request_record(
        request_index=0,
        started=10,
        ended=11.5,
        benchmark_start=10,
        scheduled_time=10,
        observation=StreamObservation("ab", (10.9, 11), 4, 2),
        error=None,
    )
    assert collected["last_output_latency_ms"] == 1000
    assert collected["ttft_ms"] == 900
    assert collected["e2el_ms"] == 1500


class SyntheticServer(BaseHTTPRequestHandler):
    """A local synthetic protocol fixture, not a model performance
    test.
    """

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(
            self.rfile.read(int(self.headers["Content-Length"]))
        )
        self.send_response(200)
        self.end_headers()
        if self.path == "/tokenize":
            self.wfile.write(
                json.dumps(
                    {
                        "tokens": [11, 12],
                        "count": 2,
                        "max_model_len": 1024,
                    }
                ).encode()
            )
            return
        chat = self.path == "/v1/chat/completions"
        assert payload["stream"] is True
        for text in ("a", "bc"):
            choice = {"delta": {"content": text}} if chat else {"text": text}
            self.wfile.write(
                (
                    "data: " + json.dumps({"choices": [choice]}) + "\n\n"
                ).encode()
            )
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": 4 if chat else len(payload["prompt"]),
                "completion_tokens": 3,
            },
        }
        self.wfile.write(
            ("data: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode()
        )
