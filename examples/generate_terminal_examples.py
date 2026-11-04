"""Export terminal acceptance examples without calling a model service."""

import copy
import json
from pathlib import Path

from luban_meter.benchmarking.generation_performance import online_serving
from luban_meter.log import get_logger, logged_script
from luban_meter.reporting.data import load_result
from luban_meter.reporting.terminal import render_text

logger = get_logger("luban_meter.examples.generate_terminal_examples")


@logged_script
def main():
    repo = Path(__file__).resolve().parents[1]
    output = repo / "results/terminal-report-v1"
    output.mkdir(parents=True, exist_ok=True)
    success = load_result(repo / "examples/terminal-result.json")
    failure = copy.deepcopy(success)
    failure["id"] = "synthetic-partial-failure"
    failure["status"] = failure["tasks"][0]["status"] = "partial_failed"
    case = failure["tasks"][0]["cases"][0]
    case["status"] = "partial_failed"
    records = [{
        "status": "success", "input_tokens": 64, "output_tokens": 16,
        "ttft_ms": 20, "last_output_latency_ms": 320, "e2el_ms": 320,
        "duration_ms": 320, "start_offset_ms": i * 330,
        "scheduled_latency_ms": 320, "dispatch_delay_ms": 0,
        "itl_samples_ms": [20] * 15,
    } for i in range(48)]
    records.extend({"status": "failed", "duration_ms": 100,
                    "dispatch_delay_ms": 0,
                    "start_offset_ms": i * 330} for i in (48, 49))
    values, complete, failed = online_serving.process_case({
        "benchmark_duration_seconds": 17,
        "maximum_request_concurrency": 1,
        "peak_concurrent_requests": 1, "requests": records,
    })
    case["coverage"].update(complete=complete, failed=failed,
                            completion_ratio=complete / 50)
    case["metrics"]["streaming_performance"].update(
        status="partial_failed", values=values,
        error={"type": "RequestFailures", "message": "2 simulated timeouts"},
    )
    quality = load_result(repo / "examples/terminal-gsm8k-result.json")
    mixed = copy.deepcopy(success)
    mixed["id"] = "synthetic-mixed-with-historical-gsm8k"
    mixed["status"] = "partial_failed"
    mixed["tasks"][0]["status"] = "partial_failed"
    mixed["tasks"][0]["cases"].append(dict(case, id="case_2"))
    mixed["tasks"].append(dict(quality["tasks"][0], name="quality"))
    for name, payload in (("success", success), ("failure", failure),
                          ("quality", quality), ("multi-case", mixed)):
        path = output / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)
                        + "\n", encoding="utf-8")
        for suffix, verbose, width in (("default", False, 80),
                                       ("verbose", True, 80),
                                       ("narrow", False, 40)):
            text = render_text(payload, verbose=verbose, width=width,
                               result_path=path, all_cases=True)
            (output / f"{name}-{suffix}.txt").write_text(
                text + "\n", encoding="utf-8",
            )
    logger.info("Examples: %s", output)
    logger.info(
        "success/failure: synthetic; quality: historical real GSM8K result"
    )
    logger.info("multi-case: synthetic composition, not a new benchmark run")


if __name__ == "__main__":
    main()
