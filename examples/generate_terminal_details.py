"""Create synthetic request-detail fixtures; never call a model service."""

import copy
import json
from pathlib import Path

from luban_meter.log import get_logger, logged_script
from luban_meter.reporting.terminal_details import render_details

logger = get_logger("luban_meter.examples.generate_terminal_details")


def write_run(root, name, raw):
    directory = root / name
    path = directory / "tasks/task_1/raw/case_1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = sum(row["status"] == "failed" for row in raw["requests"])
    status = "partial_failed" if failed else "success"
    payload = {
        "kind": "run", "id": "synthetic-details-" + name, "status": status,
        "tasks": [{
            "name": "task_1", "category": "generation_performance",
            "status": status, "cases": [{
                "id": "case_1", "status": status, "metrics": {},
                "raw": "tasks/task_1/raw/case_1.json",
                "coverage": {"selected": len(raw["samples"]),
                             "complete": len(raw["samples"]) - failed,
                             "failed": failed, "incomplete": 0},
            }],
        }],
    }
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    (directory / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload, directory


@logged_script
def main():
    root = Path(__file__).resolve().parents[1] / "results/terminal-report-v2"
    raw = {"requests": [], "samples": []}
    for index in range(14):
        failed = index >= 12
        request = {
            "request_id": f"request-{index}", "sample_id": str(index),
            "status": "failed" if failed else "success",
            "start_offset_s": index * 0.7, "end_offset_s": index * 0.7 + 0.5,
            "response": {"duration_ms": 500} if failed else {
                "input_tokens": 64, "output_tokens": 16,
                "ttft_ms": 20 + index, "e2el_ms": 320 + index * 8,
                "last_output_latency_ms": 318 + index * 8,
            },
        }
        sample = {
            "sample_id": str(index),
            "state": "failed" if failed else "complete",
            "source": {"prompt": f"Synthetic prompt {index}"},
        }
        if failed:
            request["error"] = "simulated request timeout"
            sample["error"] = {"type": "TimeoutError",
                               "message": "simulated request timeout"}
        raw["requests"].append(request)
        raw["samples"].append(sample)
    payload, directory = write_run(root, "partial", raw)
    options = [
        ("errors-80", {"details": "errors", "width": 80}),
        ("slow-top10-80", {"sort": "e2el", "limit": 10, "width": 80}),
        ("requests-100", {"width": 100}),
        ("requests-120", {"width": 120}),
        ("requests-40", {"width": 40, "limit": 2}),
        ("body-80", {"width": 80, "limit": 1, "include_body": True}),
        ("verbose-80", {"width": 80, "limit": 1, "verbose": True}),
    ]
    for name, kwargs in options:
        text = render_details(payload, raw_root=directory,
                              result_path=directory / "result.json", **kwargs)
        (root / (name + ".txt")).write_text(text + "\n", encoding="utf-8")
    missing = copy.deepcopy(payload)
    case = missing["tasks"][0]["cases"][0]
    case["id"] = "case_missing"
    case["raw"] = "tasks/task_1/raw/case_missing.json"
    path = directory / "missing-result.json"
    path.write_text(json.dumps(missing, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    text = render_details(missing, raw_root=directory, details="errors",
                          width=80, result_path=path)
    (root / "missing-80.txt").write_text(text + "\n", encoding="utf-8")
    logger.info("Synthetic examples (no model requests): %s", root)
    logger.info("ITL/TPOT not stored in response: N/A, no recalculation.")


if __name__ == "__main__":
    main()
