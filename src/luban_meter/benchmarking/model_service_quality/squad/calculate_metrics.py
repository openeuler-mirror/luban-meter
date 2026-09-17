"""Recompute SQuAD 2.0 EM/F1 from per-sample predictions and
references.
"""

from collections.abc import Mapping

from luban_meter.benchmarking.model_service_quality.squad.scoring import (
    SCORER_VERSION,
    parse_answer,
    score_answer,
)
from luban_meter.result.report_spec import bar, table

REPORT = {
    "tables": [
        table(
            "SQuAD 2.0",
            "/task_view/squad",
            {
                "/exact_match": "Exact Match",
                "/token_f1": "Token F1",
                "/has_answer/exact_match": "HasAns EM",
                "/has_answer/token_f1": "HasAns F1",
                "/no_answer/exact_match": "NoAns EM",
                "/no_answer/token_f1": "NoAns F1",
                "/total_samples": "总样本",
                "/service_failed": "服务失败",
                "/parse_failed": "解析失败",
            },
            charts=[bar("/exact_match"), bar("/token_f1")],
        )
    ],
}


def scalar(value, unit, count=None):
    metric = {"value": value, "unit": unit}
    if count is not None:
        metric["count"] = count
    return metric


def aggregate(samples):
    count = len(samples)
    return {
        key: scalar(
            sum(sample_record[key] for sample_record in samples) / count
            if count
            else None,
            "ratio",
            count,
        )
        for key in ("exact_match", "token_f1")
    }


def process(raw_result: Mapping) -> dict:
    metadata = dict(raw_result.get("metadata") or {})
    metadata.update({"scorer_version": SCORER_VERSION, "report": REPORT})
    if raw_result.get("status") == "failed":
        return {
            "status": "failed",
            "metrics": {},
            "metadata": metadata,
            "error": raw_result.get("error")
            or {
                "type": "CollectionError",
                "message": "collection failed",
            },
        }
    if raw_result.get("status") != "success":
        raise ValueError("raw status must be success or failed")
    rows = raw_result.get("metrics", {}).get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("raw samples must be a non-empty list")
    samples = []
    seen = set()
    counts = {"success": 0, "parse_failed": 0, "service_failed": 0}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") not in counts:
            raise ValueError("invalid sample or sample status")
        sample_id = row.get("id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen
        ):
            raise ValueError("sample IDs must be non-empty and unique")
        seen.add(sample_id)
        answers = row.get("reference")
        if not isinstance(answers, list) or any(
            not isinstance(answer, str) or not answer.strip()
            for answer in answers
        ):
            raise ValueError("reference must be a list of answer strings")
        impossible = row.get("is_impossible")
        if not isinstance(impossible, bool) or impossible != (not answers):
            raise ValueError("is_impossible contradicts references")
        status = row["status"]
        counts[status] += 1
        exact_match_score, token_f1_score = 0.0, 0.0
        if status == "success":
            prediction = parse_answer(row.get("raw_output"))
            if prediction != row.get("prediction"):
                raise ValueError("prediction does not match raw output")
            exact_match_score, token_f1_score = score_answer(
                prediction, answers
            )
        samples.append(
            {
                "exact_match": exact_match_score,
                "token_f1": token_f1_score,
                "is_impossible": impossible,
            }
        )
    failed = counts["service_failed"]
    status = "success"
    if failed:
        status = "failed" if failed == len(rows) else "partial_failed"
    metrics = {
        **aggregate(samples),
        "has_answer": aggregate(
            [
                sample_record
                for sample_record in samples
                if not sample_record["is_impossible"]
            ]
        ),
        "no_answer": aggregate(
            [
                sample_record
                for sample_record in samples
                if sample_record["is_impossible"]
            ]
        ),
        "total_samples": scalar(len(rows), "sample"),
        "scored_samples": scalar(len(rows), "sample"),
        "successful_samples": scalar(counts["success"], "sample"),
        "service_failed": scalar(failed, "sample"),
        "parse_failed": scalar(counts["parse_failed"], "sample"),
    }
    metadata["failure_policy"] = "zero_score_all_selected_samples"
    return {
        "status": status,
        "metrics": {"task_view": {"squad": metrics}},
        "metadata": metadata,
        "error": {
            "type": "ServiceFailures",
            "message": f"{failed} samples failed at the service",
        }
        if failed
        else None,
    }
