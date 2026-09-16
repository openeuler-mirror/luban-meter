"""Compute LCSTS ROUGE metrics from raw per-sample records.

Uses jieba segmentation and rouge-chinese library to match
OpenCompass JiebaRougeEvaluator scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from luban_meter.result.report_spec import bar, table

REPORT = {
    "tables": [
        table(
            "LCSTS ROUGE",
            "/task_view/lcsts",
            {
                "/rouge1": "ROUGE-1 F1",
                "/rouge2": "ROUGE-2 F1",
                "/rougeL": "ROUGE-L F1",
                "/total_samples": "总样本",
                "/scored_samples": "有效评分",
                "/service_failed": "服务失败",
            },
            charts=[
                bar("/rouge1"),
                bar("/rouge2"),
                bar("/rougeL"),
            ],
        )
    ]
}

SCORER_VERSION = "metrics-v1"
SAMPLE_STATUSES = ("success", "service_failed")


def scalar(
    value: float, unit: str, count: int | None = None
) -> dict[str, Any]:
    metric: dict[str, Any] = {"value": value, "unit": unit}
    if count is not None:
        metric["count"] = count
    return metric


def object_list(
    value: Any, name: str
) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(
            not isinstance(item, Mapping)
            for item in value
        )
    ):
        raise TypeError(
            f"{name} must be a list of objects"
        )
    return list(value)


def process_sample(
    raw_sample: Mapping[str, Any],
) -> Mapping[str, Any]:
    status = raw_sample.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(
            f"sample status must be one of "
            f"{SAMPLE_STATUSES}"
        )
    return raw_sample


def postprocess(text: str) -> str:
    """Two-layer post-processing matching OpenCompass.

    1. ``lcsts_postprocess``: first line, strip prefixes
       and surrounding Chinese punctuation.
    2. ``general_postprocess``: truncate at first
       newline/period/comma, remove punctuation and
       articles, collapse whitespace.
    """
    from luban_meter.benchmark.inference.common.parsers import (
        general_postprocess,
        lcsts_postprocess,
    )
    return general_postprocess(
        lcsts_postprocess(text)
    )


def score_one(
    prediction: str, reference: str
) -> dict[str, float]:
    """Compute ROUGE-1/2/L F-measures for one pair.

    Matches OpenCompass JiebaRougeEvaluator: jieba
    tokenize then ``rouge_chinese.Rouge().get_scores``.
    """
    import jieba
    from rouge_chinese import Rouge

    pred_tokens = " ".join(
        jieba.cut(postprocess(prediction))
    )
    ref_tokens = " ".join(
        jieba.cut(postprocess(reference))
    )
    if not pred_tokens.strip():
        pred_tokens = "空"
    if not ref_tokens.strip():
        ref_tokens = "空"
    scores = Rouge().get_scores(
        pred_tokens, ref_tokens
    )
    return {
        "rouge1": scores[0]["rouge-1"]["f"],
        "rouge2": scores[0]["rouge-2"]["f"],
        "rougeL": scores[0]["rouge-l"]["f"],
    }


def process(
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    raw_status = raw_result.get("status")
    metadata = raw_result.get("metadata")
    result_meta = (
        dict(metadata)
        if isinstance(metadata, Mapping) else {}
    )

    if raw_status == "failed":
        error = raw_result.get("error") or {
            "type": "UnknownError",
            "message": "raw result failed",
        }
        result_meta.setdefault(
            "scorer_version", SCORER_VERSION
        )
        return {
            "status": "failed",
            "metrics": {},
            "metadata": result_meta,
            "error": error,
        }
    if raw_status != "success":
        raise ValueError(
            f"raw status must be success or "
            f"failed: {raw_status}"
        )

    raw_metrics = raw_result.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("raw metrics must be an object")
    raw_samples = object_list(
        raw_metrics.get("samples"), "raw samples"
    )
    if not raw_samples:
        raise ValueError(
            "raw samples must not be empty"
        )

    samples = [process_sample(s) for s in raw_samples]
    scored = [
        s for s in samples
        if s.get("status") == "success"
    ]
    service_failed = sum(
        1 for s in samples
        if s.get("status") == "service_failed"
    )
    if not scored:
        raise ValueError(
            "all samples failed at the service"
        )

    rouge = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
    }
    for sample in scored:
        scores = score_one(
            sample["prediction"],
            sample["reference"],
        )
        for key in rouge:
            rouge[key] += scores[key]

    n = len(scored)
    avg = {
        k: v / n for k, v in rouge.items()
    }

    result_status = (
        "success" if service_failed == 0
        else "partial_failed"
    )
    result_meta.setdefault(
        "scorer_version", SCORER_VERSION
    )
    result_meta.update({
        "service_failed_samples": service_failed,
        "report": REPORT,
    })
    return {
        "status": result_status,
        "metrics": {
            "task_view": {
                "lcsts": {
                    "rouge1": scalar(
                        avg["rouge1"] * 100,
                        "score", n,
                    ),
                    "rouge2": scalar(
                        avg["rouge2"] * 100,
                        "score", n,
                    ),
                    "rougeL": scalar(
                        avg["rougeL"] * 100,
                        "score", n,
                    ),
                    "total_samples": scalar(
                        len(samples), "sample"
                    ),
                    "scored_samples": scalar(
                        n, "sample"
                    ),
                    "service_failed": scalar(
                        service_failed, "sample"
                    ),
                }
            }
        },
        "metadata": result_meta,
        "error": (
            {
                "type": "ServiceFailures",
                "message": (
                    f"{service_failed} samples "
                    "failed at the service"
                ),
            }
            if service_failed else None
        ),
    }
