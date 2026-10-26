"""Load recorded requests without rerunning or changing benchmark facts."""

from __future__ import annotations

import json
import math
from pathlib import Path


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def first_number(record, *keys):
    return next((record[k] for k in keys if finite(record.get(k))), None)


def request_rows(requests, samples):
    """Keep request identity separate from sample identity and state."""
    sample_indices = {}
    for index, sample in enumerate(samples):
        if sample.get("sample_id") is None:
            continue
        sample_indices.setdefault(str(sample.get("sample_id", "")), []).append(
            index
        )
    rows = []
    referenced = set()
    for index, request in enumerate(requests):
        sample_id = str(request.get("sample_id", ""))
        indices = (
            sample_indices.get(sample_id, [])
            if request.get("sample_id") is not None else []
        )
        referenced.update(indices)
        response = request.get("response")
        response = response if isinstance(response, dict) else {}
        rows.append({
            "request_index": index,
            "sample_indices": indices,
            "request_id": request.get("request_id"),
            "sample_id": sample_id,
            "status": request.get("status", "unknown"),
            "sample_states": [samples[i].get("state") for i in indices],
            "input_tokens": first_number(response, "input_tokens"),
            "output_tokens": first_number(response, "output_tokens"),
            "ttft_ms": first_number(response, "ttft_ms"),
            "latency_ms": first_number(
                response, "e2el_ms", "latency_ms", "duration_ms"
            ),
        })
    for index, sample in enumerate(samples):
        if index not in referenced:
            rows.append({
                "request_index": None,
                "sample_indices": [index],
                "request_id": None,
                "sample_id": str(sample.get("sample_id", "")),
                "status": sample.get("state", "unknown"),
                "sample_states": [sample.get("state")],
            })
    return rows


def load_details(case, task, root, *, run, allow_partial=False):
    """Read only within the explicit result root, including relocated runs."""
    if case["status"] == "discarded":
        return {"status": "unavailable", "reason": "Case 已丢弃，无原始记录"}
    root = Path(root).resolve()
    recorded = case.get("raw")
    if not recorded:
        return {"status": "unavailable", "reason": "结果未记录原始文件路径"}
    conventional = (
        root / "tasks" / task["name"] / "raw" / (case["id"] + ".json")
        if run else root / "raw" / (case["id"] + ".json")
    )
    try:
        original = Path(recorded)
        candidates = [
            original if original.is_absolute() else root / original,
            conventional,
        ]
        path = next((
            candidate.resolve() for candidate in candidates
            if candidate.resolve().is_relative_to(root) and candidate.is_file()
        ), None)
        if path is None:
            raise ValueError(
                "原始文件缺失或位于结果目录之外；请复制完整运行目录，"
                "或通过 --raw-root 指定运行目录"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("原始文件必须是 JSON 对象")
        requests, samples = raw.get("requests"), raw.get("samples")
        warnings = []
        if allow_partial and ("requests" in raw or "samples" in raw):
            for key in ("requests", "samples"):
                if key not in raw:
                    warnings.append(f"原始文件未保存 {key} 数组；详情不完整")
            requests = raw.get("requests", [])
            samples = raw.get("samples", [])
        if not isinstance(requests, list) or not isinstance(samples, list):
            raise ValueError("原始文件缺少 requests/samples 数组")
        if any(not isinstance(row, dict) for row in requests + samples):
            raise ValueError("原始记录必须是 JSON 对象")
        # Reject invalid JSON numbers before embedding the whole report.
        json.dumps(raw, allow_nan=False)
        return {
            "status": "available",
            "source": str(path),
            "requests": requests,
            "samples": samples,
            "rows": request_rows(requests, samples),
            "warnings": warnings,
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
