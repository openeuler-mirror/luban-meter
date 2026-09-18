"""Local dataset loading for model service quality benchmarking.

Re-exports the unified loaders from
:mod:`luban_meter.benchmarking.common.dataset` and keeps the
inference-specific helpers that were already available.
"""

from __future__ import annotations

from luban_meter.benchmarking.common.dataset import (
    extract_prompts,
    load_csv,
    load_json,
    load_jsonl,
    load_records,
    resolve_data_path,
    select_few_shot,
    select_records,
)

__all__ = [
    "extract_prompts",
    "load_csv",
    "load_json",
    "load_jsonl",
    "load_records",
    "resolve_data_path",
    "select_few_shot",
    "select_records",
]
