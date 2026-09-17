"""Show recorded model service quality evaluation conditions."""

import json
from collections.abc import Mapping
from typing import Any

MISSING = "未记录"


def _format_recorded_condition(
    data: Mapping[str, Any], *paths: tuple[str, ...]
) -> str:
    """Prefer recorded facts; keep explicit nulls and mark config
    values.
    """
    for path in paths:
        value = data
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            text = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, sort_keys=True)
            )
            if path[0] == "parameters":
                text += "（配置）"
            return text
    return MISSING


def _format_parameter_condition(data: Mapping[str, Any], key: str) -> str:
    return _format_recorded_condition(
        data, ("metadata", key), ("parameters", key)
    )


def evaluation_condition_rows(data: Mapping[str, Any]) -> list[list[str]]:
    """Read shared fields, independent of individual Benchmark names."""
    if (
        data.get("module") != "model_service_quality"
        or data.get("status") == "skipped"
    ):
        return []
    rows = [
        [
            "模型",
            _format_recorded_condition(
                data,
                ("metadata", "model"),
                ("model", "name"),
                ("parameters", "model"),
            ),
        ],
        [
            "模型版本",
            _format_recorded_condition(
                data,
                ("metadata", "model_version"),
                ("model", "version"),
                ("parameters", "model_version"),
            ),
        ],
        [
            "推理引擎",
            _format_recorded_condition(
                data,
                ("metadata", "engine"),
                ("environment", "engine"),
            ),
        ],
        [
            "引擎版本",
            _format_recorded_condition(
                data,
                ("metadata", "engine_version"),
                ("environment", "engine_version"),
            ),
        ],
    ]
    rows.extend(
        [label, _format_parameter_condition(data, key)]
        for label, key in (
            ("数据集", "dataset"),
            ("数据集版本", "dataset_version"),
            ("数据划分", "split"),
            ("样本数", "sample_count"),
            ("评分模式", "eval_mode"),
            ("Prompt 格式", "prompt_format"),
            ("Prompt 版本", "prompt_version"),
            ("Few-shot 设置", "few_shot"),
            ("评分器版本", "scorer_version"),
        )
    )
    checksum = _format_parameter_condition(data, "dataset_sha256")
    if checksum != MISSING:
        rows.append(["数据集 SHA256", checksum])
    for label, keys in (
        ("样本选择", ("max_samples", "shuffle", "seed")),
        ("解码参数", ("temperature", "max_tokens", "stop")),
    ):
        rows.append(
            [
                label,
                "; ".join(
                    f"{key}={_format_parameter_condition(data, key)}"
                    for key in keys
                ),
            ]
        )
    examples = _format_parameter_condition(data, "few_shot_path")
    if examples != MISSING:
        rows.append(["Few-shot 来源", examples])
    return rows
