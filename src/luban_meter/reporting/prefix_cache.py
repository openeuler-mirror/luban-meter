"""Presentation of saved cache results; no cache arithmetic or inference."""

import json
import math

METRIC = "prefix_cache_hit_rate"
FIELDS = (
    ("target_rate", "目标缓存命中率"),
    ("theoretical_shared_ratio", "理论共享比例"),
    ("actual_hit_rate", "实际缓存命中率"),
    ("observed_subset_hit_rate", "已测子集命中率（非完整 Case）"),
    ("construction_deviation", "构造偏差"),
    ("runtime_deviation", "运行偏差"),
    ("target_deviation", "总体目标偏差"),
    ("total_prompt_tokens", "成功请求输入 Token 总数（S）"),
    ("measured_prompt_tokens", "已测子集输入 Token 总数（M）"),
    ("total_cached_tokens", "已测子集缓存命中 Token 总数（M）"),
    ("request_coverage", "请求统计覆盖率"),
    ("token_coverage", "正式 Token 统计覆盖率"),
    ("successful_request_coverage", "成功集合请求覆盖率"),
    ("successful_token_coverage", "成功集合 Token 覆盖率"),
    ("success_rate", "正式请求成功率"),
    ("input_length_conformance_rate", "输入长度符合率"),
)
STATES = {
    "valid": "有效", "partial": "部分覆盖", "unavailable": "不可测",
    "invalid": "无效", "passed": "符合", "failed": "未通过",
    "unknown": "未知", "construction_limited": "受构造上限限制",
    "success": "完成", "not_applicable": "不适用",
    "observed_mismatch": "已完成但命中不符",
    "matched": "符合", "under_hit": "命中不足", "over_hit": "高于预期",
    "missing": "缺失",
}


def cache_eligible(case):
    """Require a valid cache measurement over the full formal quota."""
    metric = case.get("metrics", {}).get(METRIC)
    if metric is None:
        return "prefix_cache_target" not in case.get("parameters", {})
    planned = case.get("parameters", {}).get("num_requests")
    rate = saved_value(metric, "actual_hit_rate")
    return (
        case.get("status") == "success"
        and metric.get("status") == "success" and not metric.get("error")
        and metric.get("measurement", {}).get("status") == "valid"
        and finite(rate) and 0 <= rate <= 1
        and type(planned) is int and planned > 0
        and all(saved_value(metric, key) == planned for key in (
            "submitted_requests", "successful_requests", "reporting_requests",
        ))
    )


def cache_report_metric(metric):
    """Read historical warmup facts without the retired verdict."""
    if "experimental_conditions" not in metric:
        return metric
    result = dict(metric)
    legacy = result.pop("experimental_conditions")
    if isinstance(legacy, dict) and "warmup" not in result:
        result["warmup"] = {
            key: value for key, value in legacy.items()
            if key.startswith("warmup_") or key == "reasons"
        }
    return result


def display_value(node, key):
    value = node.get("value")
    if type(value) not in (int, float) or not math.isfinite(value):
        return "不可测"
    unit = node.get("unit", "")
    if unit == "ratio":
        unit = "pp" if key.endswith("_deviation") else "%"
        value *= 100
    return f"{value:,.4f}".rstrip("0").rstrip(".") + " " + unit


def detailed_cache_lines(metric):
    values = metric.get("values", {})
    for key, title in FIELDS:
        yield title + ": " + display_value(values.get(key, {}), key)
    warmup = metric.get("warmup", {})
    status = warmup.get("warmup_status", "unknown")
    yield "三阶段预热（全部种入 → 全部探测 → 完整验证）: " + (
        STATES.get(status, status) + " / " + status
    )
    yield from warmup_lines(warmup)
    for key, title in (
        ("measurement", "测量有效性"),
        ("target_attainment", "目标达标"),
    ):
        status = metric.get(key, {}).get("status", "unknown")
        yield title + ": " + STATES.get(status, status) + " / " + status
    yield "S=成功正式请求；M=缓存计数有效的成功子集。不同集合总数不可相除。"
    reasons = metric.get("measurement", {}).get("reasons", [])
    if reasons:
        yield "测量诊断: " + ", ".join(map(str, reasons))
    diagnostics = metric.get("diagnostics", {})
    if diagnostics:
        yield "缓存淘汰诊断（非确认）: " + str(
            diagnostics.get("eviction", "unknown")
        )


def warmup_lines(warmup):
    """Format only saved warmup facts, including incomplete diagnostics."""
    policy = warmup.get("warmup_policy", "strict")
    yield "缓存预热模式: " + str(policy)
    for key, title in (
        ("warmup_completed", "预热完成"),
        ("warmup_fully_matched", "预热完全符合"),
        ("warmup_under_hit_prefixes", "命中不足的前缀数量"),
    ):
        value = warmup.get(key)
        display = (
            "不可测" if value is None else
            "是" if value is True else "否" if value is False else str(value)
        )
        yield title + ": " + display
    for phase, rows in warmup.get("warmup_summary", {}).items():
        for row in rows:
            state = row.get("validation_state", "unknown")
            actual = row.get("actual_cached_tokens")
            deviation = row.get("deviation_tokens")
            yield (
                f"{phase} prefix_id={row.get('prefix_id')} "
                f"expected={row.get('expected_cached_tokens')} "
                f"actual={actual if actual is not None else '不可测'} "
                f"deviation_tokens={deviation} "
                f"validation_state={STATES.get(state, state)} / {state}"
            )
    if warmup.get("reasons"):
        yield "预热条件异常: " + ", ".join(warmup["reasons"])


def redact_cache_salts(payload):
    """Copy report input, removing salt keys and known diagnostic echoes."""
    secrets = set()

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "cache_salt" and isinstance(value, str) and value:
                    secrets.add(value)
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    def clean(node):
        if isinstance(node, dict):
            return {clean(key): clean(value) for key, value in node.items()
                    if key != "cache_salt"}
        if isinstance(node, list):
            return [clean(value) for value in node]
        if isinstance(node, str):
            for secret in sorted(secrets, key=len, reverse=True):
                node = node.replace(secret, "[REDACTED]")
        return node

    collect(payload)
    return clean(payload)


def saved_value(metric, key):
    if not isinstance(metric, dict):
        return None
    values = metric.get("values") or {}
    node = values.get(key) if isinstance(values, dict) else None
    return node.get("value") if isinstance(node, dict) else None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def ratio_text(value):
    return display_value(
        {"value": value, "unit": "ratio"}, "rate",
    ).replace(" %", "%")


def cache_case_label(task, case):
    """Use a target number in display labels; keep stored task IDs intact."""
    parameters = case.get("parameters", {})
    metric = case.get("metrics", {}).get(METRIC, {})
    if not (task.get("scenario") == "prefix_cache" or metric
            or "prefix_cache_target" in parameters):
        return task["name"]
    target = saved_value(metric, "target_rate")
    if target is None:
        target = parameters.get("prefix_cache_target")
    text = ratio_text(target) if finite(target) else "未记录"
    return "前缀缓存评测（目标 " + text + "）"


def cache_task_label(task):
    labels = list(dict.fromkeys(
        cache_case_label(task, case) for case in task.get("cases", [])
    ))
    if not labels or labels == [task["name"]]:
        return task["name"]
    return " / ".join(labels)


REASONS = {
    "missing_cache_stats": "缓存统计缺失",
    "failed_formal_requests": "存在失败正式请求",
    "formal_observation_incomplete": "完整正式请求范围观测不足",
    "invalid_cache_stats": "缓存统计无效",
    "collection_incomplete": "正式采集未完成",
    "request_set_not_closed": "正式请求记录不完整",
    "no_successful_requests": "没有成功正式请求",
    "zero_token_denominator": "输入 Token 分母不可用",
    "warmup_cache_mismatch": "预热命中与预期不符",
    "warmup_not_verified": "预热未通过验证",
    "cache_mismatch": "预热命中与预期不符",
    "input_length": "服务端输入长度不符",
    "missing": "缓存统计缺失",
    "invalid": "缓存统计无效",
    "interrupted": "评测已中断",
    "HTTPStatusError": "预热 HTTP 请求失败",
    "ReadTimeout": "预热请求超时",
    "RuntimeError": "预热响应或 SSE 不完整",
    "transport_error": "预热传输失败",
}

def warmup_state(warmup):
    status = warmup.get("warmup_status", warmup.get("status", "unknown"))
    completed = warmup.get("warmup_completed")
    matched = warmup.get("warmup_fully_matched")
    if completed is None:
        completed = True if status in ("success", "not_applicable") else None
    if matched is None:
        matched = True if status in ("success", "not_applicable") else None
    if (status == "not_applicable"
            and completed is True and matched is True):
        return "前缀预热不适用，能力检查通过"
    if completed is not True:
        return "未完成" if completed is False or status == "failed" else "未知"
    return "完成，符合预期" if matched else (
        "完成，未完全符合预期" if matched is False else "完成，符合情况未知"
    )


def compact_warmup(warmup, parameters):
    text = warmup_state(warmup)
    rows = warmup.get("warmup_summary", {}).get("verify")
    expected = parameters.get("prefix_count")
    if (warmup.get("warmup_completed") is True
            and isinstance(rows, list) and type(expected) is int):
        text += f" ({len(rows)}/{expected})"
    under = warmup.get("warmup_under_hit_prefixes")
    if finite(under) and under > 0:
        text += f"；{under} 个前缀命中不足"
    capability = warmup.get("warmup_summary", {}).get("capability", [])
    if any(row.get("validation_state") == "over_hit" for row in capability):
        text += "；0% 能力检查出现非零命中"
    return text


def cache_lines(metric, *, verbose=False, parameters=None):
    """Show a compact saved-value summary, expanding only real anomalies."""
    metric = cache_report_metric(metric)
    if verbose:
        yield from detailed_cache_lines(metric)
        yield "完整指标上下文: " + json.dumps(
            metric, ensure_ascii=False,
        )
        return
    parameters = parameters or {}
    def value(key):
        return saved_value(metric, key)

    warmup = metric.get("warmup", {})
    target = value("target_rate")
    actual = value("actual_hit_rate")
    theory = value("theoretical_shared_ratio")
    yield "Prefix Cache 缓存命中率"
    yield "目标命中率: " + ratio_text(target)
    yield "实际命中率: " + ratio_text(actual)
    yield "目标偏差: " + display_value(
        metric.get("values", {}).get("target_deviation", {}),
        "target_deviation",
    )
    policy = warmup.get(
        "warmup_policy",
        parameters.get("prefix_cache_warmup_policy", "strict"),
    )
    yield "预热模式: " + {"strict": "Strict", "observe": "Observe"}.get(
        policy, str(policy),
    )
    yield "预热状态: " + compact_warmup(warmup, parameters)
    measured = value("reporting_requests")
    denominator = (
        "submitted_requests"
        if metric.get("measurement", {}).get("scope")
        == "submitted formal requests" else "successful_requests"
    )
    successful = value(denominator)
    counts = (
        f" ({measured}/{successful})"
        if type(measured) is int and type(successful) is int else ""
    )
    yield "统计覆盖率: " + ratio_text(value("request_coverage")) + counts
    measurement = metric.get("measurement", {}).get("status", "unknown")
    goal = metric.get("target_attainment", {}).get("status", "unknown")
    yield "评测结论: " + " / ".join((
        {"valid": "测量有效", "partial": "部分可测", "unavailable": "不可测",
         "invalid": "无效"}.get(measurement, "测量未知"),
        {"passed": "目标达标", "failed": "目标未达标",
         "construction_limited": "受构造上限限制"}.get(goal, "目标未判定"),
    ))
    if finite(theory) and (
        (finite(target) and theory != target)
        or (finite(actual) and actual != theory)
    ):
        yield "理论共享比例: " + ratio_text(theory)
        if finite(actual) and actual != theory:
            yield "与理论偏差: " + display_value(
                metric.get("values", {}).get("runtime_deviation", {}),
                "runtime_deviation",
            )
    if (goal == "construction_limited"
            or finite(target) and finite(theory) and theory != target):
        yield "构造限制: 目标比例受完整缓存块及末块重算限制"
    if actual is None and finite(value("observed_subset_hit_rate")):
        yield "已测子集命中率（非完整 Case）: " + ratio_text(
            value("observed_subset_hit_rate"),
        )
    token_coverage = value("token_coverage")
    if (token_coverage is None
            or token_coverage != value("request_coverage")):
        yield "Token 统计覆盖率: " + ratio_text(token_coverage)
    success = value("success_rate")
    if finite(success) and success < 1:
        yield "正式请求成功率: " + ratio_text(success)
        yield (
            "覆盖率分母为已提交正式请求，Token 分母不使用配置值补造"
            if denominator == "submitted_requests" else
            "历史覆盖率分母仅包含成功正式请求"
        )
    length_rate = value("input_length_conformance_rate")
    if finite(length_rate) and length_rate < 1:
        yield "输入长度符合率: " + ratio_text(length_rate)
    reasons = metric.get("measurement", {}).get("reasons", [])
    notes = [REASONS.get(reason, reason) for reason in reasons]
    notes += [REASONS.get(reason, reason)
              for reason in warmup.get("reasons", [])]
    planned = value("planned_requests")
    submitted = value("submitted_requests")
    if finite(planned) and finite(submitted) and submitted != planned:
        notes.append(f"正式请求配额未完成 ({submitted}/{planned})")
    if notes:
        yield "注意: " + "；".join(dict.fromkeys(notes))
    if metric.get("error") or metric.get("reason"):
        yield "指标诊断: " + str(metric.get("error") or metric["reason"])
    eviction = metric.get("diagnostics", {}).get("eviction")
    if eviction and eviction != "unknown":
        label = {"suspected": "疑似淘汰，尚未确认"}.get(eviction, eviction)
        yield "缓存淘汰诊断（非确认）: " + str(label)


def setup_lines(warmup, *, verbose=False):
    if verbose:
        yield "缓存预热诊断: " + json.dumps(warmup, ensure_ascii=False)
        yield from warmup_lines(warmup)
        return
    policy = warmup.get("warmup_policy", "strict").title()
    yield f"缓存预热: {policy}，" + warmup_state(warmup)
    reason = warmup.get("reason")
    if reason:
        yield "原因: " + REASONS.get(reason, reason)
    phase = warmup.get("phase")
    if phase:
        phases = {
            "seed": "种入", "probe": "探测",
            "verify": "完整验证", "capability": "能力检查",
        }
        name = phases.get(phase, phase)
        yield f"失败阶段: {name}，前缀编号 {warmup.get('prefix_id', '未知')}"
    if "expected_cached_tokens" in warmup:
        actual = warmup.get("cached_tokens")
        yield f"预期命中 {warmup['expected_cached_tokens']} Token，实际 " + (
            str(actual) if actual is not None else "不可测"
        )
