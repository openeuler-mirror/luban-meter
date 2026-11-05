"""Select best saved values without recalculating metric distributions."""

import json
import math

from luban_meter.core.token_totals import saved_throughput_metric

from .data import format_number, metric_groups

QUALITY_DIRECTIONS = {
    "accuracy": 1, "exact_match": 1, "pass_at_1": 1, "token_f1": 1,
    "f1": 1, "rouge1": 1, "rouge2": 1, "rougeL": 1,
    "perplexity": -1, "mean_loss": -1, "bits_per_byte": -1,
}
PROTOCOL_KEYS = (
    "data", "dataset_path", "split", "eval_mode", "few_shot",
    "few_shot_path", "input_template", "input_format", "answer_parser",
    "template", "field_mapping", "max_samples", "num_requests", "seed",
    "shuffle",
)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def serving_metric(case):
    for metric in case.get("metrics", {}).values():
        values = metric.get("values", {}) if isinstance(metric, dict) else {}
        request = values.get("request_view") or {}
        service = values.get("service_view") or {}
        if (
            any(key in request for key in ("ttft", "tpot", "itl", "e2el"))
            or any(key in service for key in (
                "request_throughput", "output_token_throughput"
            ))
        ):
            return metric
    return {}


def completed(case):
    return (case.get("collection_complete") is True
            and case.get("status") != "discarded"
            and case.get("metrics_status", "complete") == "complete")


def scalar(view, key, unit):
    node = view.get(key) or {}
    value = node.get("value")
    if node.get("unit") == unit and numeric(value) and value >= 0:
        return value
    return None


def best_line(label, points, unit, minimum=False):
    if not points:
        return f"{label}: N/A"
    value = (min if minimum else max)(point[0] for point in points)
    winners = [point for point in points if point[0] == value]
    _, task, case = winners[0]
    source = f"{task['name']}/{case['id']}"
    concurrency = case.get("parameters", {}).get("max_concurrency")
    if concurrency is not None:
        source += f" · C{concurrency}"
    if len(winners) > 1:
        source += f" · {len(winners)} 个 Case 并列"
    display = value * 100 if unit == "ratio" else value
    unit = "%" if unit == "ratio" else unit
    return f"{label}: {format_number(display)} {unit} · 来源 {source}"


def performance_lines(records):
    specs = [
        ("最大输出 Token 吞吐量", "token/s", False),
        ("最大总 Token 吞吐量（输入＋输出）", "token/s", False),
        ("最低 TTFT P99", "ms", True),
        ("最低 TPOT P99", "ms/token", True),
        ("最低 E2E P99", "ms", True),
        ("最高请求成功率", "ratio", False),
        ("最大已测配置并发", "并发", False),
    ]
    points = [[] for _ in specs]
    for task, case in records:
        metric = serving_metric(case)
        values = metric.get("values", {})
        service = values.get("service_view") or {}
        if not completed(case):
            continue

        def add(index, value):
            if value is not None:
                points[index].append((value, task, case))

        total = scalar(service, "total_requests", "request")
        success = scalar(service, "successful_requests", "request")
        failed = scalar(service, "failed_requests", "request")
        valid_counts = (
            type(total) is int and total > 0 and type(success) is int
            and 0 <= success <= total and not metric.get("error")
            and ("failed_requests" not in service or
                 (type(failed) is int and failed + success == total))
        )
        if valid_counts:
            add(5, success / total)
        concurrency = case.get("parameters", {}).get("max_concurrency")
        if (type(concurrency) is int and concurrency > 0
                and type(total) is int and total > 0
                and not metric.get("error")):
            add(6, concurrency)
        coverage = case.get("coverage", {})
        usable = (
            case["status"] == "success"
            and metric.get("status", "success") == "success"
            and not metric.get("error")
            and not any(coverage.get(k, 0) > 0
                        for k in ("failed", "incomplete"))
            and not (numeric(failed) and failed > 0)
        )
        if not usable:
            continue
        add(0, scalar(service, "output_token_throughput", "token/s"))
        throughput = saved_throughput_metric(case) or {}
        total_view = throughput.get("values", {}).get("service_view", {})
        if (throughput.get("status", "success") == "success"
                and not throughput.get("error")):
            add(1, scalar(total_view, "total_token_throughput", "token/s"))
        for index, key in ((2, "ttft"), (3, "tpot"), (4, "e2el")):
            node = (values.get("request_view") or {}).get(key) or {}
            units = ({"ms/token": 1, "s/token": 1000} if key == "tpot"
                     else {"ms": 1, "s": 1000})
            value, factor = node.get("p99"), units.get(node.get("unit"))
            count, weight = node.get("count"), node.get("weight_sum")
            if (numeric(value) and value >= 0 and factor is not None
                    and (count is None or (type(count) is int and count > 0))
                    and (weight is None or (numeric(weight) and weight > 0))):
                add(index, value * factor)
    yield "最佳性能指标"
    yield "各项独立择优，不代表同一工作点；并发为配置上限。"
    for spec, candidates in zip(specs, points):
        label, unit, minimum = spec
        yield best_line(label, candidates, unit, minimum)


def quality_lines(records):
    groups = {}
    for task, case in records:
        p = case.get("parameters", {})
        protocol = json.dumps(
            {key: p.get(key) for key in PROTOCOL_KEYS},
            sort_keys=True, ensure_ascii=False,
        )
        identity = (id(task), protocol)
        group = groups.setdefault(identity, (task, p, {}))
        if (case["status"] != "success"
                or case.get("collection_complete") is False
                or case.get("metrics_status", "complete") != "complete"):
            continue
        for name, metric in case.get("metrics", {}).items():
            if (not isinstance(metric, dict) or metric.get("error")
                    or metric.get("status", "success") != "success"):
                continue
            for path, node in metric_groups(metric.get("values", metric)):
                parts = path.split("/")
                direction = QUALITY_DIRECTIONS.get(parts[-1])
                if direction is None or any(
                    part.startswith("by_") or "_by_" in part for part in parts
                ):
                    continue
                value, unit = node.get("value"), node.get("unit", "")
                count = node.get("count")
                if (not numeric(value) or value < 0
                        or (unit == "ratio" and value > 1)
                        or (count is not None and
                            (not numeric(count) or count <= 0))):
                    continue
                key = (name, path, unit, direction)
                group[2].setdefault(key, []).append((value, task, case))
    yield "最佳能力得分"
    yield "按任务、数据集和评分条件分别择优，不跨数据集排名。"
    for index, (task, parameters, metrics) in enumerate(groups.values(), 1):
        dataset = parameters.get("dataset_path") or parameters.get("data")
        yield f"{task['name']} · 条件组 {index} · 数据集 {dataset or 'N/A'}"
        if not metrics:
            yield "最佳得分: N/A（无可用核心得分）"
        for (_, path, unit, direction), points in metrics.items():
            yield best_line(path.strip("/"), points, unit, direction < 0)


def best_summary_lines(selected):
    """Use all selected Cases, independent of the detailed display limit."""
    performance = [(t, c) for t, c in selected
                   if t["category"] == "generation_performance"]
    quality = [(t, c) for t, c in selected
               if t["category"] == "model_service_quality"]
    if performance:
        yield from performance_lines(performance)
    if quality:
        yield from quality_lines(quality)
