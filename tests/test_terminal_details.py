"""Request-level text reports read evidence without changing it."""

import copy
import json
import re

import pytest

from luban_meter import cli
from luban_meter.reporting import terminal_details as details
from luban_meter.reporting.html import report_data
from luban_meter.reporting.terminal import display_width, render_text
from tests.test_report_requests import fixture


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def render(payload, root, **kwargs):
    return details.render_details(payload, raw_root=root, width=80, **kwargs)


def table_records(text):
    rows = [[cell.strip() for cell in line.split("|")[1:-1]]
            for line in text.splitlines() if line.startswith("|")]
    return [row for row in rows if len(row) == 8 and row[0] != "Request ID"]


def test_errors_separate_request_and_sample_failures(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    original = raw.read_bytes()
    before = copy.deepcopy(payload)
    text = render(payload, tmp_path, details="errors")
    assert "已保存请求 2，失败/未完成请求 1，仅样本记录 1" in text
    assert "id=b" in text and "id=a" not in text
    assert "sample-only" in text and "not_started" in text
    assert "error_type=N/A" in text and "timeout" in text
    assert "response.duration_ms=5000 ms" in text
    assert "hello" not in text and "<script>" not in text
    assert "a" * 100 not in text
    assert raw.read_bytes() == original and payload == before
    html = report_data(payload, include_requests=True, raw_root=tmp_path)
    assert html["request_details"][0][0]["requests"] == facts["requests"]


@pytest.mark.parametrize("state", ["success", "failed", "timeout",
                                  "interrupted", "incomplete"])
def test_success_failure_and_interrupted_requests(tmp_path, state):
    payload, raw, facts = fixture(tmp_path)
    facts["samples"] = []
    facts["requests"] = [{"request_id": "one", "status": state}]
    save(raw, facts)
    text = render(payload, tmp_path, details="errors")
    if state == "success":
        assert "没有错误/未完成记录" in text
        assert "| Status" not in text and "| #" not in text
    else:
        assert "未保存错误描述" in text
        assert "id=one" in text
        assert "失败/未完成请求 1" in text


def test_successful_request_with_failed_sample_is_not_failed_request(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    facts["requests"] = [facts["requests"][1]]
    facts["samples"] = [{
        "sample_id": "1", "state": "failed",
        "facts": {"status": "sandbox_failed",
                  "error": {"type": "SandboxError", "message": "bad code"}},
    }]
    save(raw, facts)
    text = render(payload, tmp_path, details="errors")
    assert "失败/未完成请求 0" in text
    assert "id=a" in text and "SandboxError" in text and "bad code" in text


def test_sort_actual_fields_missing_values_last_and_no_latency_alias(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    facts["samples"] = []
    facts["requests"] = [
        {"request_id": key, "status": "success", "response": response}
        for key, response in [
            ("missing", {"duration_ms": 99999}),
            ("slow", {"e2el_ms": 20, "input_tokens": 0}),
            ("tie", {"e2el_ms": 20}),
            ("fast", {"e2el_ms": 0}),
            ("invalid", {"e2el_ms": "99", "ttft_ms": True}),
        ]
    ]
    save(raw, facts)
    text = render(payload, tmp_path, sort="e2el", limit=5)
    assert [row[0] for row in table_records(text)] == [
        "slow", "tie", "fast", "missing", "invalid",
    ]
    assert "无有效排序值 2 条" in text
    assert "耗时较慢不等于性能异常" in text
    assert "省略 0" in text
    limited = render(payload, tmp_path, sort="e2el", limit=1)
    assert [row[0] for row in table_records(limited)] == ["slow"]
    assert "省略 4" in limited
    assert "保留在末尾" in limited


def test_all_sort_fields_read_explicit_response_fields(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    facts["samples"] = []
    facts["requests"] = [
        {"request_id": str(i), "status": "success",
         "response": dict.fromkeys(details.SORT_FIELDS.values(), i)}
        for i in (1, 3, 2)
    ]
    save(raw, facts)
    for sort in details.SORT_FIELDS:
        text = render(payload, tmp_path, sort=sort, limit=1)
        assert [row[0] for row in table_records(text)] == ["3"]


def test_no_derived_metrics_or_fake_ids(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    facts["samples"] = []
    facts["requests"] = [{
        "status": "success", "start_offset_s": 0, "end_offset_s": 1,
        "response": {"ttft_ms": 10, "last_output_latency_ms": 100,
                     "output_tokens": 16, "itl_samples_ms": [6, 6]},
    }]
    save(raw, facts)
    text = render(payload, tmp_path, verbose=True)
    row, = table_records(text)
    assert row[2:].count("N/A") == 4  # input, ITL, TPOT, E2EL
    assert "id=N/A" in text and "raw_index=0" in text
    assert "request.end_offset_s=1" in text


@pytest.mark.parametrize("contents,expected", [
    (None, "原始文件缺失"), ("bad json", "详情不可用"),
    ("[]", "JSON 对象"), ('{"samples":{}}', "requests/samples 数组"),
    ('{"requests": [null], "samples": []}', "JSON 对象"),
    ('{"legacy_responses": []}', "requests/samples 数组"),
])
def test_unavailable_is_not_no_errors(tmp_path, contents, expected):
    payload, raw, _ = fixture(tmp_path)
    if contents is None:
        raw.unlink()
    else:
        raw.write_text(contents)
    text = render(payload, tmp_path, details="errors")
    assert expected in text
    assert "不能确认全部请求无错误" in text
    assert "Case status=partial_failed" in text


def test_missing_arrays_supported_only_with_explicit_partial_warning(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {"requests": [{"status": "failed", "error": "old error"}]})
    text = render(payload, tmp_path, details="errors")
    assert "未保存 samples 数组" in text and "old error" in text
    html = report_data(payload, include_requests=True, raw_root=tmp_path)
    assert html["request_details"][0][0]["status"] == "unavailable"
    save(raw, {"samples": [{"sample_id": "0", "state": "complete"}]})
    assert "未保存 requests 数组" in render(payload, tmp_path)


def test_empty_and_unknown_results_are_explicit(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {"requests": [], "samples": []})
    assert "无请求或样本记录" in render(payload, tmp_path)
    save(raw, {"requests": [{}], "samples": []})
    text = render(payload, tmp_path, details="errors")
    assert "记录状态未知" in text
    assert "不视为成功" in text
    payload["tasks"] = []
    assert "Case: 选择 0" in render(payload, tmp_path)


def test_precise_filters_load_only_selected_raw(tmp_path, monkeypatch):
    payload, _, _ = fixture(tmp_path)
    other = copy.deepcopy(payload["tasks"][0])
    other["name"] = "task_2"
    payload["tasks"].append(other)
    calls = []
    original = details.load_details

    def load(case, task, *args, **kwargs):
        calls.append(task["name"])
        return original(case, task, *args, **kwargs)

    monkeypatch.setattr(details, "load_details", load)
    text = render(payload, tmp_path, cases=["task_1/case_1"])
    assert calls == ["task_1"] and "[task_2/case_1]" not in text
    calls.clear()
    render(payload, tmp_path, tasks=["task_1"], cases=["case_1"])
    assert calls == ["task_1"]
    with pytest.raises(ValueError, match="unknown case"):
        render(payload, tmp_path, tasks=["task_2"], cases=["task_1/case_1"])


def test_limit_is_global_across_cases(tmp_path):
    payload, _, _ = fixture(tmp_path)
    case = copy.deepcopy(payload["tasks"][0]["cases"][0])
    # Multiple Cases may deliberately reference the same saved fixture.
    case["raw"] = "tasks/task_1/raw/case_1.json"
    case["id"] = "case_2"
    payload["tasks"][0]["cases"].append(case)
    text = render(payload, tmp_path, sort="e2el", limit=2)
    assert "匹配记录 6，展示 2，省略 4" in text
    assert [row[0] for row in table_records(text)] == ["a", "a"]
    groups = [line for line in text.splitlines()
              if line.startswith("| [task_1/")]
    assert "case_1" in groups[0] and "case_2" in groups[1]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_limit(tmp_path, limit):
    payload, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        render(payload, tmp_path, limit=limit)


def test_body_opt_in_bounded_escaped_and_absence(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    text = render(payload, tmp_path, include_body=True)
    assert "敏感信息" in text and "hello" in text
    assert "<script>" in text
    assert "截断" in text and len(text) < 12000
    facts["requests"][1]["response"]["text"] = "\x1b[2J恶意\n换行"
    save(raw, facts)
    text = render(payload, tmp_path, include_body=True)
    assert "\x1b" not in text and "\\u001b" in text and "\\u000a" in text
    facts["samples"] = []
    facts["requests"] = [{"status": "success"}]
    save(raw, facts)
    assert "未保存可识别" in render(payload, tmp_path, include_body=True)


def test_quality_sample_metrics_stay_separate(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {
        "requests": [{"sample_id": "0", "request_id": "q",
                      "status": "success"}],
        "samples": [{"sample_id": "0", "state": "complete", "facts": {
            "status": "success", "latency_ms": 1300.25, "input_tokens": 12,
            "output_tokens": 20, "prompt": "math question",
            "raw_output": "math answer", "correct": False,
        }}],
    })
    text = render(payload, tmp_path)
    verbose = render(payload, tmp_path, verbose=True)
    assert "latency_ms=1300.25" in verbose
    assert "样本事实（不混入请求指标）" in verbose
    assert "latency_ms=1300.25" not in text
    assert "math question" not in text
    assert "失败/未完成请求 0" in text
    row, = table_records(text)
    assert row[2:].count("N/A") == 6
    assert "math answer" in render(payload, tmp_path, include_body=True)


def test_large_input_keeps_bounded_selection(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {"samples": [], "requests": [
        {"request_id": f"r-{i}", "status": "success",
         "response": {"e2el_ms": i}}
        for i in range(16384)
    ]})
    text = render(payload, tmp_path, sort="e2el")
    assert "匹配记录 16384，展示 10，省略 16374" in text
    assert [row[0] for row in table_records(text)] == [
        f"r-{i}" for i in range(16383, 16373, -1)
    ]
    assert len(text) < 12000


@pytest.mark.parametrize("width", [20, 40, 80, 100, 120])
def test_widths_and_unicode(tmp_path, width):
    payload, raw, facts = fixture(tmp_path)
    facts["requests"][0]["request_id"] = "请求甲乙丙" * 30
    save(raw, facts)
    text = details.render_details(payload, raw_root=tmp_path, width=width)
    lines = text.splitlines()
    assert all(display_width(line) <= width for line in lines
               if not line.startswith("raw: "))
    assert "\x1b" not in text


def test_summary_never_opens_raw(tmp_path, monkeypatch):
    payload, _, _ = fixture(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("summary must not open raw")

    monkeypatch.setattr(details, "load_details", unexpected)
    assert "终端结果摘要" in render_text(payload)


def test_raw_root_and_task_result_relocation(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    task = dict(payload["tasks"][0], kind="task")
    assert "a" in [row[0] for row in table_records(
        render(task, raw.parent.parent)
    )]
    outside = tmp_path / "outside"
    outside.mkdir()
    assert "详情不可用" in render(payload, outside)
    unsafe = outside / "tasks/task_1/raw/case_1.json"
    unsafe.parent.mkdir(parents=True)
    unsafe.symlink_to(raw)
    assert "详情不可用" in render(payload, outside)


def test_export_cannot_overwrite_input_or_raw(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    path = tmp_path / "result.json"
    save(path, payload)
    raw_before, before = raw.read_bytes(), path.read_bytes()
    alias = tmp_path / "hardlink.txt"
    alias.hardlink_to(raw)
    for output in (path, raw, alias):
        assert cli.main([
            "report", "--input", str(path), "--format", "text",
            "--details", "requests", "--output", str(output),
        ]) == 2
    assert raw.read_bytes() == raw_before and path.read_bytes() == before


@pytest.mark.parametrize("extra", [
    ["--sort", "e2el"], ["--limit", "10"], ["--include-body"],
    ["--details", "requests", "--limit", "0"],
    ["--details", "errors", "--include-requests"],
])
def test_invalid_cli_combinations(tmp_path, extra, capsys):
    payload, _, _ = fixture(tmp_path)
    path = tmp_path / "result.json"
    save(path, payload)
    assert cli.main(["report", "--input", str(path), "--format", "text",
                     *extra]) == 2
    assert "CLI command failed" in capsys.readouterr().err


def test_cli_output_redirect_and_invalid_sort(tmp_path, monkeypatch, capsys):
    payload, raw, _ = fixture(tmp_path)
    path = tmp_path / "result.json"
    save(path, payload)
    before = raw.read_bytes()
    monkeypatch.setenv("COLUMNS", "80")
    args = ["report", "--input", str(path), "--format", "text",
            "--details", "requests", "--sort", "e2el", "--limit", "1"]
    assert cli.main(args) == 0
    stdout = capsys.readouterr().out
    output = tmp_path / "out/report.txt"
    assert cli.main([*args, "--output", str(output)]) == 0
    assert output.read_text() == stdout
    assert raw.read_bytes() == before
    with pytest.raises(SystemExit) as exc:
        cli.main([*args, "--sort", "made-up"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert cli.main(["report", "--input", str(path),
                     "--details", "errors"]) == 2


def test_duplicate_ids_and_missing_ids_not_guessed(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {"requests": [{"status": "success"}], "samples": [
        {"sample_id": "x", "state": "failed", "error": "first"},
        {"sample_id": "x", "state": "failed", "error": "second"},
    ]})
    text = render(payload, tmp_path, details="errors")
    assert "仅样本记录 2" in text
    assert "first" in text and "second" in text
    assert len(re.findall(r"#\d+ \[task_1/case_1\]", text)) == 2


def test_discarded_and_unsupported_legacy_result(tmp_path):
    payload, _, _ = fixture(tmp_path)
    payload["tasks"][0]["cases"][0]["status"] = "discarded"
    assert "已丢弃" in render(payload, tmp_path)
    payload["schema_version"] = "legacy"
    with pytest.raises(ValueError, match="legacy/versioned"):
        render(payload, tmp_path)


def test_response_failure_and_negative_saved_values_are_visible(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {"samples": [], "requests": [{
        "request_id": "response-failed", "status": "success",
        "response": {"status": "service_failed", "e2el_ms": -10,
                     "error": {"type": "TransportError", "message": "bad"}},
    }]})
    text = render(payload, tmp_path, details="errors")
    assert "失败/未完成请求 1" in text
    assert "response.status=service_failed" in text
    assert "TransportError" in text and "负值" in text
    assert "-10" in text


def test_failed_linked_sample_prioritized_with_real_raw_index(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    samples = [{"sample_id": "x", "state": "complete"} for _ in range(6)]
    samples[-1].update(state="failed", error="last sample failed",
                       facts={"prompt": "sixth prompt"})
    save(raw, {"requests": [{"sample_id": "x", "status": "success"}],
               "samples": samples})
    text = render(payload, tmp_path, details="errors", include_body=True)
    assert "last sample failed" in text
    assert "sample[5].facts.prompt: sixth prompt" in text
    assert "另有 1 条关联样本" in text


def test_empty_saved_body_is_distinct_from_absent_body(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {"samples": [], "requests": [{
        "status": "success", "response": {"text": ""},
    }]})
    assert "已保存空正文" in render(payload, tmp_path, include_body=True)


@pytest.mark.parametrize("width", [80, 100, 120])
def test_request_id_table_replaces_repeated_success_metadata(tmp_path, width):
    payload, raw, facts = fixture(tmp_path)
    facts["requests"] = [facts["requests"][1]]
    facts["requests"][0]["request_id"] = "request-16383"
    facts["samples"] = facts["samples"][:1]
    save(raw, facts)
    text = details.render_details(payload, raw_root=tmp_path, width=width)
    row, = table_records(text)
    assert row == ["request-16383", "success", "0", "1", "N/A",
                   "N/A", "N/A", "20"]
    assert "| [task_1/case_1]" in text
    assert "raw_index=" not in text
    assert "sample_id=" not in text
    assert "facts.status=" not in text
    assert "#1 [task_1/case_1] request" not in text
    frames = [line for line in text.splitlines()
              if line.startswith(("+", "|"))]
    assert all(display_width(line) == width for line in frames)
    verbose = render(payload, tmp_path, verbose=True)
    assert "raw_index=0" in verbose and "sample_id=1" in verbose
    assert "facts.status=" in verbose


def test_error_message_deduplicated_with_all_sources_and_types(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    message = "identical saved timeout"
    facts["requests"] = [{
        "request_id": "r", "sample_id": "1", "status": "failed",
        "error": message,
        "response": {"error": {"type": "HTTPError", "message": message}},
    }]
    facts["samples"] = [{
        "sample_id": "1", "state": "failed",
        "error": {"type": "TimeoutError", "message": message},
        "facts": {"error": message},
    }]
    save(raw, facts)
    before = raw.read_bytes()
    for verbose in (False, True):
        text = render(payload, tmp_path, details="errors", verbose=verbose)
        assert text.count(message) == 1
        assert "request.error: error_type=N/A" in text
        assert "response.error: error_type=HTTPError" in text
        assert "sample[0].error: error_type=TimeoutError" in text
        assert "sample[0].facts.error: error_type=N/A" in text
        assert "raw_index=0" in text and "sample_id=1" in text
    assert raw.read_bytes() == before


def test_different_error_messages_not_merged_by_truncated_prefix(tmp_path):
    payload, raw, facts = fixture(tmp_path)
    prefix = "same long prefix " * 40
    facts["requests"] = [{
        "request_id": "r", "sample_id": "1", "status": "failed",
        "error": prefix + "request suffix",
    }]
    facts["samples"] = [{"sample_id": "1", "state": "failed",
                         "error": prefix + "sample suffix"}]
    save(raw, facts)
    text = render(payload, tmp_path, details="errors")
    assert text.count("Error/错误:") == 2
    assert "request.error" in text and "sample[0].error" in text
    facts["samples"][0]["error"] = "different sample message"
    save(raw, facts)
    text = render(payload, tmp_path, details="errors")
    assert "different sample message" in text
    assert text.count("Error/错误:") == 2


def test_compact_list_retains_important_diagnostics(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    save(raw, {"samples": [], "requests": [{
        "request_id": "batch", "status": "success", "logical_requests": 2,
        "response": {"e2el_ms": -1, "connection_cleanup_error": "close error"},
    }]})
    text = render(payload, tmp_path)
    assert "负值" in text and "logical_requests=2" in text
    assert "close error" in text
    assert "raw_index=" not in text and "sample_id=" not in text


def test_shared_error_reference_preserves_sources_across_cases(tmp_path):
    payload, raw, _ = fixture(tmp_path)
    message = "shared across cases"
    save(raw, {"requests": [{"request_id": "r", "sample_id": "1",
                             "status": "failed", "error": message}],
               "samples": [{"sample_id": "1", "state": "failed",
                            "facts": {"error": message}}]})
    other = copy.deepcopy(payload["tasks"][0]["cases"][0])
    other.update(id="case_2", raw="tasks/task_1/raw/case_1.json")
    payload["tasks"][0]["cases"].append(other)
    text = render(payload, tmp_path, details="errors")
    assert text.count(message) == 1
    assert "[E1] 同前" in text
    assert "#1 [task_1/case_1] request" in text
    assert "#2 [task_1/case_2] request" in text
    assert text.count("来源/Source: request.error") == 2
    assert text.count("来源/Source: sample[0].facts.error") == 2
