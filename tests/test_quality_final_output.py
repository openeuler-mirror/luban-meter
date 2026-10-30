"""Grade final answers only, preserving raw text and canonical code bodies."""

import copy
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from luban_meter.benchmarking.model_service_quality import (
    arithmetic_qa,
    code_generation,
    reading_comprehension,
    text_summarization,
)
from luban_meter.benchmarking.model_service_quality.common import choice
from luban_meter.benchmarking.model_service_quality.common.client import (
    OpenAIClient,
)
from luban_meter.benchmarking.model_service_quality.common.parsers import (
    extract_final_output,
)
from luban_meter.benchmarking.model_service_quality.common.sandbox import (
    ExecutionResult,
)
from luban_meter.core.benchmark import benchmarks
from luban_meter.core.planning import build_plan
from luban_meter.core.template_contracts import AnswerParser, answer_parsers

SCENARIOS = ["ceval", "cmmlu", "gsm8k", "squad", "lcsts", "humaneval"]
FINAL = {
    "ceval": "答案：B", "cmmlu": "答案：B", "gsm8k": "#### 42",
    "squad": "Ada", "lcsts": "简短摘要", "humaneval": "    return 1\n",
}
INVALID = [
    "", "   ", "<think>答案：B，#### 42，Ada</think>",
    "<think>未结束，#### 42", "</think>答案：B",
    "<think><think>嵌套</think></think>答案：B",
    "<think>内容</think></think>答案：B",
]


def collect(tmp_path, scenario, text, *, mode="chat", pipeline=True,
            parser=None):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{}\n')
    endpoint = "/v1/chat/completions" if mode == "chat" else "/v1/completions"
    plan = build_plan({
        "scenario": scenario, "model": "mock", "dataset_path": str(dataset),
        "endpoint": endpoint,
    })
    parameters = plan["tasks"][0]["cases"][0]["parameters"]
    implementation = benchmarks.get(scenario)
    config = implementation.domain_parameters(parameters)
    config["_pipeline"] = (
        implementation.pipeline(parameters) if pipeline else None
    )
    if parser is not None:
        config["_pipeline"] = parser
    payloads, executions = [], []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "text": text,
            "message": {"content": text, "reasoning": "独立思考字段"},
        }], "usage": {"prompt_tokens": 13, "completion_tokens": 23}})

    client = OpenAIClient("http://mock")
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        if scenario in ("ceval", "cmmlu"):
            sample = {"id": "q1", "question": "题目", "choices": [
                "甲", "乙", "丙", "丁"], "answer": "B"}
            result = choice.collect_sample(client, sample, [], config)
        elif scenario == "gsm8k":
            sample = {"id": "q1", "question": "题目", "answer": "#### 42",
                      "reference_number": 42.0}
            result = arithmetic_qa.collect_sample(client, sample, [], config)
        elif scenario == "squad":
            sample = {"id": "q1", "question": "Who?", "context": "Ada owns it",
                      "answers": ["Ada"], "is_impossible": False}
            result = reading_comprehension.collect_sample(
                client, sample, config
            )
        elif scenario == "lcsts":
            sample = {"content": "文章", "abst": "简短摘要"}
            result = text_summarization.collect_sample(
                client, sample, [], config
            )
        else:
            sample = {"task_id": "q1", "prompt": "def f():\n",
                      "test": (
                          "def check(candidate):\n    assert candidate() == 1"
                      ),
                      "entry_point": "f"}

            def execute(program):
                executions.append(program)
                return ExecutionResult(status="passed", passed=True,
                                       execution_ms=1)

            result = code_generation.collect_sample(
                client, SimpleNamespace(execute=execute),
                threading.BoundedSemaphore(1), sample, config,
            )
    finally:
        client.close()
    assert len(payloads) == 1
    assert payloads[0]["stream"] is False
    return result, executions


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("pipeline", [False, True])
def test_all_generators_filter_before_parsing_and_keep_wire_text(
    tmp_path, scenario, pipeline
):
    raw = "<think>错误答案：A，#### 999，秘密</think>\n" + FINAL[scenario]
    result, executions = collect(
        tmp_path, scenario, raw,
        mode="base" if scenario == "humaneval" else "chat",
        pipeline=pipeline,
    )
    assert result["raw_output"] == raw
    assert result["final_output"].endswith(FINAL[scenario])
    assert "秘密" not in result["final_output"]
    assert result["status"] == "success"
    assert result["input_tokens"] == 13
    assert result["output_tokens"] == 23
    if scenario in ("ceval", "cmmlu", "gsm8k"):
        assert result["correct"] is True
    if scenario == "humaneval":
        assert len(executions) == 1 and "秘密" not in executions[0]
    else:
        assert result["reasoning_text"] == "独立思考字段"


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("raw", INVALID)
def test_unusable_final_answers_fail_closed_and_keep_evidence(
    tmp_path, scenario, raw
):
    result, executions = collect(
        tmp_path, scenario, raw,
        mode="base" if scenario == "humaneval" else "chat",
    )
    assert result["status"] == "parse_failed"
    assert result["raw_output"] == raw
    assert result["final_output"] is None
    assert result["input_tokens"] == 13 and result["output_tokens"] == 23
    assert result["error"]
    assert executions == []


@pytest.mark.parametrize("raw, expected", [
    (" <THINK>秘密\n答案：A</THINK>\n答案：B", " \n\n答案：B"),
    ("前<think>秘密</think>后", "前\n后"),
    ("<think>一</think><think>二</think>正文", "\n\n正文"),
    ("  正文\n", "  正文\n"),
])
def test_text_filter_keeps_boundaries_and_whitespace(raw, expected):
    assert extract_final_output(raw) == expected


@pytest.mark.parametrize("completion", [
    '    return "<think>literal</think>"\n',
    '    # <think>unclosed is a comment\n    return 1\n',
    '    """<think>literal</think>"""\n    return 1\n',
    '```python\n    return 1\n```',
])
@pytest.mark.parametrize("prefix", ["", "<think>秘密</think>"])
def test_humaneval_preserves_code_literals_comments_and_fences(
    tmp_path, completion, prefix
):
    result, executions = collect(tmp_path, "humaneval", prefix + completion,
                                 mode="base")
    assert result["final_output"] == completion
    assert len(executions) == 1
    assert completion.rstrip() in executions[0]


def test_humaneval_multiple_prefix_blocks_preserve_indentation():
    raw = " \n<think>一</think>\n<think>二</think>\n    return 1\n"
    assert extract_final_output(raw, code=True) == "\n    return 1\n"


def test_custom_answer_parser_never_receives_thinking(tmp_path):
    class Pipeline:
        def render(self, sample, examples, config):
            return {"prompt": "Q"}

        def parse(self, sample, rendered, text):
            assert text.strip() == "答案：B"
            assert "<think>" not in text
            return {"prediction": "B"}

    result, _ = collect(tmp_path, "cmmlu", "<think>答案：A</think>\n答案：B",
                        parser=Pipeline())
    assert result["correct"] is True


@pytest.mark.parametrize("custom", [False, True])
def test_squad_replays_cleaned_output_and_rejects_tampering(tmp_path, custom):
    raw = "<think>wrong owner</think>\nAda"
    result, _ = collect(tmp_path, "squad", raw)
    if custom:
        class Parser(AnswerParser):
            def parse(self, fields):
                assert fields["raw_output"].strip() == "Ada"
                assert "<think>" not in fields["raw_output"]
                return {"prediction": fields["raw_output"].strip()}

        name = "test_final_output_reading"
        if name not in dict(answer_parsers.items()):
            answer_parsers.register(name)(Parser)
        result["answer_parser"] = name
    payload = {"status": "success", "metrics": {"samples": [result]}}
    metrics = reading_comprehension.process(payload)["metrics"]["task_view"]
    assert metrics["squad"]["exact_match"]["value"] == 1
    changed = copy.deepcopy(payload)
    changed["metrics"]["samples"][0]["final_output"] = "Other"
    with pytest.raises(ValueError, match="final_output"):
        reading_comprehension.process(changed)


def test_squad_no_answer_after_thinking_is_valid(tmp_path):
    result, _ = collect(
        tmp_path, "squad",
        "<think>no evidence</think>" + reading_comprehension.NO_ANSWER,
    )
    assert result["status"] == "success" and result["prediction"] == ""


@pytest.mark.parametrize("statuses, expected, count", [
    (["success", "parse_failed"], 50, 2),
    (["parse_failed", "parse_failed"], 0, 2),
    (["parse_failed", "service_failed"], 0, 1),
])
def test_lcsts_parse_failure_is_zero_without_calling_rouge(
    monkeypatch, statuses, expected, count
):
    calls = []

    def score(prediction, reference):
        calls.append(prediction)
        assert prediction == "摘要"
        return {key: 1.0 for key in ("rouge1", "rouge2", "rougeL")}

    monkeypatch.setattr(text_summarization, "score_one", score)
    rows = [{"status": status,
             "prediction": "摘要" if status == "success" else "",
             "reference": "空"} for status in statuses]
    result = text_summarization.process({
        "status": "success", "metrics": {"samples": rows},
    })
    metrics = result["metrics"]["task_view"]["lcsts"]
    assert metrics["rouge1"]["value"] == expected
    assert metrics["rouge1"]["count"] == count
    assert metrics["parse_failed"]["value"] == statuses.count("parse_failed")
    assert len(calls) == statuses.count("success")


def test_legacy_squad_result_without_final_output_still_replays(tmp_path):
    result, _ = collect(tmp_path, "squad", "Ada")
    result.pop("final_output")
    payload = {"status": "success", "metrics": {"samples": [result]}}
    metrics = reading_comprehension.process(payload)["metrics"]["task_view"]
    assert metrics["squad"]["exact_match"]["value"] == 1
