"""Default prompt selection, subprocess evidence and independent rates."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from luban_meter.cli import _build_parser
from luban_meter.core.planning import bootstrap, build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.runtime.evaluation_session import run_case

bootstrap()


@pytest.fixture
def service():
    class Handler(BaseHTTPRequestHandler):
        output = "解释\n答案：A"
        bodies = []

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(
                self.headers["Content-Length"]
            )))
            self.bodies.append((self.path, body))
            response = {"choices": [{
                "text": self.output, "message": {"content": self.output},
            }], "usage": {"prompt_tokens": 10, "completion_tokens": 8}}
            if self.output is None:
                response["choices"] = []
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", Handler
    server.shutdown()
    server.server_close()
    thread.join()


def dataset(tmp_path, name="ceval"):
    row = {"id": "actual-id", "question": "q", "answer": "A",
           "choices": list("abcd"), "subject": "test"}
    if name == "gsm8k":
        row = {"id": "math-id", "question": "2+2?", "answer": "#### 4"}
    path = tmp_path / f"{name}.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return {"scenario": name, "model": "mock", "dataset_path": str(path)}


def run(tmp_path, parameters):
    result, directory = RunCoordinator().run(
        build_plan(parameters), tmp_path / "out"
    )
    case = result["tasks"][0]["cases"][0]
    raw = json.loads(Path(case["raw"]).read_text()) if "raw" in case else {}
    return case, raw, directory


@pytest.mark.parametrize("command", ["plan", "run"])
def test_cli_rejects_removed_version_flag(command, capsys):
    parser, _ = _build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args([
            command, "--scenario", "ceval", "--prompt-version", "ceval-v4"
        ])
    assert error.value.code == 2
    assert "unrecognized arguments: --prompt-version" in (
        capsys.readouterr().err
    )


def test_task_json_rejects_removed_version_key(tmp_path):
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"tasks": [{
        "scenario": "ceval", "prompt_version": "ceval-v4",
    }]}))
    with pytest.raises(ValueError, match="unknown parameters: prompt_version"):
        build_plan(dataset(tmp_path), config)


@pytest.mark.parametrize("endpoint", [
    "/v1/completions", "/v1/chat/completions",
])
def test_worker_choice_rates_and_wire_prompt(service, tmp_path, endpoint):
    url, handler = service
    case, raw, directory = run(tmp_path, {
        **dataset(tmp_path), "base_url": url, "endpoint": endpoint,
        "api_key": "test-secret-value",
    })
    assert case["status"] == "success", case
    record = raw["samples"][0]["facts"]
    assert record["prompt_version"] == "ceval-v4"
    assert record["answer_parser"] == "choice"
    assert "prompt_version" not in case["parameters"]
    assert record["prediction"] == "A"
    assert record["format_compliant"] is False
    path, body = handler.bodies[0]
    assert path == endpoint
    wire = body.get("prompt") or body["messages"][0]["content"]
    assert wire == record["prompt"]
    quality = case["metrics"]["answer_output_quality"]["values"]
    assert quality["answer_parse_rate"]["value"] == 1
    assert quality["format_compliance_rate"]["value"] == 0
    assert quality["parse_failure_rate"]["value"] == 0
    meta = raw["metadata"]["evaluation"]
    assert meta["selected_samples"][0]["id"] == "actual-id"
    assert case["metrics"]["choice_accuracy"]["metadata"][
        "evaluation"
    ] == meta
    logs = list((directory / "log").glob("*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "Effective prompt protocol" in text
    assert "Raw facts saved" in text
    assert "test-secret-value" not in text
    assert "[LBM:TARGET_BEGIN]" not in text


def test_numeric_default_keeps_legacy_prediction(service, tmp_path):
    url, handler = service
    handler.output = "计算到12，尚未完成"
    case, raw, _ = run(tmp_path, {
        **dataset(tmp_path, "gsm8k"), "base_url": url,
    })
    assert case["status"] == "success"
    record = raw["samples"][0]["facts"]
    assert record["status"] == "success"
    assert record["answer_parser"] == "number"
    assert record["prompt_version"] == "gsm8k-v1"
    assert record["reference"] == 4
    assert record["prediction"] == 12
    assert record["correct"] is False
    assert record["raw_output"] == handler.output
    assert record["format_compliant"] is False


def test_custom_template_identity(service, tmp_path):
    url, handler = service
    handler.output = "#### 4"
    base = {**dataset(tmp_path, "gsm8k"), "base_url": url,
            "input_template": "format_string", "input_format": "Solve {q}",
            "field_mapping": ["q=question"],
            "answer_parser": "number"}
    _, raw, _ = run(tmp_path, base)
    row = raw["samples"][0]["facts"]
    assert row["prompt"] == "Solve 2+2?"
    assert row["prompt_version"] is None
    assert row["prompt_source"] == "custom"
    assert row["answer_parser"] == "number"
    assert row["input_format_sha256"]
    assert row["prediction"] == 4
    assert row["format_compliant"] is None


def test_few_shot_selection_and_overlap(service, tmp_path):
    url, handler = service
    handler.output = "答案：A"
    base = {**dataset(tmp_path), "base_url": url, "few_shot": 1}
    examples = tmp_path / "examples.jsonl"
    example = {
        "id": "example-id", "question": "example", "answer": "B",
        "choices": list("abcd"), "subject": "test",
    }
    examples.write_text(json.dumps(example) + "\n")
    _, raw, _ = run(tmp_path, {**base, "few_shot_path": str(examples)})
    facts = raw["samples"][0]["facts"]
    assert facts["few_shot"]["samples"][0]["id"] == "example-id"
    assert facts["few_shot"]["actual"] == 1
    assert "答案：B" in facts["prompt"]
    assert "答案：A" not in facts["prompt"]
    examples.write_text(json.dumps({**example, "question": "q"}) + "\n")
    handler.bodies.clear()
    case, _, _ = run(tmp_path, {**base, "few_shot_path": str(examples)})
    assert case["status"] == "discarded"
    assert "overlap" in case["reason"]["message"]
    assert not handler.bodies


@pytest.mark.parametrize(("scenario", "version", "row", "output"), [
    ("lcsts", "lcsts-v2", {"id": "summary-id", "content": "正文",
                          "abst": "正文"}, "摘要：正文\n第二行"),
    ("squad", "squad2-gen-v1", {"id": "reading-id", "context": "passage",
                              "question": "q", "answers": [],
                              "is_impossible": True}, "unanswerable"),
])
def test_text_task_prompt_identity(service, tmp_path, scenario, version,
                                   row, output):
    url, handler = service
    handler.output = output
    path = tmp_path / "text.jsonl"
    path.write_text(json.dumps(row) + "\n")
    case, raw, _ = run(tmp_path, {
        "scenario": scenario, "model": "mock", "base_url": url,
        "data": "jsonl", "dataset_path": str(path),
    })
    assert case["status"] == "success", case
    assert raw["metadata"]["evaluation"]["selection"]["few_shot"] == 0
    facts = raw["samples"][0]["facts"]
    assert facts["prompt_version"] == version
    assert facts["raw_output"] == output
    assert ("TARGET_BEGIN" in facts["prompt"]) is (scenario == "lcsts")
    assert facts["parse_diagnostic"]["status"] == "parsed"
    assert facts["format_compliant"] is (scenario == "squad")


@pytest.mark.parametrize("override", [{}, {"prompt_template": ""}, {
    "prompt_template": "正文：{content}\n请总结。",
}, {
    "prompt_template": "正文：{content}\n请总结。",
    "input_template": "format_string",
    "input_format": "Article: {content}",
}])
def test_lcsts_effective_renderer_identity(service, tmp_path, override):
    from luban_meter.benchmarking.model_service_quality import (
        quality_benchmarks,
    )
    from luban_meter.benchmarking.model_service_quality.common import (
        prompt_protocol,
        prompts,
    )

    url, handler = service
    handler.output = "正文\n第二行"
    sample = {"id": "summary-id", "content": "正文", "abst": "正文"}
    path = tmp_path / "summary.jsonl"
    path.write_text(json.dumps(sample) + "\n")
    plan = build_plan({
        "scenario": "lcsts", "model": "mock", "base_url": url,
        "dataset_path": str(path), "endpoint": "/v1/chat/completions",
        **{k: v for k, v in override.items() if k != "prompt_template"},
    })
    # Exercise the existing domain-level override through collection.
    class DomainOverrideLCSTS(quality_benchmarks.LCSTSBenchmark):
        @classmethod
        def domain_parameters(cls, parameters):
            domain = (
                {"prompt_template": override["prompt_template"]}
                if "prompt_template" in override else {}
            )
            return super().domain_parameters({**parameters, **domain})

    (tmp_path / "raw").mkdir()
    case = run_case(
        DomainOverrideLCSTS, plan["tasks"][0]["cases"][0], tmp_path,
        tmp_path / "stop",
    )
    raw = json.loads(Path(case["raw"]).read_text())
    record = raw["samples"][0]["facts"]
    custom_template = override.get("prompt_template")
    if "input_format" in override:
        expected = "Article: 正文"
        config_hash = prompt_protocol.fingerprint({
            "input_format": override["input_format"], "field_mapping": [],
        })
    elif custom_template:
        expected = "正文：正文\n请总结。"
        config_hash = prompt_protocol.fingerprint({
            "prompt_template": custom_template,
        })
    else:
        expected = prompts.render_lcsts_prompt(sample)["prompt"]
        config_hash = None
    assert handler.bodies[0][1]["messages"] == [
        {"role": "user", "content": expected},
    ]
    assert record["prompt"] == expected
    assert record["prompt_sha256"] == prompt_protocol.text_hash(expected)
    metrics = case["metrics"]
    evidence = raw["metadata"]["evaluation"]
    assert metrics["chinese_rouge"]["metadata"]["evaluation"] == evidence
    for identity in (record, evidence):
        assert identity["prompt_source"] == (
            "custom" if config_hash else "builtin"
        )
        assert identity["prompt_version"] == (
            None if config_hash else "lcsts-v2"
        )
        assert identity["input_format_sha256"] == config_hash
        assert identity["answer_parser"] == "text"
    assert record["prediction"] == handler.output
    assert record["format_compliant"] is (None if config_hash else False)
    assert metrics["answer_output_quality"]["values"][
        "format_compliance_rate"
    ] == {
        "value": None if config_hash else 0,
        "unit": "ratio",
        "sample_count": 0 if config_hash else 1,
    }
    assert metrics["chinese_rouge"]["values"]["task_view"]["lcsts"][
        "rougeL"
    ]["value"] == pytest.approx(100)


@pytest.mark.parametrize(("output", "reason"), [
    ("", "empty_output"),
    (" \n\t", "empty_output"),
    ("没有明确选项", "unparseable_answer"),
    ("<think>未结束", "unusable_final_output"),
    ([{"type": "text", "text": "答案：A"}], "invalid_output_type"),
])
def test_worker_parse_failure_diagnostics(service, tmp_path, output, reason):
    url, handler = service
    handler.output = output
    case, raw, _ = run(tmp_path, {
        **dataset(tmp_path), "base_url": url,
        "endpoint": "/v1/chat/completions",
    })
    facts = raw["samples"][0]["facts"]
    assert facts["raw_output"] == output
    assert facts["prediction"] is None
    assert facts["parse_diagnostic"] == {
        "status": "parse_failed", "reason": reason,
    }
    values = case["metrics"]["answer_output_quality"]["values"]
    for name, value in (
        ("answer_parse_rate", 0), ("parse_failure_rate", 1),
        ("invalid_output_rate", 1), ("format_compliance_rate", 0),
    ):
        assert values[name] == {
            "value": value, "unit": "ratio", "sample_count": 1,
        }
    scores = case["metrics"]["choice_accuracy"]["values"]["task_view"]["ceval"]
    assert scores["accuracy"]["value"] == 0


def test_worker_service_failure_is_excluded_from_diagnostic_rates(
    service, tmp_path,
):
    url, handler = service
    handler.output = None
    case, raw, _ = run(tmp_path, {**dataset(tmp_path), "base_url": url})
    record = raw["samples"][0]["facts"]
    assert record["status"] == "service_failed"
    assert record["raw_output"] is None
    assert record["parse_diagnostic"] == {
        "status": "service_failed", "reason": "service_failed",
    }
    assert record["format_compliant"] is None
    quality = case["metrics"]["answer_output_quality"]
    assert quality["status"] == "unavailable"
    for name in ("answer_parse_rate", "parse_failure_rate",
                 "invalid_output_rate", "format_compliance_rate"):
        assert quality["values"][name] == {
            "value": None, "unit": "ratio", "sample_count": 0,
        }
