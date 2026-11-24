"""Quality-only prompt identity, evidence and output diagnostics."""

import hashlib
import json
import math
import re

from luban_meter.log import get_logger

logger = get_logger(__name__)

PARSER_REVISIONS = {
    "choice": "choice-final-v1", "number": "number-legacy-v1",
    "text": "text-v1",
    "reading": "squad2-reading-v1", "humaneval_completion": "completion-v1",
}


def fingerprint(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_protocol(cls, parameters, config):
    """Describe the renderer and parser selected by the runtime config."""
    name = parameters["template"]
    template = parameters.get("input_template") or cls.default_input_template
    custom_format = None
    if template and template != cls.default_input_template:
        custom_format = {
            "input_format": parameters.get("input_format", ""),
            "field_mapping": parameters.get("field_mapping", []),
        }
    elif name == "lcsts" and config.get("prompt_template"):
        custom_format = {"prompt_template": config["prompt_template"]}
    custom = custom_format is not None
    parser = parameters.get("answer_parser") or cls.default_answer_parser
    special = not cls.default_input_template
    if special:
        parser = "humaneval_completion" if name == "humaneval" else None
    if parameters.get("eval_mode") == "ppl":
        parser = None
    return {
        "input_template": template or name,
        "prompt_source": "custom" if custom else (
            "dataset" if special else "builtin"
        ),
        "answer_parser": parser,
        "parser_revision": PARSER_REVISIONS.get(parser),
        "input_format_sha256": fingerprint(custom_format) if custom else None,
    }


def example_manifest(examples):
    return [{
        "id": row.get("id", row.get("task_id", str(index))),
        "position": index,
        "id_source": "dataset" if "id" in row or "task_id" in row
        else "position",
        "fingerprint": fingerprint(row),
        "input_fingerprint": input_fingerprint(row),
    } for index, row in enumerate(examples)]


def input_fingerprint(row):
    """Detect overlapping inputs independently of labels and source IDs."""
    fields = ("question", "choices") if "choices" in row else (
        ("question",) if "question" in row else ("content",)
    )
    return fingerprint({key: row.get(key) for key in fields})


def _gsm8k_format_compliant(answer, prompt):
    """Check the final-line protocol without extracting a prediction."""
    if (prompt and prompt in answer) or answer.count("####") != 1:
        return False
    numeric = r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?"
    line = answer.splitlines()[-1].strip()
    match = re.fullmatch(r"####[ \t]+(" + numeric + r")", line)
    return bool(match and math.isfinite(float(match[1].replace(",", ""))))


def _format_compliance(record, sample, protocol):
    template = protocol["input_template"]
    text = record.get("final_output")
    if protocol["prompt_source"] != "builtin" or template not in (
        "choice", "numeric_qa", "summarization", "reading",
    ):
        return None
    if not isinstance(text, str) or not text.strip():
        return False
    answer = text.strip()
    if any(m in answer for m in record.get("boundary_markers", [])):
        return False
    if template == "choice":
        wire = (record.get("raw_output") or "").strip()
        return bool(re.fullmatch(r"答案：[A-D]", wire))
    if template == "numeric_qa":
        return _gsm8k_format_compliant(answer, record.get("prompt", ""))
    if template == "summarization":
        return len(answer.splitlines()) == 1 and not re.match(
            r"(?:摘要\s*[:：]|摘要如下|标题\s*[:：]|[-*] |[0-9]+[.、] )",
            answer,
        )
    return answer == "unanswerable" or answer in sample["context"]


def annotate_record(record, sample, examples, config, rendered):
    """Add evidence after collection; never change predictions or scores."""
    protocol = config["_protocol"]
    actual_examples = (
        [] if protocol["input_template"] in ("format_string", "prompt_record")
        else examples
    )
    record.update({
        **protocol, "prompt": rendered["prompt"],
        "prompt_sha256": text_hash(rendered["prompt"]),
        "boundary_markers": rendered.get("boundary_markers", []),
        "sample_fingerprint": fingerprint(sample),
        "few_shot": {
            "requested": config.get("few_shot", 0),
            "actual": len(actual_examples),
            "samples": example_manifest(actual_examples),
        },
    })
    if config["eval_mode"] != "gen":
        record["parse_diagnostic"] = {"status": "not_applicable"}
        record["format_compliant"] = None
        return record
    status = record.get("status")
    if status == "service_failed":
        diagnostic = {"status": "service_failed", "reason": "service_failed"}
    else:
        parsed = status != "parse_failed"
        reason = None
        if not parsed:
            raw = record.get("raw_output")
            if raw is not None and not isinstance(raw, str):
                reason = "invalid_output_type"
            elif not raw or not raw.strip():
                reason = "empty_output"
            elif record.get("final_output") is None:
                reason = "unusable_final_output"
            else:
                reason = "unparseable_answer"
        diagnostic = {"status": "parsed" if parsed else "parse_failed",
                      "reason": reason}
    record["parse_diagnostic"] = diagnostic
    if diagnostic["status"] == "parse_failed":
        logger.warning("Answer parsing failed: sample=%s reason=%s",
                       sample.get("id", sample.get("task_id", "unidentified")),
                       diagnostic["reason"])
    record["format_compliant"] = _format_compliance(
        record, sample, protocol
    ) if status != "service_failed" else None
    return record
