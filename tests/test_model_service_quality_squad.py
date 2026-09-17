"""SQuAD 2.0 scoring and HTTP integration regression tests."""

import copy
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from luban_meter.benchmarking.model_service_quality.common.prompts import (
    render_squad_prompt,
)
from luban_meter.benchmarking.model_service_quality.squad import (
    calculate_metrics as result,
)
from luban_meter.benchmarking.model_service_quality.squad import (
    collect_raw as benchmark,
)
from luban_meter.benchmarking.model_service_quality.squad.dataset import (
    convert_official,
    load_samples,
    validate_records,
)
from luban_meter.benchmarking.model_service_quality.squad.scoring import (
    normalize_answer,
    parse_answer,
    score_answer,
)
from luban_meter.core.scenario_registry import ScenarioRegistry


def row(sample_id="q1", impossible=False):
    return {
        "id": sample_id,
        "title": "Example",
        "context": "The blue blue car belongs to Ada.",
        "question": "Who owns the car?",
        "answers": [] if impossible else ["Ada", "Ada Lovelace"],
        "is_impossible": impossible,
    }


def official():
    return {
        "version": "v2.0",
        "data": [
            {
                "title": "Example",
                "paragraphs": [
                    {
                        "context": "Ada owns a car.",
                        "qas": [
                            {
                                "id": "no",
                                "question": "What is its price?",
                                "is_impossible": True,
                                "answers": [],
                                "plausible_answers": [{"text": "Ada"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }


class ScoringTests(unittest.TestCase):
    def test_official_normalization_boundaries(self):
        self.assertEqual(normalize_answer("The, BLUE car!"), "blue car")
        self.assertEqual(normalize_answer("a_b"), "ab")
        self.assertEqual(normalize_answer("3.14"), "314")
        self.assertEqual(normalize_answer("“The car”"), "“ car”")
        self.assertEqual(score_answer("blue blue", ["blue"]), (0, 2 / 3))
        self.assertEqual(score_answer("Ada", ["Someone", "Ada"]), (1, 1))
        self.assertEqual(score_answer("", []), (1, 1))
        self.assertEqual(score_answer("Ada", []), (0, 0))
        self.assertEqual(score_answer("", ["the"]), (1, 1))

    def test_refusal_is_explicit(self):
        self.assertEqual(parse_answer(" unanswerable\n"), "")
        self.assertEqual(parse_answer("unanswerable."), "unanswerable.")
        self.assertEqual(
            parse_answer("3.14, approximately"), "3.14, approximately"
        )
        for value in (None, "", "  ", []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_answer(value)

    def test_loader_and_preparer(self):
        payload = official()
        rows = convert_official(payload)
        self.assertEqual(rows[0]["answers"], [])
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "dev.json"
            out = Path(directory) / "dev.jsonl"
            src.write_text(json.dumps(payload))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    (
                        "luban_meter.benchmarking.model_service_quality.scripts.prep"
                        "are_squad"
                    ),
                    "--source",
                    str(src),
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(load_samples(out), rows)
        for rows in (
            [],
            [row(), row()],
            [
                row(impossible=True)
                | {
                    "answers": ["plausible"],
                }
            ],
            [row() | {"is_impossible": "false"}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_records(rows)

    def test_parameters_and_registry(self):
        params = {"dataset_path": "some.json"}
        self.assertIsNone(benchmark.validate_parameters(params)["max_samples"])
        for key, value in (
            ("eval_mode", "ppl"),
            ("few_shot", 1),
            ("max_samples", 0),
            ("max_samples", True),
            ("prompt_version", "wrong"),
            ("max_concurrency", 0),
            ("request_timeout", float("nan")),
            ("temperature", 1),
            ("stop", "\n"),
        ):
            with (
                self.subTest(key=key),
                self.assertRaises((ValueError, TypeError)),
            ):
                benchmark.validate_parameters(params | {key: value})
        self.assertIn(
            "squad", ScenarioRegistry().list_scenarios("model_service_quality")
        )
        prompt = render_squad_prompt(row())
        self.assertIn(row()["context"], prompt)
        self.assertNotIn("Ada Lovelace", prompt)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = (
            body["messages"][0]["content"]
            if "messages" in body
            else body["prompt"]
        )
        if "FAIL" in prompt:
            self.send_error(503, "synthetic failure")
            return
        text = "unanswerable" if "NOANSWER" in prompt else "Ada"
        if "EMPTY" in prompt:
            text = ""
        data = json.dumps(
            {
                "choices": [{"text": text, "message": {"content": text}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class IntegrationTests(unittest.TestCase):
    def test_http_chat_base_failures_and_recomputation(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "data.jsonl"
                rows = [
                    row(),
                    row("q2", True) | {"question": "NOANSWER"},
                    row("q3", True) | {"question": "FAIL"},
                    row("q4", True) | {"question": "EMPTY"},
                ]
                path.write_text("\n".join(json.dumps(r) for r in rows))
                for fmt in ("chat", "base"):
                    raw = benchmark.collect_raw_result(
                        {"model_name": "fake"},
                        {
                            "dataset_path": str(path),
                            "prompt_format": fmt,
                            (
                                "service_url"
                            ): f"http://127.0.0.1:{server.server_port}",
                        },
                    )
                    final = result.process(raw)
                    self.assertEqual(final["status"], "partial_failed")
                    metrics = final["metrics"]["task_view"]["squad"]
                    self.assertEqual(metrics["exact_match"]["value"], 0.5)
                    self.assertEqual(metrics["token_f1"]["count"], 4)
                    self.assertEqual(
                        metrics["no_answer"]["token_f1"]["value"], 1 / 3
                    )
                    bad = copy.deepcopy(raw)
                    bad["metrics"]["samples"][0]["prediction"] = "tampered"
                    with self.assertRaises(ValueError):
                        result.process(bad)
                    raw["metrics"]["samples"] = [raw["metrics"]["samples"][2]]
                    self.assertEqual(result.process(raw)["status"], "failed")
                # Exercise the actual CLI, subprocess collector and
                # reports.
                path.write_text(json.dumps(row()) + "\n")
                config = Path(directory) / "squad.yaml"
                config.write_text(
                    json.dumps(
                        {
                            "dataset_path": str(path),
                            (
                                "service_url"
                            ): f"http://127.0.0.1:{server.server_port}",
                        }
                    )
                )
                output = Path(directory) / "runs"
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "luban_meter.cli",
                        "run",
                        "--module",
                        "model_service_quality",
                        "--benchmark",
                        "squad",
                        "--model-name",
                        "fake",
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                final_paths = list(output.rglob("result.json"))
                self.assertEqual(len(final_paths), 1)
                final = json.loads(final_paths[0].read_text())
                self.assertEqual(final["status"], "success")
                self.assertTrue(list(output.rglob("report.md")))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
