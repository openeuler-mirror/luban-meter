"""Step-load scheduling, capacity judgment and the search-capacity tool."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from luban_meter.benchmarking.generation_performance.capacity_search import (
    compute_capacity_search,
    is_timeout_failure,
)
from luban_meter.core.planning import build_plan
from luban_meter.runtime.load_profiles import profiles
from luban_meter.runtime.scheduling_strategies import StepStrategy


def resolve(profile, **options):
    parameters = build_plan({
        "scenario": "online_serving", "model": "mock",
        "profile": profile, **options,
    })["tasks"][0]["cases"][0]["parameters"]
    return profiles.get(profile)(parameters), parameters


class TestStepStrategy:
    def test_time_axis_follows_staged_rates(self):
        strategy = StepStrategy({
            "step_start_rate": 2.0,
            "step_rate_increment": 2.0,
            "step_stage_duration": 1.0,
            "step_max_rate": None,
        })
        offsets = [strategy.plan({}).offset_s for _ in range(12)]
        assert offsets[0] == 0
        assert offsets[:2] == pytest.approx([0.0, 0.5])
        assert offsets[2:6] == pytest.approx([1.0, 1.25, 1.5, 1.75])
        assert offsets[6:] == pytest.approx(
            [2.0 + index / 6 for index in range(6)]
        )

    def test_stage_rate_escalates_and_caps(self):
        strategy = StepStrategy({
            "step_start_rate": 1.0,
            "step_rate_increment": 10.0,
            "step_stage_duration": 0.1,
            "step_max_rate": 5.0,
        })
        assert strategy.stage_rate(0.05) == (0, 1.0)
        assert strategy.stage_rate(0.15) == (1, 5.0)
        assert strategy.stage_rate(10.05) == (100, 5.0)

    def test_zero_increment_keeps_constant_rate(self):
        strategy = StepStrategy({
            "step_start_rate": 7.0,
            "step_rate_increment": 0.0,
            "step_stage_duration": 0.5,
            "step_max_rate": 7.0,
        })
        for offset in (0.0, 0.4, 0.5, 9.9):
            assert strategy.stage_rate(offset) == (int(offset // 0.5), 7.0)

    def test_planning_is_deterministic(self):
        parameters = {
            "step_start_rate": 3.0,
            "step_rate_increment": 1.5,
            "step_stage_duration": 2.0,
            "step_max_rate": None,
        }
        first = [
            StepStrategy(parameters).plan({}).offset_s for _ in range(50)
        ]
        second = [
            StepStrategy(parameters).plan({}).offset_s for _ in range(50)
        ]
        assert first == second


class TestStepProfile:
    def test_public_entrypoint_resolves_strategy(self):
        profile, _ = resolve(
            "step", step_start_rate=2, step_rate_increment=2,
        )
        assert type(profile.create_strategy()) is StepStrategy

    def test_request_mode_matches_request_level_scenarios(self):
        assert profiles.get("step").execution_mode == "requests"

    def test_inverted_rate_cap_is_rejected_at_planning(self):
        with pytest.raises(ValueError, match="step_max_rate"):
            resolve("step", step_start_rate=10, step_max_rate=5)

    @pytest.mark.parametrize("value", [0, -1, "inf", True])
    def test_stage_duration_must_be_positive(self, value):
        with pytest.raises(ValueError):
            resolve("step", step_stage_duration=value)

    def test_unrelated_options_are_rejected(self):
        with pytest.raises(ValueError, match="unknown parameters"):
            resolve("step", arrival_process="poisson")
        with pytest.raises(ValueError, match="unknown parameters: burstiness"):
            resolve("step", burstiness=0.5)
        with pytest.raises(ValueError, match="unknown parameters: step_"):
            resolve("gamma", step_start_rate=2)

    def test_capacity_metric_is_selected_for_step_profile(self):
        _, parameters = resolve("step")
        assert "capacity_search" in parameters["metrics"]
        assert "streaming_performance" in parameters["metrics"]

    def test_explicit_metrics_override_capacity_default(self):
        _, parameters = resolve("step", metrics=["streaming_performance"])
        assert parameters["metrics"] == ["streaming_performance"]

    def test_capacity_metric_requires_step_profile(self):
        with pytest.raises(ValueError, match="step profile"):
            resolve("gamma", metrics=["capacity_search"])


def request_record(offset, *, status="success", error_type=None):
    response = {"status": status}
    if status == "success":
        response.update(
            ttft_ms=100, e2el_ms=200, output_tokens=4,
            last_output_latency_ms=180,
        )
    else:
        response["error"] = {
            "type": error_type or "HTTPError", "message": "failed",
        }
    return {
        "sample_id": "0",
        "request_id": "request",
        "status": status,
        "scheduled_offset_s": offset,
        "response": response,
    }


def step_facts(requests, *, slo=None, **overrides):
    parameters = {
        "profile": "step",
        "step_start_rate": 10.0,
        "step_rate_increment": 10.0,
        "step_stage_duration": 30.0,
        "step_max_rate": None,
        "stage_min_requests": 20,
        "stage_min_slo_rate": 0.99,
        "stage_min_success_rate": 0.99,
        "stage_max_timeout_rate": 0.01,
    }
    parameters.update(overrides)
    return {
        "parameters": parameters,
        "requests": requests,
        "metadata": {"slo_config": slo},
        "window": {"duration_s": 60.0},
        "samples": [],
    }


class TestCapacityJudgment:
    def test_search_finds_boundary(self):
        requests = (
            [request_record(index / 10) for index in range(20)]
            + [request_record(30 + index / 20) for index in range(19)]
            + [request_record(31.95, status="failed")]
        )
        result = compute_capacity_search(
            step_facts(requests, slo={"ttft_ms": 1000, "e2el_ms": 500}),
        )
        values = result["values"]
        assert [stage["verdict"] for stage in values["stages"]] == [
            "stable", "unstable",
        ]
        assert values["search"]["status"] == "found"
        assert values["search"]["max_stable_request_rate"]["value"] == 10.0
        assert values["search"]["first_unstable_stage"]["value"] == 1
        assert "success_rate" in values["stages"][1]["violated_criteria"]
        assert "slo_satisfied_rate" in values["stages"][1][
            "violated_criteria"
        ]

    def test_timeout_rate_is_a_separate_criterion(self):
        requests = (
            [request_record(index / 10) for index in range(19)]
            + [
                request_record(
                    1.95, status="failed", error_type="ReadTimeout"
                )
            ]
        )
        result = compute_capacity_search(
            step_facts(requests, stage_min_success_rate=0.9),
        )
        stage = result["values"]["stages"][0]
        assert stage["timeout_requests"]["value"] == 1
        assert stage["timeout_rate"]["value"] == 0.05
        assert stage["violated_criteria"] == ["timeout_rate"]
        assert result["values"]["search"]["status"] == "not_found"

    def test_insufficient_samples_is_inconclusive(self):
        requests = [request_record(index / 10) for index in range(10)]
        result = compute_capacity_search(step_facts(requests))
        stage = result["values"]["stages"][0]
        assert stage["verdict"] == "insufficient_samples"
        assert result["values"]["search"]["status"] == "inconclusive"
        assert result["values"]["search"]["max_stable_request_rate"][
            "value"
        ] is None

    def test_all_stable_up_to_cap_is_bounded(self):
        requests = [request_record(index / 10) for index in range(20)]
        result = compute_capacity_search(
            step_facts(requests, step_max_rate=10.0),
        )
        assert result["values"]["search"]["status"] == "bounded"
        assert result["values"]["search"]["max_stable_request_rate"][
            "value"
        ] == 10.0

    def test_undetermined_slo_blocks_stability(self):
        requests = []
        for index in range(20):
            record = request_record(index / 10)
            record["response"]["ttft_ms"] = None
            requests.append(record)
        result = compute_capacity_search(
            step_facts(requests, slo={"ttft_ms": 1000, "e2el_ms": 500}),
        )
        assert result["values"]["stages"][0]["verdict"] == "undetermined"
        assert result["values"]["search"]["status"] == "inconclusive"

    def test_without_slo_config_only_rates_judge(self):
        requests = (
            [request_record(index / 10) for index in range(20)]
            + [request_record(30 + index / 20) for index in range(15)]
            + [
                request_record(30.95, status="failed")
                for _ in range(5)
            ]
        )
        result = compute_capacity_search(step_facts(requests, slo=None))
        assert "slo" not in result["values"]["stages"][0]
        assert "slo" not in result["values"]["stages"][1]
        assert result["values"]["stages"][1]["verdict"] == "unstable"

    def test_missing_step_profile_is_unavailable(self):
        result = compute_capacity_search({
            "parameters": {"profile": "rate"},
            "requests": [request_record(0)],
            "metadata": {},
            "samples": [],
        })
        assert result["status"] == "unavailable"

    def test_timeout_classification_rule(self):
        assert is_timeout_failure(request_record(
            0, status="failed", error_type="ReadTimeout",
        ))
        assert is_timeout_failure(request_record(
            0, status="failed", error_type="TimeoutError",
        ))
        assert not is_timeout_failure(request_record(
            0, status="failed", error_type="HTTPError",
        ))
        scheduler_level = {
            "status": "failed",
            "error": "the operation timed out",
        }
        assert is_timeout_failure(scheduler_level)


class Handler(BaseHTTPRequestHandler):
    fail_after = None
    posts = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"mock"}]}')

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers["Content-Length"]))
        )
        type(self).posts += 1
        slow = (
            type(self).fail_after is not None
            and type(self).posts > type(self).fail_after
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        output_count = max(1, body["max_tokens"])
        for _ in range(output_count):
            payload = {"choices": [{"text": "x"}]}
            self.wfile.write(
                ("data: " + json.dumps(payload) + "\n\n").encode()
            )
            self.wfile.flush()
            # The slow path must exceed the 50 ms read timeout by a wide
            # margin so timeout classification stays deterministic.
            time.sleep(0.3 if slow else 0.001)
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": output_count,
            },
        }
        self.wfile.write(
            (
                "data: " + json.dumps(usage)
                + "\n\ndata: [DONE]\n\n"
            ).encode()
        )


@pytest.fixture
def service(tmp_path):
    from tests.tokenizer_fixture import make_tokenizer

    Handler.tokenizer_path = str(tmp_path / "tokenizer")
    make_tokenizer(Handler.tokenizer_path)
    Handler.posts = 0
    Handler.fail_after = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


SEARCH_ARGS = [
    "--model", "mock",
    "--tokenizer", "{tokenizer}",
    "--base-url", "{service}",
    "--input-tokens", "8",
    "--output-tokens", "3",
    "--num-warmups", "2",
    "--step-start-rate", "10",
    "--step-rate-increment", "10",
    "--step-stage-duration", "0.5",
    "--num-requests", "40",
    "--stage-min-requests", "5",
]


def search_args(service, *extra):
    values = [
        value.format(
            service=service, tokenizer=Handler.tokenizer_path
        )
        for value in SEARCH_ARGS
    ]
    return values + list(extra)


class TestSearchCapacityCli:
    def test_plan_lists_step_parameters(self, capsys):
        from luban_meter.cli import main

        assert main([
            "plan", "--scenario", "online_serving", "--model", "mock",
            "--profile", "step", "--step-start-rate", "2",
            "--step-rate-increment", "2", "--step-stage-duration", "10",
            "--step-max-rate", "8",
        ]) == 0
        output = json.loads(capsys.readouterr().out)
        parameters = output["tasks"][0]["cases"][0]["parameters"]
        assert parameters["profile"] == "step"
        assert "capacity_search" in parameters["metrics"]

    def test_rejects_other_profiles(self, service, tmp_path):
        from luban_meter.cli import main

        assert main([
            "search-capacity",
            *search_args(service, "--profile", "gamma"),
            "--result-dir", str(tmp_path / "runs"),
        ]) == 2

    def test_stable_service_yields_bounded_verdict(
        self, service, tmp_path, capsys
    ):
        from luban_meter.cli import main

        runs = tmp_path / "runs"
        assert main([
            "search-capacity",
            *search_args(service),
            "--step-max-rate", "40",
            "--result-dir", str(runs),
        ]) == 0
        conclusion_path = next(runs.glob("search-*/capacity-search.json"))
        conclusion = json.loads(conclusion_path.read_text())
        assert conclusion["status"] == "bounded"
        assert conclusion["max_stable_request_rate"]["value"] == 40.0
        assert len(conclusion["stages"]) == 4
        assert all(
            stage["verdict"] == "stable" for stage in conclusion["stages"]
        )
        text = capsys.readouterr().out
        assert "最大稳定 Request Rate" in text

    def test_degrading_service_finds_boundary_and_reports_verification(
        self, service, tmp_path
    ):
        from luban_meter.cli import main

        Handler.fail_after = 8
        runs = tmp_path / "runs"
        assert main([
            "search-capacity",
            *search_args(service),
            "--request-timeout", "0.05",
            "--verify",
            "--result-dir", str(runs),
        ]) == 1
        search_path = next(
            path for path in runs.glob("search-*/capacity-search.json")
        )
        conclusion = json.loads(search_path.read_text())
        assert conclusion["status"] == "found"
        assert conclusion["max_stable_request_rate"]["value"] == 10.0
        assert conclusion["verification"]["status"] == "unstable"
        assert conclusion["verification"]["rate"] == 10.0


class TestVerificationPlanning:
    def test_verification_runs_constant_rate_confirmation(self, tmp_path):
        from luban_meter import cli

        plan = build_plan({
            "scenario": "online_serving", "model": "mock",
            "profile": "step", "step_start_rate": 10,
            "step_rate_increment": 10, "step_stage_duration": 30,
            "step_max_rate": 40, "stage_min_requests": 20,
        })
        conclusion = {
            "status": "found",
            "search": {"max_stable_request_rate": {"value": 10.0}},
        }
        captured = {}

        def fake_run(verify_plan, args, run_id):
            captured["plan"] = verify_plan
            captured["run_id"] = run_id
            return ({
                "id": "search-verify-1",
                "tasks": [{
                    "name": "task_1",
                    "cases": [{
                        "id": "case_1",
                        "metrics": {
                            "capacity_search": {
                                "status": "success",
                                "values": {
                                    "stages": [{"verdict": "stable"}],
                                },
                            }
                        },
                    }],
                }],
            }, tmp_path)

        original = cli._run_capacity_plan
        cli._run_capacity_plan = fake_run
        try:
            args = SimpleNamespace(
                result_dir=None, observe=None, observe_url=None,
                observe_interval=1.0, log_level="info", log_dir=None,
            )
            result = cli._verify_capacity(
                {"scenario": "online_serving", "model": "mock"},
                plan, conclusion, args, "search-1",
            )
        finally:
            cli._run_capacity_plan = original
        assert result["verification"]["status"] == "confirmed"
        assert result["verification"]["rate"] == 10.0
        assert result["verification"]["duration_s"] == 60.0
        parameters = captured["plan"]["tasks"][0]["cases"][0]["parameters"]
        assert parameters["profile"] == "step"
        assert "arrival_process" not in parameters
        assert "burstiness" not in parameters
        assert parameters["step_start_rate"] == 10.0
        assert parameters["step_rate_increment"] == 0.0
        assert parameters["step_max_rate"] == 10.0
        assert parameters["num_requests"] == 600
        assert captured["run_id"].startswith("verify")
        assert Path(str(result["verification"]["result"])).name == (
            "result.json"
        )
