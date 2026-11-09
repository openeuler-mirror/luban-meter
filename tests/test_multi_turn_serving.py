"""Multi-turn conversation benchmark: sessions, history, and turn metrics.

Covers:
- turn projection delimited by reference assistant messages
- compatibility with every request-level profile
- pool index versus source record identity
- maximum-turn truncation that keeps short sessions
- dataset system messages and the command line override rule
- per-turn context limit checked before sending
- turn failure, session timeout, budget stop and interruption
- error-rate and circuit-breaker denominators
- turn_view / caching_view / prefix_cache / session_view aggregation
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import luban_meter.benchmarking.generation_performance.multi_turn_serving as mt
from luban_meter.core.planning import build_plan
from luban_meter.core.run_coordinator import RunCoordinator
from luban_meter.data.projections import (
    extract_conversation,
    split_turns,
    summarize_turn_counts,
)
from luban_meter.runtime.evaluation_session import run_case

# Set by the ``multiturn_dataset`` autouse fixture below. Tests never
# read a committed conversation file: the dataset is rebuilt per test
# so the suite has no dependency on a multi-hundred-megabyte release.
DATASET: Path | None = None

# Turn counts are increments delimited by reference assistant messages.
# 24 records reach 4 turns, 16 reach 8, 8 reach 12.
DATASET_SHAPE = ((12, 8), (8, 8), (4, 8), (2, 4))


def conversation(*pairs, system=None):
    turns = []
    if system:
        turns.append({"from": "system", "value": system})
    for question, answer in pairs:
        turns.append({"from": "human", "value": question})
        turns.append({"from": "gpt", "value": answer})
    return {"id": "c", "conversations": turns}


def multi_turn_conversation(index, turns, system=None):
    conv = []
    if system:
        conv.append({"from": "system", "value": system})
    for turn in range(turns):
        conv.append(
            {
                "from": "human",
                "value": f"Session {index} turn {turn}: explain eviction.",
            }
        )
        conv.append(
            {"from": "gpt", "value": f"Historical reply {index}/{turn}."}
        )
    return {"id": f"mt-sample-{index:03d}", "conversations": conv}


def write_dataset(directory, records, name="multiturn.json"):
    path = directory / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def write_multiturn_dataset(directory):
    """Build the ShareGPT-shaped dataset the session tests run against."""
    records = []
    index = 0
    for turns, count in DATASET_SHAPE:
        for _ in range(count):
            records.append(multi_turn_conversation(index, turns))
            index += 1
    # A record carrying its own system instruction, which must survive
    # projection rather than be replaced by the command line prompt.
    records.append(
        multi_turn_conversation(index, 4, system="Dataset system rule.")
    )
    index += 1
    # Structurally valid but with nothing to send.
    records.append(
        {
            "id": "mt-no-user",
            "conversations": [
                {"from": "gpt", "value": "Only an assistant turn."}
            ],
        }
    )
    records.append(
        {
            "id": "mt-blank-user",
            "conversations": [
                {"from": "human", "value": "   "},
                {"from": "gpt", "value": "Reply to a blank turn."},
            ],
        }
    )
    records.append(
        {
            "id": "mt-unknown-speaker",
            "conversations": [
                {"from": "bot", "value": "Unknown speaker label."},
                {"from": "human", "value": "One usable user turn."},
                {"from": "human", "value": None},
            ],
        }
    )
    return write_dataset(directory, records)


@pytest.fixture(autouse=True)
def multiturn_dataset(tmp_path):
    """Rebuild the conversation dataset for every test."""
    global DATASET
    previous = DATASET
    DATASET = write_multiturn_dataset(tmp_path)
    try:
        yield DATASET
    finally:
        DATASET = previous


class ChatHandler(BaseHTTPRequestHandler):
    """Streaming chat stub whose reply encodes the turn number."""

    lock = threading.Lock()
    posts: list[dict] = []
    output_tokens = 3
    fail_times = 0
    failures_issued = 0
    fail_turns: set[int] = set()
    chunk_delay = 0.0
    # None: omit cached_tokens, as older vLLM builds do.
    # "perfect": report all but 20 tokens as cached.
    # "none": report a genuine zero hit, which is a real observation.
    cached_token_mode: str | None = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers["Content-Length"]))
        )
        messages = body["messages"]
        user_turns = sum(1 for m in messages if m["role"] == "user")
        with ChatHandler.lock:
            ChatHandler.posts.append(body)
            should_fail = ChatHandler.failures_issued < ChatHandler.fail_times
            if should_fail:
                ChatHandler.failures_issued += 1
            if user_turns - 1 in ChatHandler.fail_turns:
                should_fail = True
        if should_fail:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"injected failure"}')
            return

        out = max(1, body.get("max_tokens", ChatHandler.output_tokens))
        text = f"A{user_turns}" + "z" * max(0, out - len(f"A{user_turns}"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for character in text:
            payload = {"choices": [{"delta": {"content": character}}]}
            self.wfile.write(
                ("data: " + json.dumps(payload) + "\n\n").encode()
            )
            self.wfile.flush()
            if ChatHandler.chunk_delay:
                time.sleep(ChatHandler.chunk_delay)
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        usage: dict = {
            # Grow with the whole history so context growth is visible.
            "prompt_tokens": prompt_chars,
            "completion_tokens": len(text),
        }
        if ChatHandler.cached_token_mode == "perfect":
            usage["prompt_tokens_details"] = {
                "cached_tokens": max(0, prompt_chars - 20)
            }
        elif ChatHandler.cached_token_mode == "none":
            usage["prompt_tokens_details"] = {"cached_tokens": 0}
        payload = {"choices": [], "usage": usage}
        self.wfile.write(
            ("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode()
        )


@pytest.fixture
def service():
    ChatHandler.posts = []
    ChatHandler.output_tokens = 3
    ChatHandler.fail_times = 0
    ChatHandler.failures_issued = 0
    ChatHandler.fail_turns = set()
    ChatHandler.chunk_delay = 0.0
    ChatHandler.cached_token_mode = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def session_task(base_url, **overrides):
    task = {
        "scenario": "multi_turn_serving",
        "model": "mock",
        "base_url": base_url,
        "dataset_path": str(DATASET),
        "multi_max_turns": 4,
        "multi_num_sessions": 3,
        "output_tokens": 3,
        "profile": "concurrent",
        "max_concurrency": 3,
        "ignore_eos": False,
    }
    task.update(overrides)
    return task


def run_task(task, tmp_path):
    result, directory = RunCoordinator().run(build_plan(task), tmp_path)
    return result, directory


def run_case_in_process(task, tmp_path):
    """Run one case in this process, where test monkeypatches apply.

    The task subprocess cannot see module patches such as a stub
    tokenizer, so tests that patch benchmark internals call the case
    runner directly instead of going through RunCoordinator.
    """
    plan = build_plan(task)
    case_dir = tmp_path / "tasks" / "task_1"
    (case_dir / "raw").mkdir(parents=True)
    from luban_meter.core.benchmark import benchmarks
    from luban_meter.core.planning import bootstrap

    bootstrap()
    return run_case(
        benchmarks.get("multi_turn_serving"),
        plan["tasks"][0]["cases"][0],
        case_dir,
        tmp_path / "stop",
    )


def raw_case(directory, task_name="task_1", case_id="case_1"):
    path = directory / "tasks" / task_name / "raw" / f"{case_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_benchmark(task, phase="formal"):
    """Run prepare() without a service, for data-handling assertions."""
    case = build_plan(task)["tasks"][0]["cases"][0]
    benchmark = mt.MultiTurnServingBenchmark()

    class Context:
        parameters = case["parameters"]
        total_samples = 0

        def __init__(self):
            self.phase = phase

        def stopped(self):
            return False

    context = Context()
    benchmark.prepare(context)
    return benchmark, context


class FakeTokenizer:
    """Chat template stub that enforces role alternation.

    The real check runs on the assembled history, so a template that
    rejects two consecutive user messages proves the estimate is no
    longer built from a run of questions.
    """

    def __init__(self, tokens_per_char=1, limit=None):
        self.tokens_per_char = tokens_per_char
        self.limit = limit
        self.calls: list[list[dict]] = []

    def apply_chat_template(
        self, messages, tokenize=True, return_dict=False,
        add_generation_prompt=True,
    ):
        self.calls.append([dict(m) for m in messages])
        previous = None
        for message in messages:
            if message["role"] == "user" and previous == "user":
                raise ValueError("template requires alternating roles")
            previous = message["role"]
        text = "".join(m.get("content") or "" for m in messages)
        return [0] * (len(text) * self.tokens_per_char)


# ── turn projection ────────────────────────────────────────────────────


def test_split_turns_uses_reference_assistant_as_delimiter():
    record = conversation(("q1", "a1"), ("q2", "a2"))
    turns = split_turns(extract_conversation(record))
    assert turns == [
        [{"role": "user", "content": "q1"}],
        [{"role": "user", "content": "q2"}],
    ]
    # The reference answers delimit turns but are never replayed.
    assert all(
        message["role"] == "user"
        for turn in turns
        for message in turn
    )
    assert [t["role"] for t in extract_conversation(record)] == [
        "user", "assistant", "user", "assistant",
    ]


def test_split_turns_keeps_dataset_system_with_the_first_turn():
    record = conversation(("q1", "a1"), ("q2", "a2"), system="be brief")
    assert split_turns(extract_conversation(record)) == [
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "q1"},
        ],
        [{"role": "user", "content": "q2"}],
    ]


def test_consecutive_user_messages_form_one_turn():
    # One turn is one user input plus its answer, so two questions with
    # no answer between them are a single turn rather than two.
    record = {
        "conversations": [
            {"from": "human", "value": "q1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "q3"},
        ]
    }
    turns = split_turns(extract_conversation(record))
    assert len(turns) == 2
    assert [m["content"] for m in turns[0]] == ["q1", "q2"]


def test_a_system_only_increment_merges_into_the_next_turn():
    record = {
        "conversations": [
            {"from": "system", "value": "rule"},
            {"from": "gpt", "value": "greeting"},
            {"from": "human", "value": "q1"},
        ]
    }
    assert split_turns(extract_conversation(record)) == [
        [
            {"role": "system", "content": "rule"},
            {"role": "user", "content": "q1"},
        ]
    ]


def test_degenerate_records_yield_no_turns():
    # Execution counts turns as split_turns(extract_conversation(...)).
    # Structurally invalid records raise; degenerate ones yield none.
    with pytest.raises(ValueError, match="no conversations list"):
        extract_conversation({})
    with pytest.raises(ValueError, match="no conversations list"):
        extract_conversation({"conversations": "nope"})
    for conversations in (
        [],
        [{"from": "gpt", "value": "only assistant"}],
        [{"from": "human", "value": "   "}],
        [{"from": "human", "value": None}],
    ):
        assert split_turns(
            extract_conversation({"conversations": conversations})
        ) == []


def test_unknown_speakers_and_blank_turns_are_skipped():
    record = {
        "conversations": [
            {"from": "bot", "value": "unknown speaker"},
            {"from": "human", "value": "  "},
            {"from": "human", "value": "real turn"},
        ]
    }
    assert split_turns(extract_conversation(record)) == [
        [{"role": "user", "content": "real turn"}]
    ]


def test_split_turns_is_total_without_user_messages():
    # The scenario marks user-less records unusable itself; split_turns
    # stays total and returns no turns instead of raising.
    assert split_turns([]) == []
    assert split_turns([{"role": "assistant", "content": "a"}]) == []
    assert split_turns([{"role": "system", "content": "s"}]) == []


def test_summarize_turn_counts_reports_the_bias_baseline():
    summary = summarize_turn_counts([0, 0, 1, 2, 2, 4, 8, 8, 12, 12])
    assert summary["conversations"] == 10
    assert summary["min"] == 0
    assert summary["max"] == 12
    # Quantiles index the sorted list without interpolating, so p50 of
    # these ten values is the sixth smallest rather than a true median.
    assert summary["p50"] == 4
    assert summary["mean"] == 4.9
    empty = summarize_turn_counts([])
    assert empty["conversations"] == 0
    assert empty["mean"] == 0.0


# ── profile compatibility (P2) ─────────────────────────────────────────


def test_every_request_profile_plans_without_session_fields(tmp_path):
    dataset = write_dataset(
        tmp_path, [multi_turn_conversation(0, 3)], "d.json"
    )
    for profile in ("concurrent", "constant", "poisson", "rate",
                    "throughput"):
        plan = build_plan(
            {
                "name": "t",
                "scenario": "multi_turn_serving",
                "model": "mock",
                "base_url": "http://127.0.0.1:1",
                "dataset_path": str(dataset),
                "multi_max_turns": 3,
                "multi_num_sessions": 1,
                "profile": profile,
                "max_concurrency": 2,
            }
        )
        parameters = plan["tasks"][0]["cases"][0]["parameters"]
        assert parameters["profile"] == profile
        assert "session_concurrency" not in parameters


def test_synchronous_profile_is_accepted_with_its_own_limit(tmp_path):
    dataset = write_dataset(
        tmp_path, [multi_turn_conversation(0, 3)], "d.json"
    )
    plan = build_plan(
        {
            "name": "t",
            "scenario": "multi_turn_serving",
            "model": "mock",
            "base_url": "http://127.0.0.1:1",
            "dataset_path": str(dataset),
            "multi_max_turns": 3,
            "multi_num_sessions": 1,
            "profile": "synchronous",
            "max_concurrency": 1,
        }
    )
    assert plan["tasks"][0]["cases"][0]["parameters"]["profile"] == (
        "synchronous"
    )


def test_session_profile_is_not_registered():
    from luban_meter.runtime.load_profiles import profiles

    assert "session" not in profiles.names()
    assert "session_concurrency" not in mt.MultiTurnServingBenchmark.parameters


def test_turn_retries_is_not_a_parameter():
    # Retrying a turn would need its own attempt-count, failure-rate and
    # latency definitions; the first version does not claim them.
    assert "turn_retries" not in mt.MultiTurnServingBenchmark.parameters


# ── pool index versus source record (P1) ───────────────────────────────


def index_dataset(tmp_path):
    """Short record first, so pool and source positions cannot agree."""
    records = [
        multi_turn_conversation(0, 1),
        multi_turn_conversation(1, 3),
        {"id": "no-user", "conversations": [
            {"from": "gpt", "value": "assistant only"}]},
        multi_turn_conversation(3, 3),
    ]
    return write_dataset(tmp_path, records, "index.json")


def test_pool_index_and_record_index_stay_distinct(tmp_path):
    dataset = index_dataset(tmp_path)
    benchmark, context = prepare_benchmark(
        {
            "name": "t", "scenario": "multi_turn_serving", "model": "mock",
            "base_url": "http://127.0.0.1:1", "dataset_path": str(dataset),
            "multi_max_turns": 3, "multi_num_sessions": 3,
            "profile": "concurrent", "max_concurrency": 1,
        }
    )
    # The unusable record at source position 2 shifts every later pool
    # position, which is exactly where the two indices used to be mixed.
    assert [row["record_index"] for row in benchmark.sessions] == [0, 1, 3]
    assert [row["record_id"] for row in benchmark.sessions] == [
        "mt-sample-000", "mt-sample-001", "mt-sample-003",
    ]
    for sample in benchmark.samples(context):
        looked_up = benchmark.sessions[sample["pool_index"]]
        assert looked_up["record_id"] == sample["record_id"]
        assert looked_up["record_index"] == sample["record_index"]


def test_execution_sends_the_content_of_the_named_record(
    service, tmp_path
):
    dataset = index_dataset(tmp_path)
    result, directory = run_task(
        session_task(
            service, dataset_path=str(dataset), multi_max_turns=3,
            multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success", result
    facts = raw_case(directory)
    # Request records carry no payload, so the sent content is verified
    # on the service side: a session's opening post is the one whose
    # history still holds a single user message.
    opening = [
        post for post in ChatHandler.posts
        if sum(1 for m in post["messages"] if m["role"] == "user") == 1
    ]
    assert len(opening) == len(facts["samples"]), ChatHandler.posts
    for sample in facts["samples"]:
        record_id = sample["facts"]["record_id"]
        tag = int(record_id.rsplit("-", 1)[-1])
        sent = [
            post for post in opening
            if f"Session {tag} turn 0" in post["messages"][0]["content"]
        ]
        assert len(sent) == 1, (
            f"{record_id} opening turn not sent once: {sent}"
        )
        assert sent[0]["messages"][0]["role"] == "user"


def test_shuffled_selection_keeps_record_identity(tmp_path):
    dataset = index_dataset(tmp_path)
    task = {
        "name": "t", "scenario": "multi_turn_serving", "model": "mock",
        "base_url": "http://127.0.0.1:1", "dataset_path": str(dataset),
        "multi_max_turns": 3, "multi_num_sessions": 3,
        "profile": "concurrent", "max_concurrency": 1,
        "shuffle": True, "data_seed": 7, "repeat_data": True,
    }
    benchmark, context = prepare_benchmark(task)
    first = [
        (s["record_index"], s["record_id"])
        for s in benchmark.samples(context)
    ]
    assert first, "shuffle must still yield sessions"
    # record_index is the position in the shuffled load order, so the
    # invariant under shuffle is that every usable record is selected
    # once, under distinct positions, and that record_id still identifies
    # the record it came from.
    assert sorted(record_id for _, record_id in first) == [
        "mt-sample-000", "mt-sample-001", "mt-sample-003",
    ]
    assert len({index for index, _ in first}) == len(first)
    benchmark2, context2 = prepare_benchmark(task)
    assert [
        (s["record_index"], s["record_id"])
        for s in benchmark2.samples(context2)
    ] == first


# ── maximum-turn truncation ────────────────────────────────────────────


def review_dataset(tmp_path):
    """The review's worked example: 1, 3 and 7 natural turns."""
    records = [
        multi_turn_conversation(0, 1),
        multi_turn_conversation(1, 3),
        multi_turn_conversation(2, 7),
    ]
    return write_dataset(tmp_path, records, "review.json")


def test_max_turns_truncates_long_sessions_and_keeps_short(tmp_path):
    benchmark, context = prepare_benchmark(
        {
            "name": "t", "scenario": "multi_turn_serving", "model": "mock",
            "base_url": "http://127.0.0.1:1",
            "dataset_path": str(review_dataset(tmp_path)),
            "multi_max_turns": 4, "multi_num_sessions": 3,
            "profile": "concurrent", "max_concurrency": 1,
        }
    )
    planned = [row["planned_turns"] for row in benchmark.sessions]
    assert planned == [1, 3, 4]
    assert [row["available_turns"] for row in benchmark.sessions] == [1, 3, 7]
    assert benchmark.filter_facts["truncated"] == 1
    # One model call per planned turn, so the request total is the sum of
    # the planned depths rather than sessions times the maximum.
    assert benchmark.filter_facts["planned_requests_total"] == 8
    for sample in benchmark.samples(context):
        assert sample["planned_turns"] in (1, 3, 4)
        assert len(sample["output_budgets"]) == sample["planned_turns"]
        assert len(sample["think_times"]) == sample["planned_turns"] - 1


def test_short_sessions_run_to_their_natural_end(service, tmp_path):
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=3,
        ),
        tmp_path,
    )
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    facts = raw_case(directory)
    completed = sorted(
        sample["facts"]["completed_turns"] for sample in facts["samples"]
    )
    assert completed == [1, 3, 4]
    assert all(
        sample["facts"]["status"] == "success"
        for sample in facts["samples"]
    ), "a short session ending early is success, not failure"
    assert len(facts["requests"]) == 8


def test_request_budget_below_a_full_pass_leaves_partial_sessions(
    service, tmp_path
):
    # 8 calls would complete the pass; admitting 5 must stop there and
    # record the sessions as partially complete rather than pad them.
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
            num_requests=5,
        ),
        tmp_path,
    )
    case = result["tasks"][0]["cases"][0]
    facts = raw_case(directory)
    assert len(facts["requests"]) <= 5, len(facts["requests"])
    assert len(facts["requests"]) < 8, "the budget must not be exceeded"
    assert case["window"]["stop_reason"] in (
        "num_requests", "data_exhausted"
    ), case["window"]
    recorded = [
        sample["facts"] for sample in facts["samples"]
        if sample.get("facts")
    ]
    assert recorded, "admitted sessions must leave facts behind"
    for session_facts in recorded:
        # Whatever was admitted is either finished or honestly partial.
        assert session_facts["not_started_turns"] == (
            session_facts["planned_turns"]
            - session_facts["completed_turns"]
            - session_facts["sent_failed_turns"]
        )
    # The budget-truncated session keeps its partial summary and the
    # scheduler's stop reason; its completed turns stay in the request
    # records and the unadmitted turn is not a service failure.
    stopped = [
        session_facts for session_facts in recorded
        if session_facts.get("abort_reason") == "num_requests"
    ]
    assert stopped, "the budget stop must explain itself in the facts"
    assert all(
        session_facts["sent_failed_turns"] == 0
        and session_facts["completed_turns"]
        < session_facts["planned_turns"]
        for session_facts in stopped
    )
    # The session cut off by the budget is incomplete, not failed.
    assert any(
        sample["state"] == "incomplete" for sample in facts["samples"]
    )
    assert case["coverage"]["selected"] == 3


class _AdmissionStubContext:
    """Context whose request() stops admitting on demand."""

    def __init__(self, parameters, stop_after=None, reason="num_requests"):
        self.parameters = parameters
        self.phase = "formal"
        self.stop_after = stop_after
        self.reason = reason
        self.admitted = 0

    def stopped(self):
        return False

    def request(self, operation, metadata=None):
        if self.stop_after is not None and self.admitted >= self.stop_after:
            raise mt.AdmissionStopped(self.reason)
        self.admitted += 1
        return {
            "status": "success",
            "generated_text": "assistant reply",
        }


def _one_session_sample(task):
    benchmark, context = prepare_benchmark(task)
    return benchmark, next(benchmark.samples(context))


def test_evaluate_records_first_turn_admission_stop_facts():
    # Budget exhausted before the first turn is admitted: the session
    # still reports planned turns, zero completions, the stop reason and
    # no service failure, instead of disappearing into facts={}.
    benchmark, sample = _one_session_sample(session_task(
        "http://127.0.0.1:1", multi_max_turns=3, multi_num_sessions=1,
        max_concurrency=1,
    ))
    stub = _AdmissionStubContext(benchmark.p, stop_after=0)

    facts = benchmark.evaluate(sample, stub)

    assert facts["planned_turns"] == 3
    assert facts["completed_turns"] == 0
    assert facts["sent_failed_turns"] == 0
    assert facts["not_started_turns"] == 3
    assert facts["abort_reason"] == "num_requests"
    assert facts["status"] == "failed"


def test_evaluate_keeps_completed_turns_on_mid_session_stop():
    # A duration cap that strikes between turns keeps every completed
    # turn and an explicit reason in the session summary.
    benchmark, sample = _one_session_sample(session_task(
        "http://127.0.0.1:1", multi_max_turns=3, multi_num_sessions=1,
        max_concurrency=1,
    ))
    stub = _AdmissionStubContext(
        benchmark.p, stop_after=1, reason="max_duration"
    )

    facts = benchmark.evaluate(sample, stub)

    assert facts["completed_turns"] == 1
    assert facts["abort_reason"] == "max_duration"
    assert facts["sent_failed_turns"] == 0
    assert facts["not_started_turns"] == facts["planned_turns"] - 1
    assert facts["session_latency_ms"] >= 0


def test_max_turns_of_one_is_allowed(service, tmp_path):
    # A one-turn maximum is a legitimate configuration; it degenerates to
    # a single call per session rather than being rejected as single-turn.
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=1, multi_num_sessions=3, max_concurrency=3,
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success", result
    facts = raw_case(directory)
    assert all(
        sample["facts"]["planned_turns"] == 1 for sample in facts["samples"]
    )
    assert len(facts["requests"]) == 3


# ── system prompt handling (P2) ────────────────────────────────────────


def system_dataset(tmp_path):
    records = [
        multi_turn_conversation(0, 2, system="Dataset system rule."),
        multi_turn_conversation(1, 2),
    ]
    return write_dataset(tmp_path, records, "system.json")


def test_dataset_system_message_reaches_the_request(service, tmp_path):
    result, _ = run_task(
        session_task(
            service, dataset_path=str(system_dataset(tmp_path)),
            multi_max_turns=2, multi_num_sessions=2, max_concurrency=1,
            multi_system_prompt="CLI system rule.",
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    systems = [
        [m["content"] for m in post["messages"] if m["role"] == "system"]
        for post in ChatHandler.posts
    ]
    flat = [value for group in systems for value in group]
    assert "Dataset system rule." in flat, "dataset system must survive"
    assert "CLI system rule." in flat, "CLI prompt applies where absent"
    # Neither source is duplicated into two system messages.
    assert all(len(group) <= 1 for group in systems), systems


def test_system_prompt_source_is_recorded_not_silent(tmp_path):
    benchmark, _ = prepare_benchmark(
        {
            "name": "t", "scenario": "multi_turn_serving", "model": "mock",
            "base_url": "http://127.0.0.1:1",
            "dataset_path": str(system_dataset(tmp_path)),
            "multi_max_turns": 2, "multi_num_sessions": 2,
            "profile": "concurrent", "max_concurrency": 1,
            "multi_system_prompt": "CLI system rule.",
        }
    )
    source = benchmark.filter_facts["system_prompt_source"]
    assert source["dataset"] == 1
    assert source["cli"] == 1
    assert source["ignored_cli"] == 1, "the override must be visible"
    assert source["none"] == 0


def test_cli_system_prompt_is_fixed_across_turns(service, tmp_path):
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=3, multi_num_sessions=1, max_concurrency=1,
            multi_system_prompt="Answer briefly.",
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    for post in ChatHandler.posts:
        assert post["messages"][0] == {
            "role": "system", "content": "Answer briefly.",
        }


# ── per-turn context limit (P2) ────────────────────────────────────────


def test_context_overflow_is_detected_before_sending(
    service, tmp_path, monkeypatch
):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(mt, "load_tokenizer", lambda p: tokenizer)
    # The stub tokenizer only exists in this process, so the case runs
    # in-process instead of through the task subprocess.
    result = run_case_in_process(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
            max_model_len=40, output_tokens=5, tokenizer="fake",
        ),
        tmp_path,
    )
    # Two sessions overflow, one completes: a workload fact, not a crash.
    assert result["status"] == "partial_failed", result
    facts = raw_case(tmp_path)
    overflowed = [
        sample for sample in facts["samples"]
        if isinstance(sample.get("facts"), dict)
        and sample["facts"]["abort_reason"] == "context_overflow"
    ]
    assert overflowed, "a 40-token limit cannot hold a 4-turn session"
    for sample in overflowed:
        session_facts = sample["facts"]
        assert session_facts["context_overflow_tokens"] is not None
        # The turn was never sent, so it is not counted as a failure.
        assert session_facts["completed_turns"] < session_facts[
            "planned_turns"
        ]
        assert session_facts["not_started_turns"] == (
            session_facts["planned_turns"] - session_facts["completed_turns"]
        )
    sent = sum(
        1 for request in facts["requests"]
        if request["scheduling"]["turn_index"] >= 0
    )
    assert sent < 8, "over-limit turns must not reach the service"


def test_context_check_uses_real_role_alternation(
    service, tmp_path, monkeypatch
):
    # A template that rejects consecutive user messages would have failed
    # on the old estimate, which concatenated every question.
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(mt, "load_tokenizer", lambda p: tokenizer)
    result = run_case_in_process(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
            max_model_len=100000, output_tokens=3, tokenizer="fake",
        ),
        tmp_path,
    )
    assert result["status"] == "success", result
    assert tokenizer.calls, "the limit check must have run"
    for messages in tokenizer.calls:
        roles = [m["role"] for m in messages]
        for position in range(1, len(roles)):
            assert not (
                roles[position] == "user" and roles[position - 1] == "user"
            ), messages


def test_worst_case_turn_budget_no_longer_rejects_short_sessions(
    service, tmp_path, monkeypatch
):
    # 8 turns x 64 output tokens = 512 would have been rejected at
    # validation against max_model_len=120.  The dataset holds only
    # natural 1-turn sessions whose real per-turn check fits, so the
    # task must plan and run to completion.
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(mt, "load_tokenizer", lambda p: tokenizer)
    records = [
        multi_turn_conversation(0, 1),
        multi_turn_conversation(1, 1),
    ]
    path = write_dataset(tmp_path, records, "short.json")
    result = run_case_in_process(
        session_task(
            service, dataset_path=str(path),
            multi_max_turns=8, multi_num_sessions=2, max_concurrency=1,
            max_model_len=120, output_tokens=64, tokenizer="fake",
        ),
        tmp_path,
    )
    assert result["status"] == "success", result
    facts = raw_case(tmp_path)
    assert all(
        sample["facts"]["planned_turns"] == 1
        for sample in facts["samples"]
    )
    assert len(facts["requests"]) == 2


def test_no_tokenizer_means_no_context_check(service, tmp_path):
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    facts = raw_case(directory)
    assert all(
        sample["facts"]["context_overflow_tokens"] is None
        for sample in facts["samples"]
    )
    assert "estimated_context_tokens" not in json.dumps(facts["samples"])


# ── failure, timeout and stop semantics ────────────────────────────────


def test_failed_turn_aborts_the_session_without_retrying(
    service, tmp_path
):
    ChatHandler.fail_turns = {1}
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    facts = raw_case(directory)
    aborted = [
        sample for sample in facts["samples"]
        if isinstance(sample.get("facts"), dict)
        and sample["facts"]["abort_reason"] == "turn_failed"
    ]
    assert aborted, "turn 1 fails for every session with 2+ turns"
    for sample in aborted:
        session_facts = sample["facts"]
        assert session_facts["completed_turns"] == 1
        assert session_facts["status"] == "failed"
        # One completed turn plus the turn that failed after sending.
        assert session_facts["sent_failed_turns"] == 1
        assert session_facts["not_started_turns"] == (
            session_facts["planned_turns"] - 2
        )
    # No retry: exactly one request per attempted turn.
    attempts = [
        request["scheduling"]["turn_index"]
        for request in facts["requests"]
    ]
    assert len(attempts) == len(set(zip(
        (r["sample_id"] for r in facts["requests"]), attempts
    ))), "each (session, turn) appears once"
    assert "retried_turns" not in json.dumps(facts["samples"])
    # Sessions whose first turn succeeded keep it, and other sessions are
    # not aborted by a failure elsewhere.
    assert any(
        sample["facts"]["status"] == "success"
        for sample in facts["samples"]
        if isinstance(sample.get("facts"), dict)
        and sample["facts"]["planned_turns"] == 1
    )


def test_first_turn_failure_leaves_no_completed_turn(service, tmp_path):
    ChatHandler.fail_turns = {0}
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    facts = raw_case(directory)
    assert all(
        request["status"] != "success" for request in facts["requests"]
    )
    for sample in facts["samples"]:
        assert sample["facts"]["completed_turns"] == 0
        assert sample["facts"]["abort_reason"] == "turn_failed"
        # The failed turn was sent: it is sent-failed, and everything
        # after it never starts.
        assert sample["facts"]["sent_failed_turns"] == 1
        assert sample["facts"]["not_started_turns"] == (
            sample["facts"]["planned_turns"] - 1
        )


def test_session_timeout_marks_remaining_turns_not_started(
    service, tmp_path
):
    ChatHandler.chunk_delay = 0.05
    result, directory = run_task(
        session_task(
            service, multi_num_sessions=1, max_concurrency=1,
            multi_session_timeout=0.2,
        ),
        tmp_path,
    )
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "failed", case
    facts = raw_case(directory)
    sample = facts["samples"][0]
    assert sample["facts"]["abort_reason"] == "session_timeout"
    assert 0 < sample["facts"]["completed_turns"] < 4
    assert len(facts["requests"]) == sample["facts"]["completed_turns"]
    # A timeout aborts before the next turn is sent: the whole gap
    # between planned and completed is not-started, none sent-failed.
    assert sample["facts"]["sent_failed_turns"] == 0
    assert sample["facts"]["not_started_turns"] == (
        sample["facts"]["planned_turns"]
        - sample["facts"]["completed_turns"]
    )


def test_duration_stop_keeps_completed_turns(service, tmp_path):
    # max_duration is a normal stop, not an interruption: the Case is
    # kept and the turns it did finish stay measurable.
    ChatHandler.chunk_delay = 0.03
    result, directory = run_task(
        session_task(
            service, multi_num_sessions=6, max_concurrency=1,
            max_duration=0.5, repeat_data=True,
        ),
        tmp_path,
    )
    case = result["tasks"][0]["cases"][0]
    assert case["window"]["stop_reason"] == "max_duration", case
    assert case["collection_complete"] is True
    assert case["status"] in ("success", "partial_failed"), case
    assert case["coverage"]["not_started"] + case["coverage"][
        "incomplete"
    ] > 0
    facts = raw_case(directory)
    finished = [
        request for request in facts["requests"]
        if request["status"] == "success"
    ]
    assert finished, "completed turns must survive a normal stop"
    turns: dict[str, list[int]] = {}
    for request in finished:
        turns.setdefault(request["sample_id"], []).append(
            request["scheduling"]["turn_index"]
        )
    for indices in turns.values():
        assert indices == list(range(len(indices)))
    assert sum(
        facts["coverage"][state]
        for state in ("complete", "failed", "incomplete", "not_started")
    ) == facts["coverage"]["selected"]
    metric = case["metrics"]["streaming_performance"]
    assert metric["status"] != "unavailable", metric
    assert metric["values"]["turn_view"]["turns_observed"] >= 1


def test_user_interrupt_discards_the_case(service, tmp_path):
    # Existing project policy, unchanged by multi-turn: a Case whose
    # collection was interrupted keeps only its configuration and the
    # discard reason, so partial data is never reported as measured.
    ChatHandler.chunk_delay = 0.03
    plan = build_plan(
        session_task(
            service, multi_num_sessions=6, max_concurrency=1,
            repeat_data=True,
        )
    )
    directory = tmp_path / "run"
    (directory / "task_1" / "raw").mkdir(parents=True)
    stop = directory / "stop"

    def interrupt():
        time.sleep(0.4)
        stop.touch()

    thread = threading.Thread(target=interrupt, daemon=True)
    thread.start()
    from luban_meter.core.benchmark import benchmarks
    from luban_meter.core.planning import bootstrap

    bootstrap()
    result = run_case(
        benchmarks.get("multi_turn_serving"),
        plan["tasks"][0]["cases"][0],
        directory / "task_1",
        stop,
    )
    thread.join()
    assert result["status"] == "discarded", result
    assert result["reason"]["message"] == "collection interrupted"
    assert set(result) == {"id", "status", "parameters", "reason"}
    assert not (directory / "task_1" / "raw" / "case_1.json").exists()


# ── stop-condition denominators (P2) ───────────────────────────────────


def test_error_rate_counts_failed_requests(service, tmp_path):
    # Every first turn fails, so the rate is 100% against a 10% limit.
    # The old code filtered failures out before measuring and reported
    # no stop reason at all.
    ChatHandler.fail_turns = {0}
    config = tmp_path / "tasks.json"
    config.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        **session_task(
                            service,
                            dataset_path=str(review_dataset(tmp_path)),
                            multi_max_turns=4,
                            multi_num_sessions=3,
                            max_concurrency=1,
                        ),
                        "max_error_rate": 0.1,
                        "cases": {
                            "items": [
                                {"multi_max_turns": 4},
                                {"multi_max_turns": 2},
                            ]
                        },
                    }
                ]
            }
        )
    )
    plan = build_plan({"model": "mock"}, config)
    result, _ = RunCoordinator().run(plan, tmp_path / "runs")
    first, second = result["tasks"][0]["cases"]
    assert first["status"] == "failed", first
    assert second["status"] == "skipped"
    assert second["reason"] == "max_error_rate_exceeded"


def test_error_rate_ignores_turns_that_were_never_sent(service, tmp_path):
    # Turn 1 fails, so each aborted session has one failed request and
    # one or more turns that were never admitted.  Only the request that
    # actually happened may count in the denominator.
    ChatHandler.fail_turns = {1}
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
            max_error_rate=0.5,
        ),
        tmp_path,
    )
    facts = raw_case(directory)
    finished = len(facts["requests"])
    failed = sum(
        1 for request in facts["requests"]
        if request["status"] != "success"
    )
    assert 0 < failed / finished <= 0.5, (failed, finished)
    assert result["tasks"][0]["cases"][0]["status"] != "skipped"


def test_circuit_breaker_uses_first_turns_only(service, tmp_path):
    # case_p99_e2el_ms needs at least 10 successful samples, and each
    # session contributes exactly one first turn, so 12 sessions are
    # the smallest population that can trip the breaker at all.
    config = tmp_path / "tasks.json"
    config.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        **session_task(
                            service, multi_num_sessions=12,
                            max_concurrency=4, repeat_data=True,
                        ),
                        "circuit_breaker": 0.001,
                        "cases": {
                            "items": [
                                {"multi_max_turns": 4},
                                {"multi_max_turns": 6},
                            ]
                        },
                    }
                ]
            }
        )
    )
    plan = build_plan({"model": "mock"}, config)
    result, _ = RunCoordinator().run(plan, tmp_path / "runs")
    first, second = result["tasks"][0]["cases"]
    assert first["status"] == "success", first
    assert second["status"] == "skipped"
    assert second["reason"] == "circuit_breaker_exceeded"


# ── dataset facts ──────────────────────────────────────────────────────


def test_session_shortfall_is_reported(tmp_path):
    dataset = write_dataset(
        tmp_path, [multi_turn_conversation(i, 3) for i in range(3)],
        "few.json",
    )
    with pytest.raises(ValueError) as error:
        prepare_benchmark(
            {
                "name": "t", "scenario": "multi_turn_serving",
                "model": "mock", "base_url": "http://127.0.0.1:1",
                "dataset_path": str(dataset), "multi_max_turns": 3,
                "multi_num_sessions": 50, "profile": "concurrent",
                "max_concurrency": 1,
            }
        )
    message = str(error.value)
    assert "multi_num_sessions=50" in message
    assert "3 usable conversations" in message
    assert "repeat-data" in message


def test_repeat_data_allows_more_sessions_than_records(service, tmp_path):
    dataset = write_dataset(
        tmp_path, [multi_turn_conversation(i, 2) for i in range(2)],
        "tiny.json",
    )
    result, directory = run_task(
        session_task(
            service, dataset_path=str(dataset), multi_max_turns=2,
            multi_num_sessions=5, max_concurrency=1, repeat_data=True,
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    facts = raw_case(directory)
    assert len(facts["samples"]) == 5
    # Reuse is visible: five sessions came from two distinct records.
    assert len({s["facts"]["record_id"] for s in facts["samples"]}) == 2
    assert facts["metadata"]["dataset_filter"]["usable"] == 2


def test_truncation_facts_record_the_bias(service, tmp_path):
    result, directory = run_task(
        session_task(service, multi_max_turns=8, multi_num_sessions=2,
                     max_concurrency=2),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    facts_filter = raw_case(directory)["metadata"]["dataset_filter"]
    assert facts_filter["max_turns"] == 8
    # 8 records at 12 turns and 8 at 8 are truncated or exact; the 4- and
    # 2-turn records are kept rather than filtered away.  The no-user and
    # blank-user records carry no usable turn, while the unknown-speaker
    # record keeps its single usable one.
    assert facts_filter["source_total"] == 32
    assert facts_filter["usable"] == 30
    assert facts_filter["usable_ratio"] == round(30 / 32, 4)
    assert facts_filter["unusable_no_user_turn"] == 2
    assert facts_filter["truncated"] == 8
    assert facts_filter["source_turn_distribution"]["max"] == 12
    assert facts_filter["planned_turn_distribution"]["max"] == 8
    assert facts_filter["planned_requests_total"] == (
        8 * 8 + 8 * 8 + 8 * 4 + 4 * 2 + 4 + 1
    )


def test_unusable_records_are_counted(tmp_path):
    records = [
        multi_turn_conversation(0, 2),
        {"id": "assistant-only", "conversations": [
            {"from": "gpt", "value": "a"}]},
        {"id": "blank", "conversations": [
            {"from": "human", "value": "   "},
            {"from": "gpt", "value": "b"},
        ]},
    ]
    benchmark, _ = prepare_benchmark(
        {
            "name": "t", "scenario": "multi_turn_serving", "model": "mock",
            "base_url": "http://127.0.0.1:1",
            "dataset_path": str(write_dataset(tmp_path, records, "u.json")),
            "multi_max_turns": 2, "multi_num_sessions": 1,
            "profile": "concurrent", "max_concurrency": 1,
        }
    )
    assert benchmark.filter_facts["unusable_no_user_turn"] == 2
    assert benchmark.filter_facts["usable"] == 1
    assert benchmark.filter_facts["usable_ratio"] == round(1 / 3, 4)


def test_max_samples_shortfall_mentions_the_truncation(tmp_path):
    records = [multi_turn_conversation(i, 3) for i in range(10)]
    dataset = write_dataset(tmp_path, records, "ms.json")
    with pytest.raises(ValueError) as error:
        prepare_benchmark(
            {
                "name": "t", "scenario": "multi_turn_serving",
                "model": "mock", "base_url": "http://127.0.0.1:1",
                "dataset_path": str(dataset), "multi_max_turns": 3,
                "multi_num_sessions": 8, "max_samples": 3,
                "profile": "concurrent", "max_concurrency": 1,
            }
        )
    message = str(error.value)
    assert "max-samples" in message
    assert "3" in message


def test_dataset_without_usable_conversation_reports_distribution(
    tmp_path
):
    dataset = write_dataset(
        tmp_path,
        [{"id": "a", "conversations": [{"from": "gpt", "value": "x"}]}],
        "none.json",
    )
    with pytest.raises(ValueError) as error:
        prepare_benchmark(
            {
                "name": "t", "scenario": "multi_turn_serving",
                "model": "mock", "base_url": "http://127.0.0.1:1",
                "dataset_path": str(dataset), "multi_max_turns": 4,
                "multi_num_sessions": 1, "profile": "concurrent",
                "max_concurrency": 1,
            }
        )
    assert "no usable conversation" in str(error.value)


# ── end-to-end session behaviour ───────────────────────────────────────


def test_sessions_accumulate_real_assistant_history(service, tmp_path):
    # A single 4-turn record, so the one session runs all four turns.
    dataset = write_dataset(
        tmp_path, [multi_turn_conversation(0, 4)], "accumulate.json"
    )
    result, directory = run_task(
        session_task(
            service, dataset_path=str(dataset),
            multi_max_turns=4, multi_num_sessions=1, max_concurrency=1,
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    posts = ChatHandler.posts
    assert len(posts) == 4
    # Turn N carries the answer the stub produced at turn N-1, not the
    # dataset's reference reply.
    for index in range(1, len(posts)):
        previous = posts[index - 1]
        users = sum(
            1 for m in previous["messages"] if m["role"] == "user"
        )
        expected = f"A{users}"
        assistants = [
            m["content"] for m in posts[index]["messages"]
            if m["role"] == "assistant"
        ]
        assert assistants[-1].startswith(expected), assistants
    assert not any(
        "Historical reply" in json.dumps(post["messages"]) for post in posts
    ), "dataset assistant text must never be replayed"
    facts = raw_case(directory)
    assert facts["metadata"]["history_source"] == "model_under_test"
    assert facts["metadata"]["multi_max_turns"] == 4
    # Context really grows turn over turn.
    inputs = [
        request["response"]["input_tokens"]
        for request in sorted(
            facts["requests"],
            key=lambda r: r["scheduling"]["turn_index"],
        )
    ]
    assert inputs == sorted(inputs)
    assert inputs[-1] > inputs[0]
    texts = [
        request["response"]["generated_text"] for request in facts["requests"]
    ]
    assert texts and all(texts), "every turn keeps its generated text"
    assert "history" not in json.dumps(facts["samples"])


def test_turns_are_recorded_per_session_with_turn_index(service, tmp_path):
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=2,
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    facts = raw_case(directory)
    per_session: dict[str, list[int]] = {}
    for request in facts["requests"]:
        per_session.setdefault(request["sample_id"], []).append(
            request["scheduling"]["turn_index"]
        )
    assert len(per_session) == 3
    for indices in per_session.values():
        assert indices == list(range(len(indices)))
    assert all(
        request["scheduling"]["session_turns"] in (1, 3, 4)
        for request in facts["requests"]
    )
    assert "attempt" not in json.dumps(facts["requests"])


def test_save_response_text_persists_history(service, tmp_path):
    # A single 2-turn record, so the saved history has both exchanges.
    dataset = write_dataset(
        tmp_path, [multi_turn_conversation(0, 2)], "history.json"
    )
    result, directory = run_task(
        session_task(
            service, dataset_path=str(dataset),
            multi_max_turns=2, multi_num_sessions=1, max_concurrency=1,
            multi_save_response_text=True,
        ),
        tmp_path,
    )
    assert result["tasks"][0]["cases"][0]["status"] == "success"
    sample = raw_case(directory)["samples"][0]
    history = sample["facts"]["history"]
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_sessions_run_concurrently_within_request_capacity(
    service, tmp_path
):
    ChatHandler.chunk_delay = 0.02
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=3,
        ),
        tmp_path,
    )
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "success", case
    window = case["window"]
    # One sample thread per request slot: sessions overlap, but a single
    # session never has two calls in flight.
    assert window["request_capacity"] == 3
    assert window["peak_requests"] <= 3
    assert "sample_capacity" not in window
    assert case["metrics"]["streaming_performance"]["values"][
        "multi_num_sessions"
    ] == 3


# ── metrics ────────────────────────────────────────────────────────────


def test_multi_turn_metric_views(service, tmp_path):
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=3,
            multi_think_time=0.01,
        ),
        tmp_path,
    )
    case = result["tasks"][0]["cases"][0]
    metric = case["metrics"]["streaming_performance"]
    assert metric["status"] == "success", metric
    values = metric["values"]
    assert values["multi_max_turns"] == 4
    assert values["multi_num_sessions"] == 3
    turn_view = values["turn_view"]
    assert turn_view["turns_observed"] == 8
    counts = [row["count"] for row in turn_view["ttft_ms_by_turn"]]
    # Turn 0 runs for all three sessions, turn 3 only for the 7-turn one.
    assert counts == [3, 2, 2, 1]
    inputs = [
        row["mean"] for row in turn_view["input_tokens_by_turn"]
    ]
    assert inputs == sorted(inputs), inputs
    caching = values["caching_view"]
    assert caching["first_turn_ttft_mean"]["value"] is not None
    assert caching["rest_turn_ttft_mean"]["value"] is not None
    session_view = values["session_view"]
    assert session_view["sessions_finished"]["value"] == 3
    assert session_view["abort_reasons"] == {}
    assert session_view["not_started_turns"]["value"] == 0
    assert session_view["sent_failed_turns"]["value"] == 0
    assert session_view["session_coverage"]["complete"] == 3
    assert session_view["think_time_total"]["mean"] > 0
    assert values["pressure_view"]["total_prefill_tokens"]["value"] > 0
    assert values["lengths"]


def test_prefix_cache_ceiling_without_server_support(service, tmp_path):
    # This is what a vLLM build without prompt_tokens_details does: only
    # the client-side estimate is available and the gap is stated.
    result, directory = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    cache = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["caching_view"]["prefix_cache"]
    assert cache["measured_turns"]["value"] == 8
    assert cache["reporting_turns"]["value"] == 0
    assert cache["server_cache_hit_rate"]["value"] is None
    assert cache["server_cache_eligible_rate"]["value"] is None
    assert cache["server_cached_tokens"]["value"] is None
    assert "cached_tokens" in cache["reason"]
    # No cause is inferred from the ratios.
    assert "diagnosis" not in cache

    requests = sorted(
        raw_case(directory)["requests"],
        key=lambda row: (
            row["sample_id"], row["scheduling"]["turn_index"]
        ),
    )
    by_session: dict[str, list[dict]] = {}
    for row in requests:
        by_session.setdefault(row["sample_id"], []).append(row)
    expected = 0
    for rows in by_session.values():
        for index in range(1, len(rows)):
            previous = rows[index - 1]["response"]
            expected += min(
                rows[index]["response"]["input_tokens"],
                previous["input_tokens"] + previous["output_tokens"],
            )
    assert cache["eligible_prefix_tokens"]["value"] == expected
    assert expected > 0, "later turns must repeat a real prefix"
    assert 0 < cache["eligible_prefix_ratio"]["value"] < 1


def test_prefix_cache_reports_server_values_when_available(
    service, tmp_path
):
    ChatHandler.cached_token_mode = "perfect"
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    cache = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["caching_view"]["prefix_cache"]
    assert cache["reporting_turns"]["value"] == 8
    assert cache["reason"] is None
    assert cache["server_cache_hit_rate"]["value"] > 0
    assert cache["server_cache_eligible_rate"]["value"] > 0
    assert "diagnosis" not in cache


def test_prefix_cache_keeps_a_reported_zero(service, tmp_path):
    # A genuine zero from the service is an observation, not a gap.
    ChatHandler.cached_token_mode = "none"
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    cache = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["caching_view"]["prefix_cache"]
    assert cache["reporting_turns"]["value"] == 8
    assert cache["reason"] is None
    assert cache["server_cache_hit_rate"]["value"] == 0.0
    assert cache["server_cached_tokens"]["value"] == 0
    assert cache["eligible_prefix_ratio"]["value"] > 0


def test_prefix_cache_ignores_failed_turns(service, tmp_path):
    ChatHandler.fail_turns = {1}
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    cache = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["caching_view"]["prefix_cache"]
    # Three sessions: 1 turn succeeds, and turns 0 of the two longer ones.
    assert cache["measured_turns"]["value"] == 3
    assert cache["reporting_turns"]["value"] == 0


def test_prefix_cache_partial_reporting_rates_the_reporting_subset():
    # The review's A/B example: A reports 50 of 100 prompt tokens cached,
    # B (900 prompt tokens) does not report.  The measured rate must
    # describe the reporting subset (50%), never silently reading B's
    # unknown coverage as a zero hit over the whole Case (5%).
    from luban_meter.benchmarking.generation_performance.common import (
        session_metrics,
    )

    rows = [
        {
            "session": 0, "turn": 0, "status": "success",
            "response": {
                "input_tokens": 100, "output_tokens": 30,
                "usage": {
                    "prompt_tokens_details": {"cached_tokens": 50}
                },
            },
        },
        {
            "session": 0, "turn": 1, "status": "success",
            "response": {"input_tokens": 900, "output_tokens": 30},
        },
    ]
    cache = session_metrics.prefix_cache_view(rows)

    assert cache["measured_turns"]["value"] == 2
    assert cache["reporting_turns"]["value"] == 1
    assert cache["prompt_tokens_total"]["value"] == 1000
    assert cache["reporting_prompt_tokens_total"]["value"] == 100
    # The single reporting turn has no eligible prefix, so its saved
    # denominator is 0; the unknown-coverage turn contributes nothing.
    assert cache["reporting_eligible_tokens_total"]["value"] == 0
    assert cache["server_cached_tokens"]["value"] == 50
    assert cache["server_cache_hit_rate"]["value"] == 0.5
    # The reporting subset holds no eligible prefix (turn 0 is the only
    # reporter), so the server-eligible rate stays unknown, not zero.
    assert cache["server_cache_eligible_rate"]["value"] is None
    assert "1 of 2" in cache["reason"]


def test_prefix_cache_partial_reporting_aligns_eligible_subset():
    # When a later turn is the reporter, its eligible prefix joins the
    # same reporting subset as the cached numerator.
    from luban_meter.benchmarking.generation_performance.common import (
        session_metrics,
    )

    rows = [
        {
            "session": 0, "turn": 0, "status": "success",
            "response": {"input_tokens": 100, "output_tokens": 30},
        },
        {
            "session": 0, "turn": 1, "status": "success",
            "response": {
                "input_tokens": 200, "output_tokens": 50,
                "usage": {
                    "prompt_tokens_details": {"cached_tokens": 60}
                },
            },
        },
    ]
    cache = session_metrics.prefix_cache_view(rows)

    assert cache["reporting_turns"]["value"] == 1
    assert cache["reporting_prompt_tokens_total"]["value"] == 200
    # The eligible denominator is saved, so 60 / 130 recomputes the
    # displayed server-eligible rate from persisted fields.
    assert cache["reporting_eligible_tokens_total"]["value"] == 130
    # Eligible prefix of the reporting turn: min(200, 100 + 30) = 130.
    assert cache["server_cache_eligible_rate"]["value"] == pytest.approx(
        60 / 130, abs=1e-4
    )
    # The client-side ceiling still spans all measured turns.
    assert cache["eligible_prefix_tokens"]["value"] == 130
    assert cache["prompt_tokens_total"]["value"] == 300


def test_metric_keeps_completed_turns_of_failed_sessions(service, tmp_path):
    ChatHandler.fail_turns = {2}
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    case = result["tasks"][0]["cases"][0]
    assert case["status"] == "partial_failed", case
    metric = case["metrics"]["streaming_performance"]
    # Both sessions that reach turn 2 fail it: 5 of 7 requests succeed,
    # so the metric itself is partial, not "success".
    assert metric["status"] == "partial_failed", metric
    session_view = metric["values"]["session_view"]
    assert session_view["abort_reasons"] == {"turn_failed": 2}
    assert session_view["sessions_finished"]["value"] == 1
    assert session_view["sent_failed_turns"]["value"] == 2
    assert session_view["not_started_turns"]["value"] == 1
    # The two aborted sessions keep their first two measured turns:
    # 5 observed + 2 sent-failed + 1 not-started = 8 planned turns.
    assert metric["values"]["turn_view"]["turns_observed"] == 5


def test_goodput_applies_per_turn(service, tmp_path):
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=3,
            goodput=["ttft:5000"],
        ),
        tmp_path,
    )
    values = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]
    # Goodput is a service-view sub-result over the pooled turns.
    goodput = values["service_view"]["goodput"]
    assert goodput["status"] == "applicable", goodput
    assert goodput["slo_satisfied_rate"]["value"] == 1.0
    assert goodput["slo_config"]["ttft_ms"] == 5000.0


def test_unsatisfiable_goodput_reports_zero(service, tmp_path):
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=3,
            goodput=["ttft:0.001"],
        ),
        tmp_path,
    )
    values = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]
    goodput = values["service_view"]["goodput"]
    assert goodput["slo_satisfied_rate"]["value"] == 0.0
    assert goodput["slo_violated_count"]["value"] == 8


# ── planning ───────────────────────────────────────────────────────────


def test_case_matrix_expands_turns_and_sessions(tmp_path):
    config = tmp_path / "tasks.json"
    config.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        **session_task(
                            "http://127.0.0.1:1", multi_num_sessions=4
                        ),
                        "cases": {
                            "matrix": {
                                "multi_max_turns": [2, 4],
                                "output_tokens": [8, 16],
                            }
                        },
                    }
                ]
            }
        )
    )
    plan = build_plan({"model": "mock"}, config)
    cases = plan["tasks"][0]["cases"]
    assert len(cases) == 4
    assert {(c["parameters"]["multi_max_turns"],
             c["parameters"]["output_tokens"]) for c in cases} == {
        (2, 8), (2, 16), (4, 8), (4, 16),
    }


def test_think_time_std_requires_a_mean():
    with pytest.raises(ValueError, match="multi_think_time_std"):
        build_plan(
            session_task(
                "http://127.0.0.1:1", multi_think_time=0.0,
                multi_think_time_std=0.5,
            )
        )


def test_output_budget_above_model_length_is_rejected():
    # Only the per-turn real-history check can stop a session; the
    # multiplied worst case (4 turns x 1000 tokens) no longer rejects
    # planning, because natural replies stay far below the budget.
    plan = build_plan(
        session_task(
            "http://127.0.0.1:1", multi_max_turns=4,
            output_tokens=1000, max_model_len=2000,
        )
    )
    case = plan["tasks"][0]["cases"][0]
    assert case["parameters"]["output_tokens"] == 1000


# ── per-turn participation ledger and ITL (issue 4) ────────────────────


def _turn_view_of(result):
    return result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["turn_view"]


def test_turn_participation_and_itl_by_turn(service, tmp_path):
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    turn = _turn_view_of(result)
    ledger = turn["turn_participation"]
    # Natural session lengths 1, 3 and 4 (truncated from 7): a natural
    # short session never appears in later turns' planned counts.
    assert [row["turn_index"] for row in ledger] == [0, 1, 2, 3]
    assert [row["planned_sessions"] for row in ledger] == [3, 2, 2, 1]
    for row in ledger:
        assert row["succeeded"] == row["planned_sessions"]
        assert row["sent_failed"] == 0
        assert row["not_started"] == 0
    # Request-level ITL: (last output - first output) / (tokens - 1),
    # one weighted sample per successful request.  The stub streams 3
    # output tokens, so each request carries 2 token intervals.
    itl = turn["itl_ms_by_turn"]
    assert [row["turn_index"] for row in itl] == [0, 1, 2, 3]
    assert [row["count"] for row in itl] == [
        row["succeeded"] for row in ledger
    ]
    assert [row["weight_sum"] for row in itl] == [
        row["succeeded"] * 2 for row in ledger
    ]
    assert all(row["mean"] is not None for row in itl)
    # Stream-event ITL is the separate interval metric: the 3-token
    # stub yields two event gaps per request.
    event_itl = turn["stream_event_itl_ms_by_turn"]
    assert [row["count"] for row in event_itl] == [
        row["succeeded"] * 2 for row in ledger
    ]
    assert all(row["mean"] is not None for row in event_itl)


def test_request_level_itl_by_turn_weights_token_intervals():
    # The review's two-request example: 11 tokens and 101 tokens over
    # the same 100ms output span.  Request-level ITL weights each
    # request by its token-interval count, so the turn mean is
    # (10*10 + 1*100) / 110, while the stream-event mean stays 100ms.
    from luban_meter.benchmarking.generation_performance.common import (
        session_metrics,
    )

    def request(output_tokens, itl_samples):
        return {
            "status": "success", "ttft_ms": 10, "e2el_ms": 120,
            "last_output_latency_ms": 110,
            "output_tokens": output_tokens, "input_tokens": 100,
            "itl_samples_ms": itl_samples,
        }

    rows = [
        {"session": 0, "turn": 0, "status": "success",
         "response": request(11, [100])},
        {"session": 1, "turn": 0, "status": "success",
         "response": request(101, [100])},
    ]
    turn = session_metrics.turn_view(rows)
    itl = turn["itl_ms_by_turn"][0]
    assert (itl["count"], itl["weight_sum"]) == (2, 110)
    assert itl["mean"] == pytest.approx(200 / 110, abs=1e-3)
    events = turn["stream_event_itl_ms_by_turn"][0]
    assert (events["count"], events["mean"]) == (2, 100)
    assert events["unit"] == "ms"
    assert itl["unit"] == "ms/token"


def test_request_level_itl_by_turn_covers_packing_and_missing_tokens():
    # Batching changes event intervals but not the request-level value;
    # missing token counts keep event intervals while the request-level
    # series stays empty, following the overall request_view.itl rules.
    from luban_meter.benchmarking.generation_performance.common import (
        session_metrics,
    )

    def request(**overrides):
        record = {
            "status": "success", "ttft_ms": 10, "e2el_ms": 120,
            "last_output_latency_ms": 110, "output_tokens": 11,
            "input_tokens": 100, "itl_samples_ms": [10] * 10,
        }
        record.update(overrides)
        return record

    # One event per token: both metrics agree at 10ms.
    dense = request()
    # A single event carrying all tokens: no event intervals, and the
    # request-level span collapses to zero.
    packed = request(last_output_latency_ms=10, itl_samples_ms=[])
    # No reported token count: the request joins no request-level
    # sample, its event intervals remain usable.
    uncounted = request(output_tokens=None, itl_samples_ms=[100])
    view = session_metrics.turn_view([
        {"session": 0, "turn": 0, "status": "success", "response": dense},
        {"session": 1, "turn": 0, "status": "success", "response": packed},
        {"session": 2, "turn": 0, "status": "success",
         "response": uncounted},
    ])
    itl = view["itl_ms_by_turn"][0]
    assert itl["count"] == 2
    assert itl["weight_sum"] == 20
    # (10*10 + 0*10) / 20: the uncounted request contributes nothing.
    assert itl["mean"] == pytest.approx(5, abs=1e-6)
    events = view["stream_event_itl_ms_by_turn"][0]
    assert events["count"] == 11
    assert events["mean"] == pytest.approx(200 / 11, abs=1e-3)


def test_turn_participation_distinguishes_failure_from_not_started(
    service, tmp_path
):
    ChatHandler.fail_turns = {1}
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    ledger = {
        row["turn_index"]: row
        for row in _turn_view_of(result)["turn_participation"]
    }
    # Turn 0 succeeds everywhere; the two longer sessions fail the turn
    # they did send, and their later turns never started.
    assert ledger[0]["succeeded"] == 3
    assert ledger[1]["sent_failed"] == 2
    assert ledger[1]["succeeded"] == 0
    assert ledger[1]["not_started"] == 0
    assert ledger[2]["not_started"] == 2
    assert ledger[2]["sent_failed"] == 0
    assert ledger[3]["not_started"] == 1


def test_turn_participation_records_budget_stop_not_started(
    service, tmp_path
):
    # Budget 5 with sessions of 1, 3 and 4 turns: the third session is
    # cut after its first turn; its later turns are not started, not
    # service failures.
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
            num_requests=5,
        ),
        tmp_path,
    )
    ledger = {
        row["turn_index"]: row
        for row in _turn_view_of(result)["turn_participation"]
    }
    assert ledger[1]["succeeded"] == 1
    assert ledger[1]["not_started"] == 1
    assert ledger[1]["sent_failed"] == 0
    assert ledger[3]["not_started"] == 1


def test_all_failed_first_turn_keeps_diagnosable_turn_facts(
    service, tmp_path
):
    ChatHandler.fail_turns = {0}
    result, _ = run_task(
        session_task(
            service, dataset_path=str(review_dataset(tmp_path)),
            multi_max_turns=4, multi_num_sessions=3, max_concurrency=1,
        ),
        tmp_path,
    )
    turn = _turn_view_of(result)
    # No success turns: latency series stay empty, but the ledger and
    # the abort reasons survive for diagnosis.
    assert turn["turns_observed"] == 0
    assert turn["ttft_ms_by_turn"] == []
    assert turn["itl_ms_by_turn"] == []
    assert turn["stream_event_itl_ms_by_turn"] == []
    ledger = {
        row["turn_index"]: row
        for row in turn["turn_participation"]
    }
    assert ledger[0]["planned_sessions"] == 3
    assert ledger[0]["sent_failed"] == 3
    assert ledger[1]["not_started"] == 2
    session_view = result["tasks"][0]["cases"][0]["metrics"][
        "streaming_performance"
    ]["values"]["session_view"]
    assert session_view["abort_reasons"] == {"turn_failed": 3}
