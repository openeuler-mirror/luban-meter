"""Logging scopes must preserve callers' state and isolate successive runs."""

import json
import logging
import subprocess
import sys
from io import StringIO

import pytest

from luban_meter.log import (
    configure_logging,
    get_log_file_path,
    get_logger,
    log_context,
    resolve_log_dir,
    resolve_log_level,
)


def test_logging_scope_redacts_stacks_and_restores_state(tmp_path):
    package = logging.getLogger("luban_meter")
    logger = logging.getLogger("luban_meter.test")
    handlers = package.handlers[:]
    level, propagate = package.level, package.propagate
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    with pytest.raises(ValueError):
        with configure_logging(
            first.parent, filename=first.name, task="first",
            secrets=["private-key"],
        ):
            assert get_log_file_path() == first
            with log_context(case="case_1"):
                logger.info("First run")
                raise ValueError("failure includes private-key")
    assert package.handlers == handlers
    assert get_log_file_path() is None
    assert (package.level, package.propagate) == (level, propagate)
    with configure_logging(second.parent, filename=second.name, task="second"):
        logger.info("Second run")
    first_text, second_text = first.read_text(), second.read_text()
    assert "Traceback (most recent call last)" in first_text
    assert "private-key" not in first_text
    assert "failure includes [redacted]" in first_text
    assert "task=first case=case_1" in first_text
    assert "Second run" not in first_text
    assert "First run" not in second_text
    assert "task=second case=-" in second_text
    assert package.handlers == handlers


def test_log_context_is_reset_when_exception_leaves_scope(tmp_path):
    logger = logging.getLogger("luban_meter.test")
    with configure_logging(tmp_path) as path:
        with pytest.raises(RuntimeError):
            with log_context(task="task_1", case="case_1"):
                raise RuntimeError("expected")
        logger.info("Outside the Case")
    assert "task=- case=-" in path.read_text()


def test_worker_launch_failure_is_logged_by_parent(tmp_path, monkeypatch):
    from luban_meter.core.planning import build_plan
    from luban_meter.core.run_coordinator import RunCoordinator

    def fail_spawn(*args, **kwargs):
        raise OSError("cannot spawn worker with private-key")

    monkeypatch.setattr("subprocess.Popen", fail_spawn)
    plan = build_plan({
        "scenario": "online_serving", "model": "mock",
        "api_key": "private-key",
    })
    result, directory = RunCoordinator().run(plan, tmp_path)
    assert result["status"] == "failed"
    text = next((directory / "log").glob("luban-meter-*.log")).read_text()
    assert "Task execution failed" in text
    assert "Traceback (most recent call last)" in text
    assert "cannot spawn worker with [redacted]" in text
    assert "private-key" not in text
    assert "Task cleanup finished" in text


@pytest.mark.parametrize("level,expected", [
    ("debug", logging.DEBUG), ("info", logging.INFO),
    ("warn", logging.WARNING), ("error", logging.ERROR),
    (logging.DEBUG, logging.DEBUG),
])
def test_levels_filter_file_and_console_identically(tmp_path, level, expected):
    console = StringIO()
    logger = get_logger("luban_meter.test_levels")
    assert resolve_log_level(level) == expected
    with configure_logging(tmp_path, level=level, stream=console) as path:
        for value in (logging.DEBUG, logging.INFO,
                      logging.WARNING, logging.ERROR):
            logger.log(value, "message-%s", value)
    text = path.read_text()
    assert console.getvalue() == text
    for value in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
        assert (f"message-{value}" in text) is (value >= expected)
    assert "luban_meter.test_levels" in text
    assert "test_levels_filter_file_and_console_identically:" in text


@pytest.mark.parametrize("level", ["debug", "info", "warn", "error"])
def test_file_only_details_keep_level_redaction_and_context(tmp_path, level):
    console = StringIO()
    cleared = []
    with configure_logging(
        tmp_path, level=level, stream=console, secrets=["private-key"],
        before_emit=lambda: cleared.append(True),
    ) as path:
        logger = get_logger("luban_meter.details")
        with log_context(task="task_1", case="case_1", phase="prepare"):
            logger.info("Details: private-key", extra={"file_only": True})
        assert cleared == []
        logger.error("Brief failure")
    text = path.read_text()
    assert ("Details: [redacted]" in text) is (level in ("debug", "info"))
    if level in ("debug", "info"):
        assert "task=task_1 case=case_1 phase=prepare" in text
        assert " | INFO | " in text
    assert "Details:" not in console.getvalue()
    assert "private-key" not in text + console.getvalue()
    assert "Brief failure" in text and "Brief failure" in console.getvalue()
    assert cleared == [True]


@pytest.mark.parametrize("level", ["invalid", "", True, 17, None])
def test_invalid_levels_do_not_create_files(tmp_path, level):
    target = tmp_path / "not-created"
    with pytest.raises(ValueError, match="log level"):
        with configure_logging(target, level=level):
            pass
    assert not target.exists()


def test_directory_precedence(tmp_path, monkeypatch):
    explicit, env, fallback = (tmp_path / name for name in (
        "explicit", "environment", "fallback",
    ))
    monkeypatch.setenv("LUBAN_LOG_DIR", str(env))
    assert resolve_log_dir(explicit, fallback=fallback) == explicit
    assert resolve_log_dir(fallback=fallback) == env
    monkeypatch.setenv("LUBAN_LOG_DIR", " ")
    assert resolve_log_dir(fallback=fallback) == fallback
    assert str(resolve_log_dir()) == "log"


def test_external_handlers_survive_and_console_clears_before_emit(tmp_path):
    root = logging.getLogger()
    root_handlers = root.handlers[:]
    console = StringIO()

    def clear():
        console.write("cleared\n")

    with configure_logging(tmp_path, stream=console,
                           before_emit=clear) as path:
        get_logger("luban_meter.test").info("message")
        assert root.handlers == root_handlers
    assert root.handlers == root_handlers
    assert console.getvalue().startswith("cleared\n")
    assert "cleared" not in path.read_text()


def test_plan_calls_use_distinct_logs_and_sanitize_parameters(tmp_path):
    from luban_meter.cli import main

    argv = [
        "plan", "--log-dir", str(tmp_path), "--scenario", "online_serving",
        "--model", "fake", "--api-key", "internal-key",
        "--request-params", '{"custom":{"authorization":"nested-key"}}',
    ]
    assert main(argv) == 0
    assert main(argv) == 0
    logs = list(tmp_path.glob("luban-meter-cli-*.log"))
    assert len(logs) == 2
    for path in logs:
        text = path.read_text()
        assert "Parsing command" in text and "Resolved plan" in text
        assert "max_concurrency" in text and "[redacted]" in text
        assert "internal-key" not in text and "nested-key" not in text


def test_worker_tracebacks_redact_nested_request_credentials(tmp_path):
    from luban_meter.core.planning import build_plan
    from luban_meter.log import task_secrets

    plan = build_plan({
        "scenario": "online_serving", "model": "mock",
        "request_params": {"extension": {"headers": {"X-Test": "nested-key"}}},
    })
    with configure_logging(
        tmp_path, secrets=task_secrets(plan["tasks"]),
    ) as log_file:
        try:
            raise ValueError("transport rejected nested-key")
        except ValueError:
            get_logger("luban_meter.test").exception("Failed request")
    assert "transport rejected [redacted]" in log_file.read_text()
    assert "nested-key" not in log_file.read_text()


@pytest.mark.parametrize("disabled", [False, True])
def test_level_colors_are_terminal_only(tmp_path, monkeypatch, disabled):
    class Terminal(StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    if disabled:
        monkeypatch.setenv("NO_COLOR", "1")
    console = Terminal()
    levels = {"DEBUG": "36", "INFO": "32", "WARNING": "38;5;208",
              "ERROR": "31"}
    with configure_logging(tmp_path, level="debug", stream=console) as path:
        for level in levels:
            get_logger("luban_meter.test").log(getattr(logging, level), level)
    plain = path.read_text()
    expected = plain
    for level, color in levels.items():
        if not disabled:
            expected = expected.replace(
                f" | {level} | ", f" | \033[{color}m{level}\033[0m | ",
            )
    assert "\033" not in plain
    assert console.getvalue() == expected


def test_multi_turn_prompt_is_redacted_in_cli_and_case_logs(tmp_path):
    from luban_meter.log import log_command, log_parameters

    prompt = "private conversation instructions"
    with configure_logging(tmp_path) as path:
        logger = get_logger("luban_meter.test")
        logger.info("Command: %s", log_command(["--multi-system-prompt",
                                                prompt]))
        logger.info("Case: %s",
                    log_parameters({"multi_system_prompt": prompt}))
        logger.error("Failure: %s", prompt)
    text = path.read_text()
    assert prompt not in text
    assert text.count("[redacted]") == 3


def test_multiline_command_timeout_redacts_original_and_repr(tmp_path):
    from luban_meter.core.errors import ExecutionError
    from luban_meter.core.run_contracts import CommandSpec
    from luban_meter.runtime.local_command import LocalCommandRunner

    marker = "SYNTHETIC_PRIVATE_TIMEOUT"
    program = f"import time\nprivate = {marker!r}\ntime.sleep(5)"
    console = StringIO()
    with configure_logging(tmp_path, stream=console) as path:
        with pytest.raises(
            ExecutionError, match="command timed out",
        ) as caught:
            LocalCommandRunner().run(CommandSpec(
                argv=(sys.executable, "-c", program), timeout=0.05,
            ))
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    assert caught.value.__cause__.cmd == [sys.executable, "-c", program]
    for text in (path.read_text(), console.getvalue()):
        assert "Command timed out" in text
        assert "Traceback (most recent call last)" in text
        assert "TimeoutExpired" in text
        assert "[redacted]" in text
        assert marker not in text


@pytest.mark.parametrize("secret", [
    "SYNTHETIC_KEY\nwith\ttab",
    "SYNTHETIC_KEY'\"\\中文\n",
])
def test_secret_escaped_forms_are_redacted_in_errors(tmp_path, secret):
    from luban_meter.log import add_log_secrets, redact_text

    console = StringIO()
    with configure_logging(tmp_path, stream=console) as path:
        add_log_secrets([secret])
        logger = get_logger("luban_meter.test")
        for representation in (secret, repr(secret), ascii(secret)):
            logger.error("Failure value=%s", representation)
            assert "SYNTHETIC_KEY" not in redact_text(representation)
    for text in (path.read_text(), console.getvalue()):
        assert "SYNTHETIC_KEY" not in text
        assert text.count("[redacted]") == 3


def test_worker_stderr_read_failure_still_closes_pipe(tmp_path, monkeypatch):
    import codecs
    import os

    from luban_meter.core.run_coordinator import RunCoordinator

    reader, writer = os.pipe()
    stream = os.fdopen(reader, "rb", buffering=0)
    os.close(writer)
    coordinator = RunCoordinator()
    coordinator.worker_stderr = stream
    coordinator.stderr_decoder = codecs.getincrementaldecoder("utf-8")()

    def fail_read(*args):
        raise OSError("fixture stderr read failed")

    monkeypatch.setattr(os, "read", fail_read)
    with configure_logging(tmp_path) as path:
        coordinator._drain_worker_stderr(final=True)
    assert stream.closed
    assert coordinator.worker_stderr is None
    text = path.read_text()
    assert "Worker stderr forwarding failed" in text
    assert "OSError: fixture stderr read failed" in text


@pytest.mark.parametrize("level", ["debug", "info", "warn", "error"])
@pytest.mark.parametrize("arguments,marker", [
    (["plan", "--scenario", "online_serving", "--model", "mock"], None),
    (["benchmarks", "list"], "generation_performance\tonline_serving"),
    (["metrics", "list"], "streaming_performance\t"),
])
def test_cli_results_stay_on_stdout_at_all_log_levels(
    tmp_path, capsys, level, arguments, marker,
):
    from luban_meter.cli import main

    assert main([*arguments, "--log-dir", str(tmp_path),
                 "--log-level", level]) == 0
    captured = capsys.readouterr()
    if marker is None:
        assert json.loads(captured.out)["case_count"] == 1
        assert "Resolved plan:" not in captured.err
    else:
        assert marker in captured.out
    assert " | INFO | " not in captured.out
    path, = tmp_path.glob("*.log")
    text = path.read_text()
    if marker is None:
        assert ("Resolved plan:" in text) is (level in ("debug", "info"))
    else:
        assert text == captured.err


def test_dataset_preparation_logs_without_copying_body_to_log(
    tmp_path, monkeypatch, capsys,
):
    from luban_meter.benchmarking.model_service_quality.scripts import (
        prepare_wikitext,
    )

    source = tmp_path / "source"
    source.mkdir()
    body = "Private dataset paragraph."
    (source / "wiki.valid.raw").write_text(body, encoding="utf-8")
    output = tmp_path / "output"
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LUBAN_LOG_DIR", str(log_dir))
    monkeypatch.setattr(sys, "argv", [
        "prepare_wikitext", "--source", str(source), "--out", str(output),
    ])
    prepare_wikitext.main()
    captured = capsys.readouterr()
    assert captured.out == ""
    path, = log_dir.glob("*.log")
    text = path.read_text()
    assert text == captured.err
    assert " | INFO | " in text and " | WARNING | " in text
    assert str(output / "val.jsonl") in text
    assert "wiki.test.raw" in text
    assert body not in text
    assert json.loads((output / "val.jsonl").read_text())["text"] == body
