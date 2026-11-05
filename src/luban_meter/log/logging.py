"""Central logging configuration, formatting and execution context."""

from __future__ import annotations

import copy
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from functools import wraps
from pathlib import Path

from luban_meter.utils.run_id import create_run_id

DEFAULT_LOG_DIR = "log"
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(funcName)s:%(lineno)d | %(levelname)s | "
    "pid=%(process)d task=%(task)s case=%(case)s phase=%(phase)s "
    "sample=%(sample)s request=%(request)s | %(message)s"
)
LOG_LEVEL_NAMES = ("debug", "info", "warn", "error")
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}
_context = ContextVar("log_context", default={})
_log_file = ContextVar("log_file", default=None)
_logger = logging.getLogger("luban_meter")
_logger.addHandler(logging.NullHandler())


def get_logger(name):
    """Get a module logger; configuration belongs to the process entry."""
    return logging.getLogger(name)


def get_log_file_path():
    """Return the file opened by the current logging scope, if any."""
    return _log_file.get()


def resolve_log_level(level):
    if isinstance(level, int) and not isinstance(level, bool):
        if level in _LEVELS.values():
            return level
    elif isinstance(level, str) and level.strip().lower() in _LEVELS:
        return _LEVELS[level.strip().lower()]
    raise ValueError(f"log level must be one of: {', '.join(LOG_LEVEL_NAMES)}")


def resolve_log_dir(explicit=None, *, fallback=None):
    """Resolve explicit option, LUBAN_LOG_DIR, then the run's default."""
    for value in (explicit, os.environ.get("LUBAN_LOG_DIR"), fallback):
        if value is not None and str(value).strip():
            return Path(str(value).strip()).expanduser()
    return Path(DEFAULT_LOG_DIR)


def instance_filename(run_id):
    # Run IDs contain their UTC start date; every process uses the same date.
    stamp = run_id.split("-")[1][:8]
    day = date.fromisoformat(stamp).isoformat()
    return f"luban-meter-{run_id}-{day}.log"


_HIDDEN_FIELDS = {
    "api_key", "authorization", "password", "secret", "token", "prompt",
    "messages", "answer", "headers", "input", "body", "completion", "output",
    "program", "access_token", "multi_system_prompt",
}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (dict, list)):
        for item in value.values() if isinstance(value, dict) else value:
            yield from _strings(item)


def log_parameters(value, *, secrets=None):
    """Render public configuration; optionally collect text to redact."""
    if isinstance(value, dict):
        public = {}
        for key, item in value.items():
            if (str(key).lower().replace("-", "_") in _HIDDEN_FIELDS
                    and not isinstance(item, Path)):
                if secrets is not None:
                    secrets.extend(_strings(item))
                public[key] = "[redacted]"
            else:
                public[key] = log_parameters(item, secrets=secrets)
        return public
    if isinstance(value, list):
        return [log_parameters(item, secrets=secrets) for item in value]
    if isinstance(value, str) and "://" in value:
        scheme, _, rest = value.partition("://")
        authority, sep, path = rest.partition("/")
        public = (scheme + "://" + authority.rsplit("@", 1)[-1] + sep
                  + path).split("?", 1)[0].split("#", 1)[0]
        if secrets is not None and public != value:
            secrets.append(value)
        return public
    return value


def log_command(argv, *, hidden_flags=()):
    """Render an argument vector without credentials or inline programs."""
    result = []
    hide_next = False
    secrets = []
    for argument in argv:
        text = str(argument)
        if hide_next:
            result.append("[redacted]")
            secrets.append(text)
            hide_next = False
            continue
        key, equal, value = text.partition("=")
        if key in hidden_flags or key.lower().replace("_", "-") in {
            "--api-key", "--password", "--token", "--secret", "--header",
            "--authorization", "--prompt", "--multi-system-prompt",
            "--request-params",
            "-c", "-e", "--env",
        } or key == "-H":
            hide_next = not equal
            result.append(key + ("=[redacted]" if equal else ""))
            if equal:
                secrets.append(value)
        else:
            public = log_parameters(text)
            result.append(public)
            if text != public:
                secrets.append(text)
    if secrets:
        add_log_secrets(secrets)
    return result


def redact_text(text):
    for handler in _logger.handlers:
        if isinstance(handler.formatter, RunFormatter):
            for secret in handler.formatter.secrets:
                text = text.replace(secret, "[redacted]")
            break
    return text


def _secret_forms(secrets):
    return sorted({form for secret in secrets if secret
                   for form in (secret, repr(secret)[1:-1],
                                ascii(secret)[1:-1])}, key=len, reverse=True)


def add_log_secrets(secrets):
    secrets = _secret_forms(secrets)
    if not secrets:
        return
    for handler in _logger.handlers:
        if isinstance(handler.formatter, RunFormatter):
            handler.formatter.secrets = sorted(
                set(handler.formatter.secrets).union(secrets),
                key=len, reverse=True,
            )


class RunFormatter(logging.Formatter):
    def __init__(self, task, secrets):
        super().__init__(DEFAULT_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        self.task = task
        self.secrets = _secret_forms(secrets)

    def format(self, record):
        record = copy.copy(record)
        fields = dict.fromkeys(("case", "phase", "sample", "request"), "-")
        fields.update(task=self.task)
        fields.update(_context.get())
        for key, value in fields.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        text = super().format(record)
        # Include formatted exception messages and source lines in redaction.
        for secret in self.secrets:
            text = text.replace(secret, "[redacted]")
        return text


class InstanceFileHandler(logging.FileHandler):
    """Serialize complete records including tracebacks across POSIX workers."""

    def emit(self, record):
        if os.name != "posix":
            return super().emit(record)
        import fcntl

        fcntl.flock(self.stream, fcntl.LOCK_EX)
        try:
            super().emit(record)
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)


class ConsoleHandler(logging.StreamHandler):
    """Clear a live progress line before writing a complete log record."""

    colors = {
        logging.DEBUG: "36", logging.INFO: "32", logging.WARNING: "38;5;208",
        logging.ERROR: "31", logging.CRITICAL: "1;31",
    }

    def __init__(self, stream, before_emit):
        super().__init__(stream)
        self.before_emit = before_emit
        self.color = (
            getattr(self.stream, "isatty", lambda: False)()
            and os.environ.get("TERM") != "dumb"
            and not os.environ.get("NO_COLOR")
        )

    def filter(self, record):
        return (not getattr(record, "file_only", False)
                and super().filter(record))

    def format(self, record):
        return self.colorize(super().format(record))

    def colorize(self, text):
        if self.color:
            for level, color in self.colors.items():
                label = logging.getLevelName(level)
                text = text.replace(
                    f" | {label} | ",
                    f" | \033[{color}m{label}\033[0m | ",
                )
        return text

    def emit(self, record):
        try:
            if self.before_emit is not None:
                self.before_emit()
            super().emit(record)
        except Exception:
            self.handleError(record)


@contextmanager
def console_progress(before_emit):
    """Coordinate an existing CLI scope without adding duplicate handlers."""
    handlers = [(handler, handler.before_emit) for handler in _logger.handlers
                if isinstance(handler, ConsoleHandler)]
    for handler, _ in handlers:
        handler.before_emit = before_emit
    try:
        yield
    finally:
        for handler, previous in handlers:
            handler.before_emit = previous


def relay_console(text):
    """Forward worker stderr only to terminals; workers already save logs."""
    for handler in _logger.handlers:
        if not isinstance(handler, ConsoleHandler):
            continue
        handler.acquire()
        try:
            if handler.before_emit is not None:
                handler.before_emit()
            handler.stream.write(handler.colorize(redact_text(text)))
            handler.flush()
        except (OSError, UnicodeError):
            _logger.warning("Worker console output failed", exc_info=True,
                            extra={"file_only": True})
        finally:
            handler.release()


def logged_script(main):
    """Give a standalone dataset preparation command its own invocation log."""
    @wraps(main)
    def run(*args, **kwargs):
        with configure_logging(resolve_log_dir()):
            _logger.info("Starting command: %s", log_command(sys.argv))
            return main(*args, **kwargs)
    return run


def task_secrets(tasks):
    """Collect declared secret strings without logging the task plan."""
    from luban_meter.core.benchmark import benchmarks
    from luban_meter.core.planning import schema_for

    values = []
    for task in tasks:
        scenario = benchmarks.get(task["scenario"])
        for case in task["cases"]:
            parameters = case["parameters"]
            log_parameters(parameters, secrets=values)
            for name, spec in schema_for(scenario, parameters).items():
                value = parameters.get(name)
                if spec.secret and isinstance(value, str) and value:
                    values.append(value)
    return values


@contextmanager
def log_context(**fields):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def configure_logging(
    log_dir, *, level="info", filename=None, task="-", secrets=(),
    stream=None, before_emit=None,
):
    """Configure file and stderr handlers, then close and restore on exit.

    Only the application namespace is configured. Caller-owned and root
    handlers are preserved, including handlers installed by model engines.
    """
    resolved = resolve_log_level(level)
    filename = filename or instance_filename(create_run_id("cli"))
    path = Path(log_dir).resolve() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [InstanceFileHandler(path, encoding="utf-8")]
    handlers.append(ConsoleHandler(stream, before_emit))
    formatter = RunFormatter(task, secrets)
    previous_level, propagate = _logger.level, _logger.propagate
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(resolved)
        _logger.addHandler(handler)
    _logger.setLevel(resolved)
    _logger.propagate = False
    token = _log_file.set(path)
    try:
        yield path
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code:
            _logger.exception("Execution aborted")
        else:
            _logger.debug("Command exited normally")
        raise
    finally:
        for handler in handlers:
            _logger.removeHandler(handler)
            handler.close()
        _logger.setLevel(previous_level)
        _logger.propagate = propagate
        _log_file.reset(token)
