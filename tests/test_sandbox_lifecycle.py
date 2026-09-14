"""Lifecycle regression tests: ownership, cancellation and cleanup failures."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from luban_meter.core.errors import ExecutionError
from luban_meter.core.models import BenchmarkSpec, CommandSpec, ResolvedRun, RunRequest
from luban_meter.execution.command import LocalCommandRunner
from luban_meter.execution.host import HostSession
from luban_meter.utils.docker_sandbox import (
    SCOPE_LABEL,
    DockerSandbox,
    DockerSandboxConfig,
    DockerSandboxScope,
    DockerSandboxUnavailable,
)


def make_scope(tmp_path):
    return DockerSandboxScope(
        tmp_path,
        DockerSandboxConfig(docker_host="unix:///test/docker.sock", image="test"),
    )


def test_scope_removes_only_its_own_containers(tmp_path, monkeypatch):
    scope = make_scope(tmp_path)
    (scope.directory / "candidate.pending").touch()
    containers = {"owned": ("candidate", scope.token), "other": ("other", "other-run")}

    def control(args, **kwargs):
        if args[0] == "ps":
            assert f"label={SCOPE_LABEL}={scope.token}" in args
            return "\n".join(
                f"{cid} {name}"
                for cid, (name, owner) in containers.items()
                if owner == scope.token
            )
        assert args[:2] == ["rm", "-f"]
        del containers[args[2]]
        return ""

    monkeypatch.setattr(scope.sandbox, "_control", control)
    scope.cleanup()
    assert containers == {"other": ("other", "other-run")}
    assert (scope.directory / "cleanup-ok").exists()


def test_interrupted_create_is_not_reported_clean(tmp_path, monkeypatch):
    scope = make_scope(tmp_path)
    (scope.directory / "candidate.pending").touch()
    monkeypatch.setattr(scope.sandbox, "_control", lambda *a, **kw: "")
    with pytest.raises(DockerSandboxUnavailable, match="unconfirmed"):
        scope.cleanup(grace_seconds=0)
    assert (scope.directory / "cleanup-error.txt").exists()
    assert not (scope.directory / "cleanup-ok").exists()


def test_delayed_creation_is_reconciled(tmp_path, monkeypatch):
    scope = make_scope(tmp_path)
    (scope.directory / "candidate.pending").touch()
    replies = iter(["", "id candidate", ""])
    removed = []

    def control(args, **kwargs):
        if args[0] == "ps":
            return next(replies)
        removed.append(args)
        return ""

    monkeypatch.setattr(scope.sandbox, "_control", control)
    scope.cleanup()
    assert removed == [["rm", "-f", "id"]]


def test_remove_failure_is_not_silenced(monkeypatch):
    sandbox = DockerSandbox(DockerSandboxConfig("unix:///test/docker.sock", "test"))

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "daemon unavailable")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(DockerSandboxUnavailable, match="daemon unavailable"):
        sandbox.remove_container("candidate")


@pytest.mark.parametrize("error", [ExecutionError("timeout"), KeyboardInterrupt()])
def test_host_always_cleans_scope_and_stops_monitor(tmp_path, monkeypatch, error):
    events = []

    class Monitor:
        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    class Runner:
        def run(self, command):
            assert command.env["LUBAN_METER_SANDBOX_SCOPE"]
            events.append("run")
            raise error

    monkeypatch.setattr("luban_meter.execution.host.DeviceMonitorDaemon", Monitor)
    monkeypatch.setattr(
        DockerSandboxScope, "cleanup", lambda self: events.append("cleanup")
    )
    request = RunRequest(
        "test", "inference", "humaneval", tmp_path / "c.yaml", None, None, tmp_path
    )
    benchmark = BenchmarkSpec(
        "inference", "humaneval", tmp_path / "b.py", tmp_path / "r.py"
    )
    with pytest.raises(type(error)):
        HostSession(
            Runner(), monitor_url="http://127.0.0.1:9400"
        ).execute(ResolvedRun(request, benchmark, {}))
    assert events == ["start", "run", "cleanup", "stop"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group termination")
def test_timeout_kills_descendant(tmp_path):
    marker = tmp_path / "leaked"
    child = f"import time; time.sleep(0.5); open({str(marker)!r}, 'w').close()"
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
    )
    with pytest.raises(ExecutionError, match="timed out"):
        LocalCommandRunner().run(
            CommandSpec([sys.executable, "-c", parent], timeout=0.2)
        )
    import time

    time.sleep(0.6)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
def test_sigterm_unwinds_runner_and_restores_handler(tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    child = (
        "import os,signal,time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"
    )
    with pytest.raises(ExecutionError, match="interrupted by signal"):
        LocalCommandRunner().run(CommandSpec([sys.executable, "-c", child], timeout=5))
    assert signal.getsignal(signal.SIGTERM) == previous


def test_create_failure_never_starts_candidate(tmp_path, monkeypatch):
    from luban_meter.benchmark.inference.humaneval.executor import (
        DockerSandboxExecutor,
        SandboxConfig,
    )
    from luban_meter.utils.docker_sandbox import REGISTRY_ENV

    executor = DockerSandboxExecutor(SandboxConfig("unix:///test/docker.sock", "test"))
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path))

    def create(**kwargs):
        raise DockerSandboxUnavailable("interrupted create")

    monkeypatch.setattr(executor.sandbox, "create_container", create)
    monkeypatch.setattr(executor.sandbox, "remove_container", lambda name: None)

    def unexpected_start(*args, **kwargs):
        pytest.fail("must not start after failed create")

    monkeypatch.setattr(subprocess, "Popen", unexpected_start)
    with pytest.raises(DockerSandboxUnavailable, match="interrupted create"):
        executor.execute("assert True")
    assert len(list(tmp_path.glob("*.pending"))) == 1
    assert not list(tmp_path.glob("*.done"))


@pytest.mark.skipif(
    os.name != "posix" or os.environ.get("LUBAN_METER_RUN_DOCKER_TESTS") != "1",
    reason="requires opt-in real Docker on POSIX",
)
def test_real_outer_timeout_reclaims_container(tmp_path):
    config = DockerSandboxConfig(
        os.environ.get("LUBAN_METER_DOCKER_HOST", "unix:///var/run/docker.sock"),
        os.environ.get(
            "LUBAN_METER_HUMANEVAL_IMAGE", "luban-meter-humaneval-sandbox:v1"
        ),
    )
    scope = DockerSandboxScope(tmp_path / "registry", config)
    scope.sandbox.preflight()
    marker = tmp_path / "created"
    script = (
        "import os, subprocess, time, uuid\n"
        "from pathlib import Path\n"
        "from luban_meter.utils.docker_sandbox import DockerSandbox, DockerSandboxConfig, REGISTRY_ENV\n"
        f"s = DockerSandbox(DockerSandboxConfig({config.docker_host!r}, {config.image!r}))\n"
        "name = 'luban-lifecycle-' + uuid.uuid4().hex\n"
        "Path(os.environ[REGISTRY_ENV], name + '.pending').touch()\n"
        "cid = s.create_container(name=name, hostname='lifecycle-test')\n"
        "p = subprocess.Popen([*s.client_prefix(), 'start', '--attach', '--interactive', cid], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=s.environment())\n"
        "for _ in range(100):\n"
        "    if s._control(['inspect', '--format', '{{.State.Running}}', cid]) == 'true':\n"
        "        break\n"
        "    time.sleep(0.05)\n"
        "else:\n"
        "    raise RuntimeError('container did not enter running state')\n"
        f"Path({str(marker)!r}).write_text(cid)\n"
        "time.sleep(60)\n"
    )
    try:
        try:
            result = LocalCommandRunner().run(
                CommandSpec(
                    [sys.executable, "-c", script],
                    timeout=10,
                    env={
                        **scope.environment(),
                        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
                    },
                )
            )
        except ExecutionError as exc:
            assert "timed out" in str(exc)
        else:
            pytest.fail(
                f"helper exited before timeout ({result.returncode}): {result.stderr}"
            )
    finally:
        scope.cleanup()
    assert marker.exists(), "Docker create did not finish before outer timeout"
    assert (scope.directory / "cleanup-ok").exists()
    assert not scope.sandbox._control(
        [
            "ps",
            "-aq",
            "--filter",
            f"label={SCOPE_LABEL}={scope.token}",
        ]
    )
