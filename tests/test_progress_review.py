"""Regression tests for real renderer, terminal and cleanup failure boundaries."""

from __future__ import annotations

import io
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from legal_landscape.progress import create_progress_reporter

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@pytest.fixture(autouse=True)
def real_terminal_environment(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("COLUMNS", raising=False)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
    ))


@pytest.mark.parametrize("operation", ["encoding", "fileno"])
def test_stream_metadata_failure_preserves_progress_and_business(operation):
    class Stream(io.StringIO):
        @property
        def encoding(self):
            if operation == "encoding":
                raise RuntimeError("encoding detection unavailable")
            return None

        def fileno(self):
            if operation == "fileno":
                raise RuntimeError("terminal detection unavailable")
            return super().fileno()

    stream = Stream()
    business = []
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": "always"}, is_terminal=True,
    ) as reporter:
        with reporter.task("Train", total=1) as task:
            business.append("optimizer")
            task.advance()
    assert business == ["optimizer"]
    assert "1/1 completed" in ANSI.sub("", stream.getvalue())


@pytest.mark.parametrize("failure_point", ["start", "update", "stop", "background"])
def test_real_rich_broken_pipe_never_exits_or_redirects_stdout(failure_point):
    # Isolate the old Rich default handler: it really calls dup2 before SystemExit.
    probe = textwrap.dedent('''
        import io, json, os, sys, time
        from rich.console import Console
        from legal_landscape.progress import create_progress_reporter
        report_fd = os.dup(1)
        before = os.fstat(1)
        events = []
        default_calls = []
        old_handler = Console.on_broken_pipe
        def observe_default(self):
            default_calls.append("default")
            return old_handler(self)
        Console.on_broken_pipe = observe_default
        class Stream(io.StringIO):
            broken = False
            failures = 0
            def write(self, text):
                if self.broken:
                    self.failures += 1
                    raise BrokenPipeError("closed progress pipe")
                return super().write(text)
        stream = Stream()
        failure_point = sys.argv[1]
        stream.broken = failure_point == "start"
        error = None
        try:
            with create_progress_reporter(
                stream=stream, environ={"PROGRESS": "always", "LC_ALL": "C.UTF-8"},
                is_terminal=True, plain_interval=0,
            ) as reporter:
                with reporter.task("Generate counterfactuals", total=2) as task:
                    events.append("generate")
                    if failure_point in {"update", "background"}:
                        stream.broken = True
                    if failure_point == "background":
                        time.sleep(0.35)
                    task.advance()
                events.append("save")
                if failure_point == "stop":
                    stream.broken = True
            events.append("json")
        except BaseException as caught:
            error = type(caught).__name__
        after = os.fstat(1)
        result = dict(events=events, error=error, default_calls=default_calls,
                      stdout_same=(before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                      failures=stream.failures)
        os.write(report_fd, json.dumps(result).encode())
        os.close(report_fd)
    ''')
    result = subprocess.run(
        [sys.executable, "-c", probe, failure_point], cwd=ROOT,
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["error"] is None, evidence
    assert evidence["default_calls"] == [], evidence
    assert evidence["stdout_same"], evidence
    assert evidence["failures"] >= 1
    assert evidence["events"] == ["generate", "save", "json"]


@pytest.mark.parametrize("mode", ["never", "always"])
@pytest.mark.parametrize("error", [BrokenPipeError, OSError, ValueError])
def test_evaluation_entrypoint_saves_json_when_progress_stream_is_broken(
    tmp_path, monkeypatch, mode, error
):
    class BrokenStream(io.StringIO):
        def write(self, text):
            raise error("progress unavailable")

    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps({
        "group_id": "g1", "true_charges": [1, 0], "predicted_charges": [1, 0],
        "true_articles": [1, 0], "predicted_articles": [1, 0],
        "predicted_months": 12.0, "true_months": 12.0, "penalty_type": "fixed_term",
    }) + "\n")
    destination = tmp_path / "metrics.json"
    main = runpy.run_path(str(ROOT / "scripts/evaluate_model.py"))["main"]
    output = io.StringIO()
    monkeypatch.setenv("PROGRESS", mode)
    monkeypatch.setattr(sys, "stderr", BrokenStream())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_model.py", "--input", str(source), "--output", str(destination),
        "--bootstrap-iterations", "2",
    ])
    assert main() == 0
    payload = json.loads(output.getvalue())
    assert payload == json.loads(destination.read_text())
    assert payload["bootstrap"]["iterations"] == 2


@pytest.mark.parametrize("mode", ["never", "always"])
def test_ascii_normalizes_production_descriptions_and_fields(mode):
    storage = io.BytesIO()
    stream = io.TextIOWrapper(storage, encoding="ascii", write_through=True)
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": mode, "LC_ALL": "C", "COLUMNS": "140"},
        is_terminal=True, plain_interval=0,
    ) as reporter:
        with reporter.task(
            "Assign groups · cmdl", total=None, loss="—", **{"état": "prêt"}
        ) as task:
            task.advance()
            task.update(description="Paired bootstrap · charge_macro_f1", loss="—")
    rendered = ANSI.sub("", storage.getvalue().decode("ascii"))
    assert "Assign groups - cmdl" in rendered
    assert "Paired bootstrap - charge_macro_f1" in rendered
    assert "loss=-" in rendered
    assert "etat=pret" in rendered
    assert "1/?" in rendered and "completed" in rendered
    assert "fallback" not in rendered


def _render_on_real_tty(probe, width):
    import errno
    import fcntl
    import struct
    import termios

    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))
    environment = {**os.environ, "PROGRESS": "auto", "LC_ALL": "C.UTF-8", "NO_COLOR": "1"}
    environment.pop("COLUMNS", None)
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", probe], cwd=ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=slave,
        )
        os.close(slave)
        chunks = []
        while True:
            try:
                data = os.read(master, 4096)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            chunks.append(data)
        assert process.wait(timeout=10) == 0
    finally:
        os.close(master)
    return ANSI.sub("", b"".join(chunks).decode())


@pytest.mark.parametrize("width", [8, 24, 48])
def test_real_tty_width_keeps_complete_core_before_secondary_fields(width):
    probe = textwrap.dedent('''
        from legal_landscape.progress import create_progress_reporter
        with create_progress_reporter() as reporter:
            with reporter.task("Generate counterfactuals", total=100, completed=50,
                               valid=40, invalid=5, failed=5, resumed=0, duplicate=0):
                pass
    ''')
    rendered = _render_on_real_tty(probe, width)
    final = next((line for line in reversed(rendered.splitlines()) if "completed" in line), "")
    assert "50/100" in final, rendered
    if width >= 24:
        assert len(final) <= width, rendered
    assert "valid=" not in final, rendered


def test_real_tty_resize_keeps_complete_core_at_current_width():
    probe = textwrap.dedent('''
        import fcntl, struct, sys, termios
        from legal_landscape.progress import create_progress_reporter
        with create_progress_reporter() as reporter:
            with reporter.task("Generate counterfactuals", total=100, completed=50,
                               valid=40, invalid=5, failed=5, resumed=0, duplicate=0):
                fcntl.ioctl(sys.stderr.fileno(), termios.TIOCSWINSZ,
                            struct.pack("HHHH", 24, 24, 0, 0))
    ''')
    rendered = _render_on_real_tty(probe, 48)
    final = next((line for line in reversed(rendered.splitlines()) if "completed" in line), "")
    assert "50/100" in final and len(final) <= 24, rendered


def test_rich_large_counts_keep_every_digit():
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": "always", "COLUMNS": "140"},
        is_terminal=True,
    ) as reporter:
        with reporter.task("Assign groups", total=1234567891, completed=1234567890):
            pass
    assert "1234567890/1234567891 completed" in ANSI.sub("", stream.getvalue())


@pytest.mark.parametrize("total,status", [(None, "completed"), (10, "completed"), (10, "failed")])
def test_rich_terminal_tasks_freeze_elapsed_and_spinner_across_next_task(total, status):
    now = [0.0]
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": "always", "LC_ALL": "C.UTF-8"},
        is_terminal=True, clock=lambda: now[0],
    ) as reporter:
        with reporter.task("Assign groups · cmdl", total=total) as task:
            now[0] = 5.0
            task.advance()
            if status == "failed":
                task.fail()
        first = reporter.progress.tasks[0]
        with reporter.task("Build train · cmdl", total=None) as task:
            now[0] = 30.0
            task.advance()
            assert first.stop_time == 5.0
            assert first.elapsed == 5.0
            assert first.total == total and first.completed == 1
            # Both clock and visual state of the finished row must remain unchanged.
            with reporter.progress.console.capture() as capture:
                reporter.progress.console.print(reporter.progress.get_renderable())
            at_thirty = capture.get()
            now[0] = 40.3
            with reporter.progress.console.capture() as capture:
                reporter.progress.console.print(reporter.progress.get_renderable())
            assert at_thirty.splitlines()[0] == capture.get().splitlines()[0]


@pytest.mark.parametrize("mode", ["never", "always"])
def test_rate_eta_uses_only_new_work_after_resume(mode):
    now = [0.0]
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": mode, "LC_ALL": "C", "COLUMNS": "180"},
        is_terminal=True, clock=lambda: now[0], plain_interval=0,
    ) as reporter:
        with reporter.task("Generate counterfactuals", total=100, completed=80) as task:
            initial = stream.getvalue()
            assert "rate=" not in initial and "eta=" not in initial
            now[0] = 10.0
            task.advance(10)
            rendered = ANSI.sub("", stream.getvalue())
            assert "rate=1.00/s" in rendered, rendered
            assert "eta=10.0s" in rendered, rendered


def test_rich_update_changes_state_without_per_item_output_and_forces_boundaries():
    now = [0.0]
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": "always", "LC_ALL": "C.UTF-8"},
        is_terminal=True, clock=lambda: now[0],
    ) as reporter:
        with reporter.task("Scan", total=1000) as task:
            after_start = stream.getvalue()
            assert "Scan" in after_start
            for _ in range(100):
                task.advance()
            task.update(description="Scan all", valid=100)
            assert reporter.progress.tasks[0].completed == 100
            assert reporter.progress.tasks[0].description == "Scan all"
            assert reporter.progress.tasks[0].fields["details"] == "valid=100"
            assert stream.getvalue() == after_start
            now[0] = 1.0
            task.advance()
            assert "101/1000" in ANSI.sub("", stream.getvalue())
        assert "completed" in stream.getvalue()
        with reporter.task("Fail", total=None) as task:
            before_fail = stream.getvalue()
            task.fail()
            assert "failed" in stream.getvalue()[len(before_fail):]
        assert reporter.progress.live._refresh_thread is None


@pytest.mark.parametrize("progress", ["never", "always"])
@pytest.mark.parametrize("exit_status", [0, 7])
@pytest.mark.parametrize("stream_failure", ["closed_fd", "broken_pipe"])
def test_closed_shell_progress_preserves_exit_cleanup_and_summary(
    tmp_path, progress, exit_status, stream_failure
):
    source = (ROOT / "run.sh").read_text()
    # Execute the actual helpers and EXIT trap, including the real kill/wait and JSON writer.
    helpers = source[source.index("phase_log() {"):source.index('\nphase "environment"')]
    pid_file = tmp_path / "server.pid"
    events_file = tmp_path / "events"
    reopened_stream = tmp_path / "reopened-stderr"
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    break_stream = "exec 2>&-" if stream_failure == "closed_fd" else f"exec 2>&{write_fd}"
    script = textwrap.dedent(f'''\
        set -Eeuo pipefail
        DRY_RUN=0 PHASE_TOTAL=9 PHASE_INDEX=0 PHASE_NAME="" PHASE_STARTED_AT=0
        WAIT_ACTIVE=0 WAIT_DYNAMIC_VISIBLE=0 WAIT_TERMINAL_MODE=0 WAIT_COMPACT=0
        PROGRESS_DYNAMIC={int(progress == "always")} PROGRESS_UNICODE=0 PROGRESS_STREAM_FAILED=0
        INFER_TIMEOUT=30 INFER_PID="" INFER_PGID="" COLUMNS=80
        OUTPUT_ROOT={shlex.quote(str(tmp_path))}
        MODE=smoke GPU_IDS=0 EXPERIMENTS=M SEEDS=42
        GENERATOR_MODEL=qwen38 GENERATOR_PATH=fixture P2P_POLICY=disable
        {helpers}
        sleep 30 >/dev/null 2>&1 &
        INFER_PID=$!
        printf '%s' "$INFER_PID" > {shlex.quote(str(pid_file))}
        phase "start vLLM"
        start_vllm_wait
        {break_stream}
    ''')
    if exit_status == 0:
        script += textwrap.dedent(f'''\
            update_vllm_wait
            finish_vllm_wait ready
            exec 2>{shlex.quote(str(reopened_stream))}
            phase "generate counterfactuals"
            printf 'business\n' >> {shlex.quote(str(events_file))}
            finish_phase completed
            stop_inference_server
            write_run_summary completed 0
        ''')
    script += f"exit {exit_status}\n"
    try:
        process = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=10,
            check=False, pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)
    pid = int(pid_file.read_text())
    try:
        assert process.returncode == exit_status, process.stderr
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        summary = json.loads((tmp_path / "run-summary.json").read_text())
        assert summary["exit_status"] == exit_status
        assert summary["status"] == ("failed" if exit_status else "completed")
        if not exit_status:
            assert events_file.read_text().splitlines() == ["business"]
            assert reopened_stream.read_text() == ""
    finally:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
