from __future__ import annotations

import io
import re

import pytest

from legal_landscape.progress import (
    NullProgressReporter,
    PlainProgressReporter,
    RichProgressReporter,
    create_progress_reporter,
    resolve_dynamic_progress,
)


class FailingProgressStream(io.StringIO):
    def __init__(self, *, operation, fail_on, error=BrokenPipeError):
        super().__init__()
        self.operation = operation
        self.fail_on = fail_on
        self.error = error
        self.write_calls = 0
        self.flush_calls = 0

    def write(self, value):
        self.write_calls += 1
        if self.operation == "write" and self.write_calls == self.fail_on:
            raise self.error("progress stream unavailable")
        return super().write(value)

    def flush(self):
        self.flush_calls += 1
        if self.operation == "flush" and self.flush_calls == self.fail_on:
            raise self.error("progress stream unavailable")
        return super().flush()


def test_progress_terminal_detection_stream_failure_preserves_business():
    class UndetectableStream(io.StringIO):
        def isatty(self):
            raise OSError("terminal disappeared")

    business = []
    with create_progress_reporter(
        stream=UndetectableStream(), environ={"PROGRESS": "auto"}
    ) as reporter:
        with reporter.task("Train", total=1) as task:
            business.append("optimizer")
            task.advance()
    assert business == ["optimizer"]


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("fail_on", [1, 2, 3], ids=["start", "intermediate", "terminal"])
@pytest.mark.parametrize("error", [BrokenPipeError, OSError, ValueError])
def test_plain_stream_failure_preserves_business_and_stops_retrying(operation, fail_on, error):
    stream = FailingProgressStream(operation=operation, fail_on=fail_on, error=error)
    business = []
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": "never"}, plain_interval=0
    ) as reporter:
        with reporter.task("Train", total=1) as task:
            business.append("optimizer")
            task.advance()
        with reporter.task("Save checkpoint", total=1) as task:
            business.append("checkpoint")
            task.advance()
        business.append("writer:close")
    assert business == ["optimizer", "checkpoint", "writer:close"]
    assert getattr(stream, f"{operation}_calls") == fail_on


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_plain_stream_failure_on_task_exit_preserves_original_business_exception(operation):
    stream = FailingProgressStream(operation=operation, fail_on=2)
    original = RuntimeError("original training failure")
    with pytest.raises(RuntimeError) as caught:
        with create_progress_reporter(stream=stream, environ={"PROGRESS": "never"}) as reporter:
            with reporter.task("Train", total=1):
                raise original
    assert caught.value is original


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("fail_on", [1, 2], ids=["fallback-notice", "snapshot"])
def test_rich_stream_failure_during_fallback_does_not_repeat_business(
    monkeypatch, operation, fail_on
):
    class BrokenRenderer:
        def start(self):
            pass

        def add_task(self, *args, **kwargs):
            return 1

        def update(self, *args, **kwargs):
            raise RuntimeError("render failed")

        def stop(self):
            pass

    monkeypatch.setattr(
        "legal_landscape.progress._new_rich_progress", lambda *args: BrokenRenderer()
    )
    stream = FailingProgressStream(operation=operation, fail_on=fail_on)
    business = []
    with create_progress_reporter(
        stream=stream, environ={"PROGRESS": "always"}, is_terminal=True, plain_interval=0
    ) as reporter:
        with reporter.task("Train", total=2) as task:
            business.append("first")
            task.advance()
            business.append("second")
            task.advance()
        with reporter.task("Checkpoint", total=1) as task:
            business.append("checkpoint")
            task.advance()
    assert business == ["first", "second", "checkpoint"]
    assert getattr(stream, f"{operation}_calls") == fail_on


@pytest.mark.parametrize("failure_point", ["initialization", "start", "stop"])
def test_rich_stream_failure_on_lifecycle_fallback_preserves_business_exception(
    monkeypatch, failure_point
):
    class BrokenRenderer:
        def start(self):
            if failure_point == "start":
                raise RuntimeError("start failed")

        def add_task(self, *args, **kwargs):
            return 1

        def update(self, *args, **kwargs):
            pass

        def stop(self):
            if failure_point == "stop":
                raise RuntimeError("stop failed")

    def new_renderer(*args):
        if failure_point == "initialization":
            raise RuntimeError("initialization failed")
        return BrokenRenderer()

    monkeypatch.setattr("legal_landscape.progress._new_rich_progress", new_renderer)
    stream = FailingProgressStream(operation="write", fail_on=1)
    business = []
    original = RuntimeError("original business failure")
    with pytest.raises(RuntimeError) as caught:
        with create_progress_reporter(
            stream=stream, environ={"PROGRESS": "always"}, is_terminal=True
        ) as reporter:
            with reporter.task("Train", total=1):
                business.append("entered")
                raise original
    assert business == ["entered"]
    assert caught.value is original
    assert stream.write_calls == 1


@pytest.mark.parametrize(
    ("mode", "is_terminal", "expected"),
    [
        ("auto", True, True),
        ("auto", False, False),
        ("always", False, True),
        ("never", True, False),
    ],
)
def test_progress_mode_resolution(mode, is_terminal, expected):
    assert resolve_dynamic_progress(mode, is_terminal=is_terminal) is expected


def test_invalid_progress_mode_is_rejected():
    with pytest.raises(ValueError, match="auto, always, or never"):
        resolve_dynamic_progress("sometimes", is_terminal=True)


def test_plain_progress_writes_only_to_selected_stream(capsys):
    progress_stream = io.StringIO()
    with create_progress_reporter(
        stream=progress_stream,
        environ={"PROGRESS": "never"},
        is_terminal=False,
        plain_interval=0,
    ) as reporter:
        with reporter.task("Generate CAIL", total=4, completed=1, valid=1) as task:
            task.advance(valid=2)
    rendered = progress_stream.getvalue()
    assert "Generate CAIL" in rendered
    assert "2/4" in rendered
    assert "valid=2" in rendered
    assert "completed" in rendered
    assert capsys.readouterr() == ("", "")


def test_disabled_reporter_is_silent_even_when_progress_is_always():
    stream = io.StringIO()
    with create_progress_reporter(
        enabled=False, stream=stream, environ={"PROGRESS": "always"}, is_terminal=True
    ) as reporter:
        assert isinstance(reporter, NullProgressReporter)
        with reporter.task("Hidden", total=1) as task:
            task.advance()
    assert stream.getvalue() == ""


def test_plain_progress_rate_limits_only_intermediate_updates():
    stream = io.StringIO()
    time = [0.0]
    with create_progress_reporter(
        stream=stream,
        environ={"PROGRESS": "never"},
        clock=lambda: time[0],
        plain_interval=30,
    ) as reporter:
        with reporter.task("Generate", total=5) as task:
            task.advance()
            time[0] = 31
            task.advance()
            time[0] = 32
            task.advance()
    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert "started" in lines[0]
    assert "2/5 progress" in lines[1]
    assert "3/5 completed" in lines[2]


def test_rich_renderer_respects_color_width_ascii_and_final_status():
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream,
        environ={"PROGRESS": "always", "NO_COLOR": "1", "COLUMNS": "48", "LC_ALL": "C"},
        is_terminal=True,
    ) as reporter:
        assert isinstance(reporter, RichProgressReporter)
        with reporter.task("CAIL", total=4, completed=1, valid=1) as task:
            task.advance(valid=2)
    rendered = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stream.getvalue())
    assert "CAIL" in rendered
    assert "2/4" in rendered
    assert "completed" in rendered
    assert "valid=2" in rendered
    assert re.search(r"\d+:\d\d", rendered)
    assert all(ord(character) < 128 for character in rendered)
    assert "\x1b[" not in stream.getvalue()


def test_rich_renderer_uses_ascii_for_named_non_utf8_locale():
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream,
        environ={
            "PROGRESS": "always",
            "NO_COLOR": "1",
            "COLUMNS": "80",
            "LC_ALL": "en_US.ISO-8859-1",
        },
        is_terminal=True,
    ) as reporter:
        with reporter.task("CAIL", total=2) as task:
            task.advance()
    assert all(ord(character) < 128 for character in stream.getvalue())
    assert "1/2" in stream.getvalue()
    assert "completed" in stream.getvalue()


def test_rich_initialization_failure_falls_back_to_plain(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(
        "legal_landscape.progress._new_rich_progress",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no terminal")),
    )
    with create_progress_reporter(
        stream=stream,
        environ={"PROGRESS": "always"},
        is_terminal=True,
        plain_interval=0,
    ) as reporter:
        assert isinstance(reporter, PlainProgressReporter)
        with reporter.task("Fallback", total=1) as task:
            task.advance()
    assert "Fallback" in stream.getvalue()
    assert "completed" in stream.getvalue()


def test_rich_start_failure_falls_back_once_and_preserves_business_state(monkeypatch):
    class BrokenRenderer:
        def start(self):
            raise RuntimeError("start failed")

        def stop(self):
            pass

    monkeypatch.setattr(
        "legal_landscape.progress._new_rich_progress", lambda *args: BrokenRenderer()
    )
    stream = io.StringIO()
    business_calls = 0
    with create_progress_reporter(
        stream=stream,
        environ={"PROGRESS": "always"},
        is_terminal=True,
        plain_interval=0,
    ) as reporter:
        with reporter.task("Generate", total=1) as task:
            business_calls += 1
            task.advance()
    rendered = stream.getvalue()
    assert business_calls == 1
    assert rendered.count("fallback") == 1
    assert "1/1 completed" in rendered


def test_rich_update_failure_preserves_completed_count(monkeypatch):
    class BrokenRenderer:
        def start(self):
            pass

        def add_task(self, *args, **kwargs):
            return 1

        def update(self, *args, **kwargs):
            raise RuntimeError("render failed")

        def stop(self):
            pass

    monkeypatch.setattr(
        "legal_landscape.progress._new_rich_progress", lambda *args: BrokenRenderer()
    )
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream,
        environ={"PROGRESS": "always"},
        is_terminal=True,
        plain_interval=0,
    ) as reporter:
        with reporter.task("Recover", total=3, completed=1) as task:
            task.advance()
    rendered = stream.getvalue()
    assert rendered.count("fallback") == 1
    assert "2/3 completed" in rendered


def test_rich_stop_failure_preserves_final_status_and_count(monkeypatch):
    class BrokenRenderer:
        def start(self):
            pass

        def add_task(self, *args, **kwargs):
            return 1

        def update(self, *args, **kwargs):
            pass

        def stop(self):
            raise RuntimeError("stop failed")

    monkeypatch.setattr(
        "legal_landscape.progress._new_rich_progress", lambda *args: BrokenRenderer()
    )
    stream = io.StringIO()
    with create_progress_reporter(
        stream=stream,
        environ={"PROGRESS": "always"},
        is_terminal=True,
        plain_interval=0,
    ) as reporter:
        with reporter.task("Recover", total=3, completed=1) as task:
            task.advance(valid=2)
    rendered = stream.getvalue()
    assert rendered.count("fallback") == 1
    assert "2/3 completed" in rendered
    assert "valid=2" in rendered
