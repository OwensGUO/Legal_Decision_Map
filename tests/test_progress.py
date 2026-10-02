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
