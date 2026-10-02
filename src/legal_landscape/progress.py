"""Progress reporting for pipeline work, independent of domain operations."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from time import monotonic
from typing import Literal, Protocol, Self, TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

ProgressMode = Literal["auto", "always", "never"]
type Scalar = str | int | float | bool | None


class ProgressTask(Protocol):
    def advance(self, amount: int = 1, **fields: Scalar) -> None: ...

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
        **fields: Scalar,
    ) -> None: ...

    def succeed(self, **fields: Scalar) -> None: ...

    def fail(self, **fields: Scalar) -> None: ...


class ProgressReporter(AbstractContextManager["ProgressReporter"], Protocol):
    def task(
        self,
        description: str,
        *,
        total: int | None,
        completed: int = 0,
        **fields: Scalar,
    ) -> AbstractContextManager[ProgressTask]: ...


def resolve_dynamic_progress(mode: str, *, is_terminal: bool) -> bool:
    if mode not in {"auto", "always", "never"}:
        raise ValueError("PROGRESS must be auto, always, or never")
    return mode == "always" or (mode == "auto" and is_terminal)


class _NullTask(AbstractContextManager["_NullTask"]):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def advance(self, amount: int = 1, **fields: Scalar) -> None:
        pass

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
        **fields: Scalar,
    ) -> None:
        pass

    def succeed(self, **fields: Scalar) -> None:
        pass

    def fail(self, **fields: Scalar) -> None:
        pass


class NullProgressReporter(AbstractContextManager["NullProgressReporter"]):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def task(
        self,
        description: str,
        *,
        total: int | None,
        completed: int = 0,
        **fields: Scalar,
    ) -> AbstractContextManager[ProgressTask]:
        return _NullTask()


def _single_line(value: object) -> str:
    return " ".join(str(value).splitlines())


class _PlainTask(AbstractContextManager["_PlainTask"]):
    def __init__(
        self,
        reporter: PlainProgressReporter,
        description: str,
        total: int | None,
        completed: int,
        fields: dict[str, Scalar],
    ) -> None:
        self.reporter = reporter
        self.description = description
        self.total = total
        self.completed = completed
        self.fields = fields
        self.started = reporter.clock()
        self.last_update = self.started
        self.finished = False

    def __enter__(self) -> Self:
        self._emit("started")
        return self

    def __exit__(self, *exc: object) -> bool:
        if not self.finished:
            self._finish("failed" if exc[0] is not None else "completed")
        return False

    def _emit(self, status: str) -> None:
        count = f"{self.completed}/{self.total if self.total is not None else '?'}"
        details = " ".join(f"{key}={_single_line(value)}" for key, value in self.fields.items())
        elapsed = max(0.0, self.reporter.clock() - self.started)
        line = f"{_single_line(self.description)}: {count} {status} elapsed={elapsed:.1f}s"
        if details:
            line += f" {details}"
        self.reporter.stream.write(line + "\n")
        self.reporter.stream.flush()

    def _intermediate(self) -> None:
        now = self.reporter.clock()
        if now - self.last_update >= self.reporter.interval:
            self._emit("progress")
            self.last_update = now

    def advance(self, amount: int = 1, **fields: Scalar) -> None:
        self.completed += amount
        self.fields.update(fields)
        self._intermediate()

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
        **fields: Scalar,
    ) -> None:
        if completed is not None:
            self.completed = completed
        if total is not None:
            self.total = total
        if description is not None:
            self.description = description
        self.fields.update(fields)
        self._intermediate()

    def _finish(self, status: str, **fields: Scalar) -> None:
        if not self.finished:
            self.fields.update(fields)
            self.finished = True
            self._emit(status)

    def succeed(self, **fields: Scalar) -> None:
        self._finish("completed", **fields)

    def fail(self, **fields: Scalar) -> None:
        self._finish("failed", **fields)


class PlainProgressReporter(AbstractContextManager["PlainProgressReporter"]):
    def __init__(self, stream: TextIO, *, clock: Callable[[], float], interval: float) -> None:
        self.stream = stream
        self.clock = clock
        self.interval = interval

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def task(
        self,
        description: str,
        *,
        total: int | None,
        completed: int = 0,
        **fields: Scalar,
    ) -> AbstractContextManager[ProgressTask]:
        return _PlainTask(self, description, total, completed, dict(fields))


class _FieldsColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        status = task.fields.get("status", "running")
        details = task.fields.get("details", "")
        return Text(f"{status} {details}".strip())


def _new_rich_progress(
    stream: TextIO, environ: Mapping[str, str], clock: Callable[[], float]
) -> Progress:
    raw_width = environ.get("COLUMNS", "")
    width = int(raw_width) if raw_width.isdecimal() and int(raw_width) > 0 else None
    locale = environ.get("LC_ALL", environ.get("LC_CTYPE", environ.get("LANG", "")))
    ascii_only = locale in {"C", "POSIX"}
    console = Console(
        file=stream,
        force_terminal=True,
        no_color="NO_COLOR" in environ,
        width=width,
        emoji=not ascii_only,
    )
    columns: list[ProgressColumn] = [SpinnerColumn("line" if ascii_only else "dots")]
    columns.append(TextColumn("{task.description}"))
    if (width is None or width >= 72) and not ascii_only:
        columns.append(BarColumn())
    columns.append(MofNCompleteColumn())
    if width is None or width >= 72:
        columns.append(TaskProgressColumn())
    columns.append(TimeElapsedColumn())
    if width is None or width >= 72:
        columns.append(TimeRemainingColumn())
    columns.append(_FieldsColumn())
    return Progress(
        *columns,
        console=console,
        get_time=clock,
        expand=width is None or width >= 72,
        redirect_stdout=False,
        redirect_stderr=False,
    )


class _RichTask(AbstractContextManager["_RichTask"]):
    def __init__(
        self,
        reporter: RichProgressReporter,
        description: str,
        total: int | None,
        completed: int,
        fields: dict[str, Scalar],
    ) -> None:
        self.reporter = reporter
        self.description = description
        self.total = total
        self.completed = completed
        self.fields = fields
        self.task_id: int | None = None
        self.plain_task: _PlainTask | None = None
        self.finished = False

    def _details(self) -> str:
        return " ".join(f"{key}={_single_line(value)}" for key, value in self.fields.items())

    def _activate_plain(self) -> None:
        plain = self.reporter.plain
        if plain is not None and self.plain_task is None:
            self.plain_task = _PlainTask(
                plain, self.description, self.total, self.completed, dict(self.fields)
            )
            self.plain_task.__enter__()

    def __enter__(self) -> Self:
        self.reporter.active.append(self)
        if self.reporter.plain is not None:
            self._activate_plain()
        else:
            self.reporter._render(lambda progress: self._add_rich_task(progress))
        return self

    def _add_rich_task(self, progress: Progress) -> None:
        self.task_id = progress.add_task(
            self.description,
            total=self.total,
            completed=self.completed,
            status="running",
            details=self._details(),
        )

    def _refresh(self, status: str = "running") -> None:
        if self.plain_task is not None:
            return
        self.reporter._render(
            lambda progress: progress.update(
                self.task_id,
                description=self.description,
                total=self.total,
                completed=self.completed,
                status=status,
                details=self._details(),
                refresh=True,
            )
        )

    def advance(self, amount: int = 1, **fields: Scalar) -> None:
        self.completed += amount
        self.fields.update(fields)
        if self.plain_task is not None:
            self.plain_task.advance(amount, **fields)
        else:
            self._refresh()

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
        **fields: Scalar,
    ) -> None:
        if completed is not None:
            self.completed = completed
        if total is not None:
            self.total = total
        if description is not None:
            self.description = description
        self.fields.update(fields)
        if self.plain_task is not None:
            self.plain_task.update(
                completed=completed, total=total, description=description, **fields
            )
        else:
            self._refresh()

    def _finish(self, status: str, **fields: Scalar) -> None:
        if self.finished:
            return
        self.fields.update(fields)
        self.finished = True
        if self.plain_task is not None:
            self.plain_task._finish(status, **fields)
        else:
            self._refresh(status)
            if self.plain_task is not None:
                self.plain_task._finish(status, **fields)

    def succeed(self, **fields: Scalar) -> None:
        self._finish("completed", **fields)

    def fail(self, **fields: Scalar) -> None:
        self._finish("failed", **fields)

    def __exit__(self, *exc: object) -> bool:
        if not self.finished:
            self._finish("failed" if exc[0] is not None else "completed")
        self.reporter.active.remove(self)
        return False


class RichProgressReporter(AbstractContextManager["RichProgressReporter"]):
    def __init__(
        self,
        progress: Progress,
        stream: TextIO,
        *,
        clock: Callable[[], float],
        plain_interval: float,
    ) -> None:
        self.progress = progress
        self.stream = stream
        self.clock = clock
        self.plain_interval = plain_interval
        self.plain: PlainProgressReporter | None = None
        self.active: list[_RichTask] = []

    def _fallback(self) -> None:
        if self.plain is not None:
            return
        try:
            self.progress.stop()
        except Exception:
            pass
        self.plain = PlainProgressReporter(
            self.stream, clock=self.clock, interval=self.plain_interval
        )
        self.stream.write("Progress renderer fallback to plain logs\n")
        self.stream.flush()
        for task in self.active:
            task._activate_plain()

    def _render(self, operation: Callable[[Progress], object]) -> None:
        if self.plain is not None:
            return
        try:
            operation(self.progress)
        except Exception:
            self._fallback()

    def __enter__(self) -> ProgressReporter:
        try:
            self.progress.start()
        except Exception:
            self._fallback()
            assert self.plain is not None
            return self.plain
        return self

    def __exit__(self, *exc: object) -> bool:
        if self.plain is None:
            self._render(lambda progress: progress.stop())
        return False

    def task(
        self,
        description: str,
        *,
        total: int | None,
        completed: int = 0,
        **fields: Scalar,
    ) -> AbstractContextManager[ProgressTask]:
        return _RichTask(self, description, total, completed, dict(fields))


def create_progress_reporter(
    *,
    enabled: bool = True,
    stream: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    is_terminal: bool | None = None,
    clock: Callable[[], float] = monotonic,
    plain_interval: float = 30.0,
) -> ProgressReporter:
    if not enabled:
        return NullProgressReporter()
    target = sys.stderr if stream is None else stream
    environment = os.environ if environ is None else environ
    dynamic = resolve_dynamic_progress(
        environment.get("PROGRESS", "auto"),
        is_terminal=target.isatty() if is_terminal is None else is_terminal,
    )
    if not dynamic:
        return PlainProgressReporter(target, clock=clock, interval=plain_interval)
    try:
        progress = _new_rich_progress(target, environment, clock)
    except Exception:
        target.write("Progress renderer fallback to plain logs\n")
        target.flush()
        return PlainProgressReporter(target, clock=clock, interval=plain_interval)
    return RichProgressReporter(progress, target, clock=clock, plain_interval=plain_interval)
