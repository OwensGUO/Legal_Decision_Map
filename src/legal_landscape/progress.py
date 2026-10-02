"""Progress reporting for pipeline work, independent of domain operations."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from locale import getpreferredencoding
from time import monotonic
from typing import Literal, Protocol, Self, TextIO
from unicodedata import combining, normalize

from rich.console import Console, ConsoleDimensions
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
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


def _display(value: object, *, ascii_only: bool) -> str:
    text = " ".join(str(value).splitlines())
    if ascii_only:
        text = text.translate(str.maketrans({
            "·": "-", "—": "-", "–": "-", "−": "-", "…": "...",
            "‘": "'", "’": "'", "“": '"', "”": '"',
        }))
        text = "".join(char for char in normalize("NFKD", text) if not combining(char))
        text = text.encode("ascii", errors="replace").decode("ascii")
    return text


def _ascii_only(stream: TextIO, environ: Mapping[str, str]) -> bool:
    locale = next(
        (environ[key] for key in ("LC_ALL", "LC_CTYPE", "LANG") if environ.get(key)),
        getpreferredencoding(False),
    )
    encoding = locale.split(".", 1)[-1].split("@", 1)[0]
    try:
        stream_encoding = stream.encoding or "utf-8"
    except Exception:
        stream_encoding = "utf-8"
    return any(
        value.casefold().replace("-", "").replace("_", "") != "utf8"
        for value in (encoding, stream_encoding)
    )


def _configured_width(environ: Mapping[str, str]) -> int | None:
    raw_width = environ.get("COLUMNS", "")
    if raw_width.isdecimal() and int(raw_width) > 0:
        return int(raw_width)
    return None


def _terminal_width(stream: TextIO, environ: Mapping[str, str]) -> int | None:
    configured = _configured_width(environ)
    if configured is not None:
        return configured
    try:
        width = os.get_terminal_size(stream.fileno()).columns
        return width if width > 0 else None
    except Exception:
        return None


def _details(fields: Mapping[str, Scalar], *, ascii_only: bool) -> str:
    return " ".join(
        f"{_display(key, ascii_only=ascii_only)}={_display(value, ascii_only=ascii_only)}"
        for key, value in fields.items()
    )


def _timing(
    completed: float, initial: float, total: float | None, elapsed: float, status: str,
) -> str:
    added = completed - initial
    if elapsed <= 0 or added <= 0:
        return ""
    rate = added / elapsed
    text = f"rate={rate:.2f}/s"
    if total is not None and total > completed and status in {"running", "progress"}:
        text += f" eta={(total - completed) / rate:.1f}s"
    return text


class _PlainTask(AbstractContextManager["_PlainTask"]):
    def __init__(
        self,
        reporter: PlainProgressReporter,
        description: str,
        total: int | None,
        completed: int,
        fields: dict[str, Scalar],
        *,
        started: float | None = None,
        initial_completed: int | None = None,
        stopped: float | None = None,
    ) -> None:
        self.reporter = reporter
        self.description = description
        self.total = total
        self.completed = completed
        self.fields = fields
        self.started = reporter.clock() if started is None else started
        self.initial_completed = completed if initial_completed is None else initial_completed
        self.stopped = stopped
        self.last_update = reporter.clock()
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
        details = _details(self.fields, ascii_only=self.reporter.ascii_only)
        elapsed = max(0.0, (
            self.reporter.clock() if self.stopped is None else self.stopped
        ) - self.started)
        description = _display(self.description, ascii_only=self.reporter.ascii_only)
        if self.reporter.compact:
            self.reporter._write(f"{count} {status}\n")
            return
        line = f"{description}: {count} {status} elapsed={elapsed:.1f}s"
        timing = _timing(self.completed, self.initial_completed, self.total, elapsed, status)
        if timing:
            line += f" {timing}"
        if details:
            line += f" {details}"
        self.reporter._write(line + "\n")

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
            if self.stopped is None:
                self.stopped = self.reporter.clock()
            self._emit(status)

    def succeed(self, **fields: Scalar) -> None:
        self._finish("completed", **fields)

    def fail(self, **fields: Scalar) -> None:
        self._finish("failed", **fields)


class PlainProgressReporter(AbstractContextManager["PlainProgressReporter"]):
    def __init__(
        self, stream: TextIO, *, clock: Callable[[], float], interval: float,
        ascii_only: bool = False,
        compact: bool = False,
    ) -> None:
        self.stream = stream
        self.clock = clock
        self.interval = interval
        self.ascii_only = ascii_only
        self.compact = compact
        self._stream_failed = False

    def _write(self, message: str) -> None:
        if self._stream_failed:
            return
        try:
            self.stream.write(message)
            self.stream.flush()
        except Exception:
            # Progress is presentation only; stop touching a broken output stream.
            self._stream_failed = True

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
        return Text(task.fields.get("details", ""))


class _StatusSpinnerColumn(SpinnerColumn):
    def render(self, task: Task) -> Text:
        status = task.fields.get("status", "running")
        if status != "running":
            return Text("ok" if status == "completed" else "!")
        return super().render(task)


class _FrozenBarColumn(BarColumn):
    def render(self, task: Task):
        bar = super().render(task)
        if task.stop_time is not None:
            bar.animation_time = task.stop_time
        return bar


class _TimingColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(_timing(
            task.completed, task.fields.get("initial_completed", 0), task.total,
            task.elapsed or 0.0, task.fields.get("status", "running"),
        ))


class _AdaptiveProgress(Progress):
    def __init__(self, *, ascii_only: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ascii_only = ascii_only
        self.spinner = _StatusSpinnerColumn("line" if ascii_only else "dots")
        self.bar = _FrozenBarColumn(bar_width=20)
        self.percent = TaskProgressColumn()
        self.elapsed = TimeElapsedColumn()
        self.timing = _TimingColumn()
        self.details = _FieldsColumn()

    def make_tasks_table(self, tasks) -> Table:
        visible = [task for task in tasks if task.visible]
        if not visible:
            return Table.grid()
        width = self.console.width
        core = [Text(
            f"{task.completed}/{task.total if task.total is not None else '?'} "
            f"{task.fields.get('status', 'running')}"
        ) for task in visible]
        core_width = max(text.cell_len for text in core)
        if width < core_width:
            # No lossless Rich row fits; the adapter will emit complete plain records.
            raise ValueError("terminal too narrow for progress count and status")
        descriptions = [Text(task.description) for task in visible]
        description_width = max(text.cell_len for text in descriptions)
        optional = {
            "spinner": [self.spinner(task) for task in visible],
            "elapsed": [self.elapsed(task) for task in visible],
            "details": [self.details(task) for task in visible],
        }
        if width >= 72:
            if not self.ascii_only:
                optional["bar"] = [self.bar(task) for task in visible]
            optional["percent"] = [self.percent(task) for task in visible]
            optional["timing"] = [self.timing(task) for task in visible]
        sizes = {
            name: 20 if name == "bar" else max(item.cell_len for item in items)
            for name, items in optional.items()
        }
        optional = {name: items for name, items in optional.items() if sizes[name]}
        # Secondary fields go first; shorten descriptions only after dropping all extras.
        for name in ("details", "bar", "percent", "timing", "elapsed", "spinner"):
            needed = core_width + description_width + sum(
                sizes[key] for key in optional
            ) + len(optional) + 1
            if needed <= width:
                break
            optional.pop(name, None)
        description_width = min(
            description_width, width - core_width - sum(sizes[key] for key in optional)
            - len(optional) - 1,
        )
        names = [name for name in ("spinner",) if name in optional]
        if description_width > 0:
            names.append("description")
        names += [name for name in ("bar",) if name in optional]
        names.append("core")
        names += [name for name in ("percent", "elapsed", "timing", "details") if name in optional]
        sizes.update(description=description_width, core=core_width)
        table = Table.grid(*(
            Column(width=sizes[name], no_wrap=True, overflow="crop") for name in names
        ), padding=(0, 1))
        for index, description in enumerate(descriptions):
            if description.cell_len > description_width:
                suffix = "..." if self.ascii_only else "…"
                if description_width > len(suffix):
                    description.truncate(description_width - len(suffix), overflow="crop")
                    description.append(suffix)
                else:
                    description.truncate(max(0, description_width), overflow="crop")
            table.add_row(*(
                description if name == "description" else core[index] if name == "core"
                else optional[name][index] for name in names
            ))
        return table


class _ProgressConsole(Console):
    @property
    def size(self) -> ConsoleDimensions:
        dimensions = super().size
        if self._width is None:
            try:
                width = os.get_terminal_size(self.file.fileno()).columns
                if width > 0:
                    return ConsoleDimensions(width, dimensions.height)
            except Exception:
                pass
        return dimensions

    def on_broken_pipe(self) -> None:
        # Rich's default handler redirects process stdout and raises SystemExit.
        # A presentation stream must never change the independent JSON channel.
        raise BrokenPipeError("progress output pipe closed")


def _new_rich_progress(
    stream: TextIO, environ: Mapping[str, str], clock: Callable[[], float]
) -> Progress:
    width = _configured_width(environ)
    ascii_only = _ascii_only(stream, environ)
    console = _ProgressConsole(
        file=stream,
        force_terminal=True,
        no_color="NO_COLOR" in environ,
        width=width,
        emoji=not ascii_only,
    )
    return _AdaptiveProgress(
        ascii_only=ascii_only,
        console=console,
        get_time=clock,
        auto_refresh=False,
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
        self.started = reporter.clock()
        self.initial_completed = completed
        self.stopped: float | None = None
        self.task_id: int | None = None
        self.plain_task: _PlainTask | None = None
        self.finished = False
        self.final_status: str | None = None

    def _details(self) -> str:
        return _details(self.fields, ascii_only=self.reporter.ascii_only)

    def _activate_plain(self) -> None:
        plain = self.reporter.plain
        if plain is not None and self.plain_task is None:
            self.plain_task = _PlainTask(
                plain, self.description, self.total, self.completed, dict(self.fields),
                started=self.started, initial_completed=self.initial_completed,
                stopped=self.stopped,
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
            _display(self.description, ascii_only=self.reporter.ascii_only),
            total=self.total,
            completed=self.completed,
            status="running",
            details=self._details(),
            initial_completed=self.initial_completed,
        )
        self.reporter.last_refresh = self.reporter.clock()

    def _refresh(self, status: str = "running", *, force: bool = False) -> None:
        if self.plain_task is not None:
            return
        now = self.reporter.clock()
        refresh = force or now - self.reporter.last_refresh >= self.reporter.refresh_interval
        self.reporter._render(
            lambda progress: progress.update(
                self.task_id,
                description=_display(self.description, ascii_only=self.reporter.ascii_only),
                total=self.total,
                completed=self.completed,
                status=status,
                details=self._details(),
                refresh=refresh,
            )
        )
        if refresh:
            self.reporter.last_refresh = now

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
        self.stopped = self.reporter.clock()
        self.final_status = status
        if self.plain_task is not None:
            self.plain_task._finish(status, **fields)
        else:
            self.reporter._render(lambda progress: progress.stop_task(self.task_id))
            self._refresh(status, force=True)
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
        if self.reporter.plain is None:
            self.reporter.finished.append(self)
        return False


class RichProgressReporter(AbstractContextManager["RichProgressReporter"]):
    def __init__(
        self,
        progress: Progress,
        stream: TextIO,
        *,
        clock: Callable[[], float],
        plain_interval: float,
        ascii_only: bool = False,
    ) -> None:
        self.progress = progress
        self.stream = stream
        self.clock = clock
        self.plain_interval = plain_interval
        self.ascii_only = ascii_only
        self.plain: PlainProgressReporter | None = None
        self.active: list[_RichTask] = []
        self.finished: list[_RichTask] = []
        self.refresh_interval = 0.1
        self.last_refresh = clock()

    def _fallback(self) -> None:
        if self.plain is not None:
            return
        try:
            self.progress.stop()
        except Exception:
            pass
        self.plain = PlainProgressReporter(
            self.stream, clock=self.clock, interval=self.plain_interval,
            ascii_only=self.ascii_only,
        )
        self.plain._write("Progress renderer fallback to plain logs\n")
        for task in self.active:
            task._activate_plain()
        for task in self.finished:
            plain_task = _PlainTask(
                self.plain, task.description, task.total, task.completed, dict(task.fields),
                started=task.started, initial_completed=task.initial_completed,
                stopped=task.stopped,
            )
            plain_task._finish(task.final_status or "completed")

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
        self.finished.clear()
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
    if is_terminal is None:
        try:
            is_terminal = target.isatty()
        except Exception:
            is_terminal = False
    dynamic = resolve_dynamic_progress(
        environment.get("PROGRESS", "auto"),
        is_terminal=is_terminal,
    )
    ascii_only = _ascii_only(target, environment)
    width = _terminal_width(target, environment) if dynamic else None
    compact = width is not None and width < 20
    if not dynamic or compact:
        return PlainProgressReporter(
            target, clock=clock, interval=plain_interval, ascii_only=ascii_only, compact=compact
        )
    try:
        progress = _new_rich_progress(target, environment, clock)
    except Exception:
        plain = PlainProgressReporter(
            target, clock=clock, interval=plain_interval, ascii_only=ascii_only
        )
        plain._write("Progress renderer fallback to plain logs\n")
        return plain
    return RichProgressReporter(
        progress, target, clock=clock, plain_interval=plain_interval, ascii_only=ascii_only
    )
