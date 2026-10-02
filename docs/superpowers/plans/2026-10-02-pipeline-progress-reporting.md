# Pipeline Progress Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add polished, accurate progress reporting to every pipeline phase while preserving scientific behavior, resumability, distributed-training safety, and machine-readable output.

**Architecture:** `run.sh` owns dependency-free top-level phase rendering and the vLLM startup wait. A new `legal_landscape.progress` adapter owns Rich rendering and deterministic plain-text fallback for Python workloads; domain functions accept the adapter's narrow protocol and remain usable with a null reporter. Progress goes to stderr, while existing JSON stays on stdout.

**Tech Stack:** Bash 4+, Python 3.12, Rich 14-compatible progress API, pytest, unittest, Accelerate, existing JSONL/checkpoint pipeline

## Global Constraints

- `PROGRESS` accepts exactly `auto`, `always`, or `never`; the default is `auto`.
- `auto` uses dynamic output only when stderr is an interactive terminal.
- `always` forces dynamic output; `never` always uses periodic newline-delimited text.
- `NO_COLOR` disables color without disabling progress.
- `--dry-run` prints numbered phases and commands but never starts animation or timed updates.
- Progress is presentation only and must not alter model, sampling, training, metric, checkpoint, resume, generated-record, prediction, or exit-status behavior.
- All progress output goes to stderr; existing JSON summaries stay on stdout.
- Renderer failure falls back to plain text and never repeats domain work.
- Only the Accelerate main process may render training or prediction progress.
- vLLM startup remains indeterminate until the health endpoint succeeds; do not infer a fake model-loading percentage from logs.
- Dynamic rendering must fall back to ASCII under a non-UTF-8 locale and must remain readable in narrow terminals.

---

## File Structure

- Create `src/legal_landscape/progress.py`: progress protocols, mode resolution, Rich renderer, plain renderer, null renderer, and reporter factory.
- Create `tests/test_progress.py`: deterministic adapter tests using in-memory streams and injected clocks/TTY state.
- Modify `requirements.txt`: add the bounded direct Rich dependency.
- Modify `run.sh`: validate `PROGRESS`, render nine numbered phases, report durations/failures, and animate or periodically log the vLLM wait.
- Modify `tests/test_run_script.py`: shell progress contract, dry-run behavior, invalid mode, and vLLM fallback-log tests.
- Modify `src/legal_landscape/data/build.py` and `scripts/build_dataset.py`: assignment/build split progress.
- Modify `tests/test_cli.py`: dataset progress stays on stderr and JSON stays on stdout.
- Modify `src/legal_landscape/counterfactual/generate.py` and `scripts/generate_counterfactuals.py`: resume-aware concurrent request progress.
- Modify `tests/test_counterfactual.py` and `tests/test_cli.py`: completed/resumed/duplicate/failure accounting and stream separation.
- Modify `src/legal_landscape/training/train.py` and `scripts/train_model.py`: optimizer-step and prediction-batch progress owned by the main process.
- Modify `tests/test_training.py`: progress initialization/update helpers, resume step, and worker suppression.
- Modify `src/legal_landscape/evaluation/bootstrap.py` and `scripts/evaluate_model.py`: bootstrap and comparison progress.
- Modify `tests/test_evaluation.py` and `tests/test_cli.py`: exact iteration accounting and JSON stream separation.
- Modify `README.md`: user-facing behavior, controls, and monitoring examples.

---

### Task 1: Build the Python Progress Adapter

**Files:**
- Create: `src/legal_landscape/progress.py`
- Create: `tests/test_progress.py`
- Modify: `requirements.txt:1-25`

**Interfaces:**
- Produces: `ProgressMode = Literal["auto", "always", "never"]`
- Produces: `ProgressTask` protocol with `advance(amount: int = 1, **fields: Scalar) -> None`, `update(*, completed: int | None = None, total: int | None = None, description: str | None = None, **fields: Scalar) -> None`, `succeed(**fields: Scalar) -> None`, and `fail(**fields: Scalar) -> None`
- Produces: `ProgressReporter` protocol extending `ContextManager[ProgressReporter]`, with `task(description: str, *, total: int | None, completed: int = 0, **fields: Scalar) -> ContextManager[ProgressTask]`
- Produces: `create_progress_reporter(*, enabled: bool = True, stream: TextIO | None = None, environ: Mapping[str, str] | None = None, is_terminal: bool | None = None, clock: Callable[[], float] = monotonic, plain_interval: float = 30.0) -> ProgressReporter`
- Produces: `NullProgressReporter`, used when progress is explicitly disabled for a worker rather than when `PROGRESS=never` requests plain logs

- [ ] **Step 1: Write failing mode-resolution and stream-separation tests**

Create `tests/test_progress.py` with table-driven tests that inject the terminal state instead of depending on the test runner's TTY:

```python
from __future__ import annotations

import io

import pytest

from legal_landscape.progress import create_progress_reporter, resolve_dynamic_progress


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


def test_plain_progress_writes_only_to_selected_stream():
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
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest tests/test_progress.py -v`

Expected: collection fails because `legal_landscape.progress` does not exist.

- [ ] **Step 3: Implement the mode resolver, protocols, null reporter, and plain reporter**

Create the public contracts and a newline-safe plain implementation. Use a `ProgressTask` implementation that tracks its own completed count and clock, emits start/completion/failure unconditionally, and rate-limits only intermediate updates:

```python
ProgressMode = Literal["auto", "always", "never"]
Scalar = str | int | float | bool | None


def resolve_dynamic_progress(mode: str, *, is_terminal: bool) -> bool:
    if mode not in {"auto", "always", "never"}:
        raise ValueError("PROGRESS must be auto, always, or never")
    return mode == "always" or (mode == "auto" and is_terminal)


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
    target = stream or sys.stderr
    environment = os.environ if environ is None else environ
    dynamic = resolve_dynamic_progress(
        environment.get("PROGRESS", "auto"),
        is_terminal=target.isatty() if is_terminal is None else is_terminal,
    )
    if not dynamic:
        return PlainProgressReporter(target, clock=clock, interval=plain_interval)
    return RichProgressReporter(target, environ=environment, clock=clock)
```

Do not import Rich yet; return a temporary plain reporter on the dynamic branch so the first tests can pass without hiding missing Rich behavior.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest tests/test_progress.py -v`

Expected: all mode and plain-output tests pass.

- [ ] **Step 5: Write failing Rich, color, width, ASCII, and fallback tests**

Add tests that force `PROGRESS=always`, inject `NO_COLOR`, set `COLUMNS=48`, and inject `LC_ALL=C`. Strip ANSI sequences before asserting that descriptions, counts, elapsed time, and final status remain. Monkeypatch the internal Rich-construction helper to raise and assert that reporter creation returns a working `PlainProgressReporter` instead of propagating:

```python
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
        with reporter.task("Fallback", total=1) as task:
            task.advance()
    assert "Fallback" in stream.getvalue()
    assert "completed" in stream.getvalue()
```

- [ ] **Step 6: Run the Rich tests and verify RED**

Run: `pytest tests/test_progress.py -v`

Expected: Rich-specific tests fail because the dynamic renderer and `_new_rich_progress` do not exist.

- [ ] **Step 7: Add Rich and implement the dynamic reporter**

Add `rich>=14.1,<15` to `requirements.txt`. Implement `_new_rich_progress` using `Console(file=stream, force_terminal=True, no_color="NO_COLOR" in environ, width=parsed_columns)` and a `Progress` configured with `SpinnerColumn`, `TextColumn`, `BarColumn`, `MofNCompleteColumn`, `TaskProgressColumn`, `TimeElapsedColumn`, `TimeRemainingColumn`, and a compact custom field column. Use `total=None` for indeterminate tasks, `expand=True` for wide terminals, and compact columns below 72 characters. Set `redirect_stdout=False` and `redirect_stderr=False` to preserve the JSON/output contract.

Wrap all Rich startup/update/stop calls at the adapter boundary. On any renderer exception, stop Rich if it started, replace it with `PlainProgressReporter`, emit one fallback notice, and continue from the same completed count without invoking domain work again.

- [ ] **Step 8: Run adapter tests and lint**

Run: `pytest tests/test_progress.py -v`

Expected: PASS.

Run: `ruff check src/legal_landscape/progress.py tests/test_progress.py`

Expected: PASS with no diagnostics.

- [ ] **Step 9: Commit the adapter**

```bash
git add requirements.txt src/legal_landscape/progress.py tests/test_progress.py
git commit -m "feat: add terminal progress adapter"
```

---

### Task 2: Add Top-Level Pipeline and vLLM Progress

**Files:**
- Modify: `run.sh:6-210,264-490,646-650`
- Modify: `tests/test_run_script.py:100-175,380-550`

**Interfaces:**
- Consumes: `PROGRESS=auto|always|never` from the shared user contract; the Bash layer does not import Python or Rich for phase rendering.
- Produces: nine numbered stage headings, stage completion/failure durations, a vLLM wait indicator, and a final total duration.
- Preserves: existing phase names as substrings so external log searches remain valid.

- [ ] **Step 1: Write failing shell-contract tests**

Extend `tests/test_run_script.py`:

```python
def test_dry_run_numbers_all_phases_without_terminal_control():
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "MODE": "smoke", "PROGRESS": "always"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for index, name in enumerate(
        (
            "environment", "validate generator provenance", "audit data",
            "build datasets", "start vLLM", "generate counterfactuals",
            "stop vLLM", "train and export predictions", "evaluate",
        ),
        start=1,
    ):
        assert f"[phase {index}/9] {name}" in result.stdout
    assert "\x1b[" not in result.stdout + result.stderr


def test_invalid_progress_mode_aborts_before_any_phase():
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "PROGRESS": "sometimes"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "PROGRESS must be auto, always, or never" in result.stderr
    assert "[phase" not in result.stdout
```

Add an assertion to the existing fake-vLLM real orchestration test that `PROGRESS=never` emits a newline-delimited `Waiting for vLLM` start record to stderr, even if the fake endpoint becomes healthy immediately.

- [ ] **Step 2: Run the shell tests and verify RED**

Run: `pytest tests/test_run_script.py -k 'progress or dry_run_lists_gpu_phases or real_server_p2p' -v`

Expected: numbering, invalid-mode, and vLLM-wait assertions fail against the current script.

- [ ] **Step 3: Implement dependency-free phase state**

Near the environment-variable defaults, add:

```bash
PROGRESS="${PROGRESS:-auto}"
case "$PROGRESS" in
  auto|always|never) ;;
  *)
    printf 'PROGRESS must be auto, always, or never; got %s.\n' "$PROGRESS" >&2
    exit 2
    ;;
esac
PHASE_TOTAL=9
PHASE_INDEX=0
PHASE_NAME=""
PHASE_STARTED_AT=0
PIPELINE_STARTED_SECONDS=$SECONDS
```

Replace `phase()` with helpers that finalize the previous stage, increment the index, store `$SECONDS`, and print exactly `[phase N/9] name`. Preserve plain dry-run output on stdout. Real-run status goes to stderr; dynamic mode may add color and carriage-return updates only when forced or when `[[ -t 2 ]]`. Use Unicode only when the locale contains `UTF-8` or `utf8`; otherwise use ASCII labels.

Add `rich` to the real environment import gate alongside the existing runtime
packages so a missing presentation dependency fails during the environment
phase rather than after expensive work begins.

Update `cleanup()` to finalize the active phase as failed exactly once before vLLM cleanup and summary writing. Finalize the evaluate phase before the completed run summary and print total elapsed time with the output path.

- [ ] **Step 4: Implement the vLLM indeterminate wait**

Before the `until curl ... /health` loop, emit the initial wait state. During the loop, render spinner frames and elapsed/timeout every health-check cycle in dynamic mode. In plain mode, emit at elapsed 0 and then no more frequently than every 30 seconds. Clear/finalize the line on health success, process death, timeout, or trap cleanup.

Keep the existing five-second polling interval, deadline calculation, `kill -0`, log tail, and exit messages unchanged.

- [ ] **Step 5: Run focused shell tests and syntax validation**

Run: `bash -n run.sh`

Expected: exit 0.

Run: `pytest tests/test_run_script.py -k 'progress or dry_run_lists_gpu_phases or real_server_p2p' -v`

Expected: PASS.

- [ ] **Step 6: Commit shell progress**

```bash
git add run.sh tests/test_run_script.py
git commit -m "feat: show pipeline and vllm startup progress"
```

---

### Task 3: Instrument Dataset Assignment and Building

**Files:**
- Modify: `src/legal_landscape/data/build.py:29-124`
- Modify: `scripts/build_dataset.py:24-47`
- Modify: `tests/test_cli.py:150-250`

**Interfaces:**
- Consumes: `ProgressReporter` and `NullProgressReporter` from Task 1.
- Changes: `build_dataset(..., progress: ProgressReporter | None = None) -> dict[str, Any]`.
- Produces: one task for duplicate-group assignment and one task for each output split, with fields `dataset`, `pass_name`, `split`, `accepted`, and `dropped`.

- [ ] **Step 1: Write a failing dataset progress test**

Extend the existing CLI dataset fixture test to invoke `scripts/build_dataset.py` with `PROGRESS=never`, then assert stdout is one valid JSON summary while stderr includes assignment and all three split names:

```python
payload = json.loads(result.stdout)
assert payload["dataset"] == "cail"
assert "Assign groups" in result.stderr
assert "Build train" in result.stderr
assert "Build valid" in result.stderr
assert "Build test" in result.stderr
assert "\x1b[" not in result.stderr
```

- [ ] **Step 2: Run the CLI test and verify RED**

Run: `pytest tests/test_cli.py -k 'build and progress' -v`

Expected: FAIL because dataset building emits no progress.

- [ ] **Step 3: Thread the reporter through data building**

Add an optional reporter to `_assign_groups_to_splits` and `build_dataset`. Use `NullProgressReporter` when omitted. Wrap the assignment scan and every output split in reporter tasks:

```python
with reporter.task(
    f"Assign groups · {data['dataset']}",
    total=len(paths) * limit if limit is not None else None,
    split="all",
) as task:
    for split, source in paths.items():
        for case in _iter_cases(data, source, split=split, limit=limit):
            # existing assignment logic
            task.advance(split=split)

with reporter.task(
    f"Build {split} · {data['dataset']}",
    total=limit,
    accepted=0,
    dropped=0,
) as task:
    for case in iterator:
        # advance on both accepted and dropped cases; preserve write order
        task.advance(accepted=count, dropped=dropped_units[split])
```

Move the advance into a small `finally`-free branch shared by the accepted and
dropped paths so each input case advances exactly once and exceptions do not
falsely report completion.

For files shorter than a configured limit, call `succeed(processed=...)` without forcing the displayed completed count to the upper bound.

In `scripts/build_dataset.py`, open one reporter context around `build_dataset` and pass it by keyword. Leave the final `print(json.dumps(summary, ...))` unchanged.

- [ ] **Step 4: Run data and CLI tests**

Run: `pytest tests/test_cli.py tests/test_data.py -v`

Expected: PASS.

Run: `ruff check scripts/build_dataset.py src/legal_landscape/data/build.py tests/test_cli.py`

Expected: PASS.

- [ ] **Step 5: Commit dataset progress**

```bash
git add scripts/build_dataset.py src/legal_landscape/data/build.py tests/test_cli.py
git commit -m "feat: report dataset build progress"
```

---

### Task 4: Instrument Resume-Aware Counterfactual Generation

**Files:**
- Modify: `src/legal_landscape/counterfactual/generate.py:234-321`
- Modify: `scripts/generate_counterfactuals.py:90-171`
- Modify: `tests/test_counterfactual.py:80-220,290-380`
- Modify: `tests/test_cli.py:1-150`

**Interfaces:**
- Consumes: `ProgressReporter` and `NullProgressReporter` from Task 1.
- Changes: `generate_records(..., progress: ProgressReporter | None = None) -> GenerationResult`.
- Progress total is `len(requests)`; initial completion is `skipped_completed + skipped_duplicate`; every pending future advances exactly once on success or transport failure.
- Fields: `valid`, `invalid`, `failed`, `resumed`, and `duplicate`.

- [ ] **Step 1: Write failing success/failure/resume accounting tests**

In `tests/test_counterfactual.py`, create a plain reporter backed by `io.StringIO` with `plain_interval=0`, pass it to the existing mock-generation and resume tests, and assert the final line reports the exact total and counters. Add a mixed generator that returns one valid record and raises `GenerationTransportError` for one request; assert the task still reaches `2/2` with `failed=1`.

In `tests/test_cli.py`, run `generate_counterfactuals.py --mock --execute` with `PROGRESS=never` and assert `json.loads(result.stdout)` still succeeds while progress appears only in stderr.

- [ ] **Step 2: Run generation tests and verify RED**

Run: `pytest tests/test_counterfactual.py tests/test_cli.py -k 'progress or resume or transport' -v`

Expected: FAIL because `generate_records` has no `progress` argument and the CLI emits no progress.

- [ ] **Step 3: Add terminal-outcome progress accounting**

After building `pending`, create the task with initial completion and fields:

```python
reporter = progress or NullProgressReporter()
initial = outcome.skipped_completed + outcome.skipped_duplicate
with reporter.task(
    "Generate counterfactuals",
    total=len(requests),
    completed=initial,
    valid=0,
    invalid=0,
    failed=0,
    resumed=outcome.skipped_completed,
    duplicate=outcome.skipped_duplicate,
) as progress_task:
    # existing bounded ThreadPoolExecutor loop
```

On transport failure, append the existing failure record and advance with the updated `failed` count. On success, preserve `write -> flush -> fsync -> append` ordering, then advance with `valid` or `invalid` counts derived from `record["validation"]["valid"]`. Do not count submission as completion.

Construct the reporter in `scripts/generate_counterfactuals.py`, pass it to `generate_records`, close it before printing the JSON report, and preserve generator cleanup in `finally`.

- [ ] **Step 4: Run generation tests and lint**

Run: `pytest tests/test_counterfactual.py tests/test_cli.py -k 'counterfactual or generate' -v`

Expected: PASS.

Run: `ruff check scripts/generate_counterfactuals.py src/legal_landscape/counterfactual/generate.py tests/test_counterfactual.py tests/test_cli.py`

Expected: PASS.

- [ ] **Step 5: Commit generation progress**

```bash
git add scripts/generate_counterfactuals.py src/legal_landscape/counterfactual/generate.py tests/test_counterfactual.py tests/test_cli.py
git commit -m "feat: report counterfactual generation progress"
```

---

### Task 5: Instrument Distributed Training and Prediction

**Files:**
- Modify: `src/legal_landscape/training/train.py:640-1293`
- Modify: `scripts/train_model.py:31-70`
- Modify: `tests/test_training.py:1-320`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `create_progress_reporter`, `ProgressReporter`, and `NullProgressReporter` from Task 1.
- Adds: `_training_progress_start(step: int, max_steps: int) -> tuple[int, int]` to clamp restored progress safely for display.
- Changes: `run_real_training(..., progress_factory: Callable[..., ProgressReporter] = create_progress_reporter) -> dict[str, Any]`.
- Produces: training task fields `experiment`, `step`, and `loss`; prediction task fields `kind` and `batches`.

- [ ] **Step 1: Write failing pure progress-state tests**

Add tests to `tests/test_training.py`:

```python
def test_training_progress_starts_at_restored_optimizer_step():
    from legal_landscape.training.train import _training_progress_start

    assert _training_progress_start(240, 1000) == (240, 1000)
    assert _training_progress_start(1200, 1000) == (1000, 1000)


def test_worker_progress_factory_is_disabled():
    from legal_landscape.training.train import _progress_enabled_for_process

    assert _progress_enabled_for_process(is_main_process=True) is True
    assert _progress_enabled_for_process(is_main_process=False) is False
```

Add a lightweight injected recording factory test around a factored `_report_optimizer_step(task, step, max_steps, total_loss)` helper to prove only completed optimizer steps advance and the current total loss is included.

- [ ] **Step 2: Run training tests and verify RED**

Run: `pytest tests/test_training.py -k 'progress' -v`

Expected: FAIL because the progress helpers do not exist.

- [ ] **Step 3: Implement optimizer-step reporting after Accelerate initialization**

After `Accelerator(...)` is created and the resume metadata has established `step`, create the reporter with `enabled=accelerator.is_main_process`. Start the training task at the clamped restored step and total `max_steps`:

```python
reporter = progress_factory(enabled=accelerator.is_main_process)
initial_step, display_total = _training_progress_start(step, max_steps)
with reporter as active_progress:
    with active_progress.task(
        f"Train {experiment.name}",
        total=display_total,
        completed=initial_step,
        experiment=experiment.name,
        loss="—",
    ) as train_progress:
        # existing epoch and loader loops
```

Call `_report_optimizer_step` only inside `if completed_optimizer_step:` after incrementing `step`. Pass `float(losses.total.detach())`. Keep JSONL/TensorBoard logging and checkpoint ordering unchanged. A worker receives a null reporter and executes no terminal operations.

- [ ] **Step 4: Add failing prediction-batch tests**

Test a new pure helper `_prediction_progress_total(loader: Sized) -> int` and a recording task passed through `_report_prediction_batch`. Assert one advance per evaluation-loader batch and one per counterfactual-loader batch, with independent `kind` fields.

Run: `pytest tests/test_training.py -k 'prediction_progress' -v`

Expected: FAIL because the helpers are missing.

- [ ] **Step 5: Instrument static and counterfactual prediction loops**

Within the same main-process reporter context, create sequential tasks around lines 1179 and 1244 using `total=len(evaluation_loader)` and `total=len(prediction_pair_loader)`. Advance after `gather_for_metrics` and after the main process has incorporated gathered rows. Workers use null tasks. Preserve final ordering by `_prediction_index` and all output JSONL contents.

Ensure `scripts/train_model.py` prints its final JSON only after `run_real_training` has closed all progress displays.

- [ ] **Step 6: Run training, CLI, and lint checks**

Run: `pytest tests/test_training.py tests/test_cli.py -k 'training or prediction or progress' -v`

Expected: PASS.

Run: `ruff check scripts/train_model.py src/legal_landscape/training/train.py tests/test_training.py tests/test_cli.py`

Expected: PASS.

- [ ] **Step 7: Commit training and prediction progress**

```bash
git add scripts/train_model.py src/legal_landscape/training/train.py tests/test_training.py tests/test_cli.py
git commit -m "feat: report training and prediction progress"
```

---

### Task 6: Instrument Bootstrap Evaluation

**Files:**
- Modify: `src/legal_landscape/evaluation/bootstrap.py:28-145`
- Modify: `scripts/evaluate_model.py:148-229`
- Modify: `tests/test_evaluation.py:120-195`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ProgressReporter` and `NullProgressReporter` from Task 1.
- Changes: `cluster_bootstrap(..., progress: ProgressReporter | None = None)`, `bootstrap_metric_set(..., progress: ProgressReporter | None = None)`, and `paired_cluster_test(..., progress: ProgressReporter | None = None)`.
- Each bootstrap function advances exactly once per completed resample and does not advance for the non-resampled point estimate.

- [ ] **Step 1: Write failing bootstrap-count tests**

Use a recording reporter in `tests/test_evaluation.py` and pass it to each bootstrap function. For `iterations=40`, assert the task starts with total 40, advances exactly 40 times, and succeeds. For a failing metric callback, assert the task is marked failed and the original exception is preserved.

Add a CLI test invoking `evaluate_model.py` with `PROGRESS=never`; assert stdout remains valid JSON and stderr contains the evaluation kind plus `iterations/iterations`.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `pytest tests/test_evaluation.py tests/test_cli.py -k 'bootstrap and progress' -v`

Expected: FAIL because bootstrap functions do not accept `progress`.

- [ ] **Step 3: Add one progress task per bootstrap operation**

In each bootstrap function, use a null reporter by default and wrap only the resampling loop:

```python
reporter = progress or NullProgressReporter()
with reporter.task(description, total=iterations, groups=len(groups)) as task:
    for index in range(iterations):
        # existing sample and metric calculation
        task.advance()
```

Use descriptions `Bootstrap confidence intervals` and `Paired bootstrap · {metric_name}`. Add an optional `description` keyword to `paired_cluster_test` so `evaluate_model.py` can include the endpoint name without changing metric semantics.

In `scripts/evaluate_model.py`, create one reporter context for the command, pass it to `bootstrap_metric_set`, and pass it to each sequential paired comparison. Close progress before writing and printing the payload.

- [ ] **Step 4: Run evaluation tests and lint**

Run: `pytest tests/test_evaluation.py tests/test_cli.py -k 'evaluation or bootstrap' -v`

Expected: PASS.

Run: `ruff check scripts/evaluate_model.py src/legal_landscape/evaluation/bootstrap.py tests/test_evaluation.py tests/test_cli.py`

Expected: PASS.

- [ ] **Step 5: Commit evaluation progress**

```bash
git add scripts/evaluate_model.py src/legal_landscape/evaluation/bootstrap.py tests/test_evaluation.py tests/test_cli.py
git commit -m "feat: report evaluation bootstrap progress"
```

---

### Task 7: Document Controls and Run Full Verification

**Files:**
- Modify: `README.md:30-70,75-145,420-445`
- Modify: `tests/test_run_script.py`

**Interfaces:**
- Consumes: all progress behavior implemented in Tasks 1-6.
- Produces: user documentation for default behavior, mode overrides, stdout/stderr separation, and representative output.

- [ ] **Step 1: Write the documentation assertions first**

Add a focused test that reads `README.md` and requires all three `PROGRESS` values, `NO_COLOR`, and an explanation that JSON remains on stdout while progress uses stderr:

```python
def test_readme_documents_progress_controls():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for value in ("PROGRESS=auto", "PROGRESS=always", "PROGRESS=never", "NO_COLOR"):
        assert value in readme
    assert "stderr" in readme
    assert "stdout" in readme
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run: `pytest tests/test_run_script.py::test_readme_documents_progress_controls -v`

Expected: FAIL because the controls are not documented.

- [ ] **Step 3: Update README usage and operational guidance**

Document:

```bash
# Default: dynamic in a terminal, periodic text in logs
MODE=smoke PROGRESS=auto bash run.sh

# Force plain log-safe progress
MODE=smoke PROGRESS=never bash run.sh

# Keep progress but disable color
NO_COLOR=1 MODE=smoke bash run.sh
```

Explain the nine top-level phases, nested counters, resume-aware initial counts,
main-process-only distributed output, stderr/stdout separation, and the fact
that vLLM startup is intentionally indeterminate.

- [ ] **Step 4: Run the complete verification suite**

Run: `python -m pip check`

Expected: exit 0 and no broken requirements.

Run: `bash -n run.sh`

Expected: exit 0.

Run: `MODE=smoke PROGRESS=never bash run.sh --dry-run`

Expected: exit 0, nine ordered numbered phases, no ANSI control sequences, and no real work.

Run: `pytest -q`

Expected: all tests pass with zero failures.

Run: `ruff check .`

Expected: all checks pass.

- [ ] **Step 5: Inspect final behavior and scope**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only the intended progress implementation, tests, requirements, and README are modified.

Review the diff and confirm that model configuration, sampling values, loss calculations, checkpoint format, generated JSONL schema, prediction schema, and evaluation payload schema are unchanged.

- [ ] **Step 6: Commit documentation and final verification changes**

```bash
git add README.md tests/test_run_script.py
git commit -m "docs: explain pipeline progress controls"
```

---

## Completion Criteria

- Every `run.sh` phase starts and finishes visibly with stage number and elapsed time.
- vLLM startup is visibly alive without claiming an invented percentage.
- Dataset, generation, training, prediction, and bootstrap work expose truthful completion boundaries.
- Counterfactual and training resume displays start from persisted progress.
- Distributed runs emit one training/prediction display only.
- Redirected logs contain periodic complete lines and no cursor-control noise.
- Existing JSON stdout remains parseable and unchanged in schema.
- Renderer failures degrade presentation only.
- The complete test suite, shell syntax check, dry-run, dependency check, Ruff, and diff check pass immediately before completion is claimed.
