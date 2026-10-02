# Pipeline Progress Reporting Design

**Date:** 2026-10-02  
**Status:** Approved for planning  
**Scope:** `run.sh` orchestration and the long-running Python data, generation,
training, prediction, and evaluation entry points

## Goal

Give operators immediate, polished feedback throughout a pipeline run without
changing the pipeline's scientific behavior, resumability, or machine-readable
outputs. Interactive terminals receive compact dynamic progress displays. Logs,
CI, redirected output, and unsupported terminals receive stable periodic text.

Every top-level phase must visibly start and finish. Work with a measurable
total must additionally report completion, throughput, elapsed time, and an ETA
when enough observations exist.

## User Interface

`run.sh` presents the pipeline as nine numbered stages:

1. environment
2. validate generator provenance
3. audit data
4. build datasets
5. start vLLM
6. generate counterfactuals
7. stop vLLM
8. train and export predictions
9. evaluate

The active stage has a prominent heading containing its number, name, and
overall completion. When the next stage starts, the prior stage is finalized
with a success mark and elapsed duration. On failure, the active stage is
finalized as failed before the existing diagnostic is printed. The final line
contains total elapsed time and `OUTPUT_ROOT`.

Long-running work adds a nested display:

- vLLM startup: spinner, elapsed wait, configured timeout, and endpoint;
- dataset building: dataset, split, records processed, and known limit;
- counterfactual generation: completed/total, valid, transport failures,
  throughput, elapsed time, and ETA;
- training: dataset, experiment, seed, optimizer step/maximum, current total
  loss, elapsed time, and ETA;
- prediction export: static or counterfactual batch completion;
- evaluation: evaluation job completion and bootstrap iteration progress.

Resume behavior is explicit. A counterfactual run begins with the number of
previously completed or duplicate requests. A resumed training run begins at
its restored optimizer step instead of visually restarting from zero.

## Display Modes

The new environment variable `PROGRESS` accepts:

- `auto` (default): dynamic rendering only when the relevant stream is an
  interactive terminal;
- `always`: force dynamic rendering, primarily for terminals whose TTY state is
  hidden by a launcher;
- `never`: disable dynamic rendering and use periodic text updates.

An invalid value exits with status 2 before any phase starts. `NO_COLOR` removes
color without disabling progress. `--dry-run` prints numbered stage headings
and the commands that would run, but never starts a spinner or emits timed
updates.

Dynamic progress is written to standard error. Existing JSON summaries remain
on standard output so scripts can continue to parse them. In non-dynamic mode,
updates are complete newline-terminated records and are rate-limited to avoid
large logs. Start, failure, and completion messages are never rate-limited.

## Architecture

### Shell orchestration

`run.sh` remains responsible for overall phase presentation because it owns the
phase order and must be able to report progress before Python dependencies are
installed. Small shell helpers will:

- validate and resolve `PROGRESS`;
- detect color and terminal width;
- render numbered phase headings;
- record phase and pipeline start times;
- finalize the current phase from normal flow or the existing exit trap;
- render the vLLM startup spinner or periodic fallback message.

The shell layer will use standard terminal control sequences only in dynamic
mode. It will not require `tput`, `watch`, or a separate executable. This keeps
the first environment phase operational on a minimally provisioned Ubuntu
host.

### Python progress adapter

Add `rich` as a direct runtime dependency and introduce a small progress module
under `src/legal_landscape/`. It will centralize:

- `PROGRESS` and TTY detection;
- color and terminal-width behavior;
- dynamic Rich bars and spinners;
- periodic plain-text fallback;
- elapsed time, rate, and ETA calculation;
- task creation, advancement, field updates, success, and failure.

The adapter exposes a narrow reporter interface and a no-op implementation.
Domain code accepts optional progress callbacks/reporters with no-op defaults;
it does not import Rich directly. This preserves the ability to test data,
generation, training, and evaluation logic without a terminal.

### Instrumentation boundaries

- Dataset iterators advance after a record has been durably accepted for the
  current split.
- Counterfactual generation advances only after each future reaches a terminal
  outcome. Successful records continue to be flushed and synchronized before
  the progress count advances.
- Training advances after a completed optimizer step, not after every gradient
  accumulation microbatch. Only the Accelerate main process owns a display.
- Prediction advances after a batch is gathered. Only the main process renders
  it.
- Bootstrap evaluation advances after each completed resample. Pure metric
  functions remain callable without a reporter.

The scripts construct reporters and pass them into core functions. Existing
callers that omit progress arguments retain current behavior.

## Dependency and Compatibility

`rich` is added to `requirements.txt` with a bounded compatible version range.
It is treated as a presentation dependency, not a source of pipeline state.
Progress failures must never alter generated records, checkpoints, predictions,
metrics, or exit status. If Rich cannot initialize, reporting falls back to
plain text and the computation continues.

The display supports UTF-8 symbols when available and an ASCII fallback for
non-UTF-8 locales. It avoids terminal cursor control when standard error is not
a TTY. Output remains legible at narrow widths by dropping secondary fields
before truncating task descriptions.

## Error Handling

The existing failure semantics remain authoritative. A failed task closes its
dynamic line, records its elapsed time, and then allows the original exception
or shell error to surface. Cleanup and vLLM process-group termination continue
to run through the existing trap.

Progress callbacks are best-effort presentation hooks. Renderer failures are
caught at the adapter boundary, switch the run to plain output, and do not
retry or repeat domain work. Broken pipes on a progress stream do not mask the
pipeline's real result.

## Testing

Implementation follows test-driven development. Tests will cover:

- `PROGRESS=auto|always|never` and invalid values;
- TTY, non-TTY, `NO_COLOR`, narrow terminal, and ASCII fallback rendering;
- nine ordered top-level stages in dry-run output;
- vLLM wait updates without changing timeout or process-liveness behavior;
- correct counterfactual totals for completed, resumed, duplicate, valid, and
  failed requests;
- training progress starting from a restored optimizer step;
- one renderer on the Accelerate main process and none on worker processes;
- prediction and bootstrap advancement at their true completion boundaries;
- unchanged JSON standard output and readable non-interactive logs;
- graceful fallback when the dynamic renderer cannot initialize.

The focused progress tests run first, followed by the complete CPU test suite,
shell syntax validation, Ruff, and the existing smoke dry-run. No real GPU or
model checkpoint is required for automated verification.

## Non-Goals

- No web dashboard or remote monitoring service.
- No changes to model, sampling, training, metric, checkpoint, or resume logic.
- No parsing of vLLM log text to invent a model-loading percentage; startup is
  shown as indeterminate until its health endpoint succeeds.
- No simultaneous output from every distributed worker.
- No requirement that machine-readable JSON consumers understand Rich markup
  or terminal control sequences.
