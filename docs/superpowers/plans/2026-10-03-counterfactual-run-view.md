# Counterfactual Run View Implementation Plan

> **For agentic workers:** Use executing-plans to implement this plan in the current session.

**Goal:** Train only on the current generation requests while preserving reusable cached results.

**Architecture:** Publish an atomic current-request view from the append-only cache, align generation
and training parent limits, and run smoke in a separate fresh-results namespace.

**Tech Stack:** Python JSONL, pathlib, tempfile/os atomic publication, Bash, pytest.

## Global Constraints

- Preserve existing cache bytes and generator provenance protections.
- Preserve invalid generation records for A6.
- Preserve main/matrix checkpoint recovery and prediction/metric definitions.
- Keep the single-graph training implementation unchanged.

## Task 1: Atomic view

- [x] Add failing tests in `tests/test_counterfactual_view.py` for current-request selection,
  missing/duplicate/malformed/spec/identity rejection and publication failure preservation.
- [x] Run `pytest -q tests/test_counterfactual_view.py` and confirm missing-helper failure.
- [x] Create `src/legal_landscape/counterfactual/view.py` with
  `publish_request_view(requests, cache_path, view_path, *, generator_identity, prompt_version)`.
  Match IDs to full specs, validate selected rows, stream them into an fsynced temporary file,
  and call `os.replace` only after every unique requested ID exists exactly once.
- [x] Re-run the focused tests.

## Task 2: CLI and parent scope

- [x] Add real CLI tests for cache resume plus view selection and `_requests(parent_limit=...)`.
- [x] Confirm failing tests before edits.
- [x] Add `--view-output` and `--parent-limit` to `scripts/generate_counterfactuals.py`.
  Publish the view only after `generate_records` returns with no transport failures;
  report view row/valid counts separately from newly generated counts.
- [x] Re-run CLI/view tests.

## Task 3: Pipeline and pair guards

- [x] Add failing dry-run tests checking view inputs, smoke checkpoint avoidance, and
  unchanged main checkpoint recovery. Add a runtime test for M with zero eligible pairs.
- [x] Pass view paths and parent limits from `run.sh`; archive existing smoke experiment
  directories before fresh training; use the smoke length namespace for evaluation too.
- [x] Fail typed experiments with no eligible pairs before tokenizer/model loading.
- [x] Run training, CLI, counterfactual, and pipeline tests; update README with cache/view
  semantics, smoke result locations, and reliable Bash pipeline exit reporting.

## Task 4: Verification and handoff

- [x] Run `ruff check .`, `bash -n run.sh`, Python compilation, `git diff --check`, and `pytest -q`.
- [x] Review cache preservation, invalid-row semantics, repeated smoke artifacts, and data scope.
- [x] Commit the verified files on local main and provide the server upload/run command.
- [x] Mark server GPU acceptance pending until a real run passes all nine stages.

## Verification notes

The read-only reviewer found an invalid-generation `parsed` schema edge. A failing regression
reproduced it, and publication now rejects malformed parsed values for both valid and invalid
records while retaining legitimate invalid records (including parsed null with raw JSON fallback)
for A6. Actual archive moves and transport-failure view preservation are also tested.

Server GPU acceptance is **pending**. The uploaded log fails during parent-record joining before
model load; passing CPU/control-flow tests does not establish OOM resolution or a successful
nine-stage server run. Simultaneous pipelines sharing one OUTPUT_ROOT are not supported here.
