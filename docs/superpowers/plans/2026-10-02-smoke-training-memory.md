# Smoke Training Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MODE=smoke` train the full `M` experiment with a 1024-token default while preserving all production sequence lengths.

**Architecture:** Resolve and validate a smoke-only maximum length in `run.sh`, then reuse the existing `max_length_for_experiment` command-construction boundary to pass it to the Python trainer. Keep the trainer and formal experiment configurations unchanged; test behavior through the real dry-run CLI.

**Tech Stack:** Bash 5+, Python 3.12, pytest, Markdown.

## Global Constraints

- `MODE=smoke` defaults `SMOKE_MAX_LENGTH` to exactly `1024`.
- A caller may override the smoke value with a positive integer.
- B1 remains capped at 512 tokens in every mode.
- `main` and `matrix` retain existing lengths: B2 uses 4096; Qwen experiments use 4096 for CAIL and 8192 for CMDL.
- Invalid smoke values fail before expensive pipeline work.
- Do not change QLoRA, gradient-checkpointing, or distributed-training semantics.

---

### Task 1: Smoke Length Selection and Validation

**Files:**
- Modify: `tests/test_run_script.py`
- Modify: `run.sh:16-40,143-172,704-714`

**Interfaces:**
- Consumes: `MODE`, optional `SMOKE_MAX_LENGTH`, dataset name, and experiment name.
- Produces: the existing `--set model.max_length=<positive integer>` argument in each training command.

- [x] **Step 1: Add failing dry-run behavior tests**

Add tests for the following real CLI behaviors:

```python
def test_smoke_uses_memory_safe_training_length_by_default() -> None:
    result = _run_dry_run(MODE="smoke")
    assert _training_lengths(result) == {"cail": "1024", "cmdl": "1024"}


def test_smoke_training_length_can_be_overridden() -> None:
    result = _run_dry_run(MODE="smoke", SMOKE_MAX_LENGTH="768")
    assert _training_lengths(result) == {"cail": "768", "cmdl": "768"}


def test_main_keeps_production_training_lengths() -> None:
    result = _run_dry_run(MODE="main")
    assert _training_lengths(result, experiment="M") == {"cail": "4096", "cmdl": "8192"}


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_smoke_rejects_invalid_max_length(value: str) -> None:
    result = _run_dry_run(MODE="smoke", SMOKE_MAX_LENGTH=value)
    assert result.returncode == 2
    assert f"SMOKE_MAX_LENGTH must be a positive integer, got {value}." in result.stderr
    assert "[phase" not in result.stdout
```

- [x] **Step 2: Run the focused tests and verify RED**

```bash
pytest -q tests/test_run_script.py -k 'smoke_uses_memory_safe or smoke_training_length or main_keeps_production_training_lengths or smoke_rejects_invalid_max_length'
```

Expected: default and override assertions fail because smoke still emits 4096/8192, and invalid values are accepted.

- [x] **Step 3: Implement the minimum Bash behavior**

Resolve `SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-1024}"` in the `smoke)` branch. Reject values that do not match `^[1-9][0-9]*$` with `SMOKE_MAX_LENGTH must be a positive integer, got <value>.` and exit 2. Update `max_length_for_experiment` so B1 remains 512, smoke returns the resolved smoke value for every other experiment, and existing non-smoke branches remain unchanged.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [x] **Step 5: Run related script tests**

```bash
pytest -q tests/test_run_script.py
bash -n run.sh
```

Expected: all tests pass and Bash reports no syntax error.

### Task 2: Operator Documentation and Help Contract

**Files:**
- Modify: `tests/test_run_script.py`
- Modify: `run.sh:16-40`
- Modify: `README.md:70-105`

**Interfaces:**
- Consumes: the public environment variable `SMOKE_MAX_LENGTH`.
- Produces: discoverable help and README instructions describing its default and scope.

- [x] **Step 1: Add failing documentation contract assertions**

Extend the help test with `assert "SMOKE_MAX_LENGTH=1024" in help_result.stdout`. Add a README test that asserts the document includes `SMOKE_MAX_LENGTH=1024`, the example `SMOKE_MAX_LENGTH=768 MODE=smoke bash run.sh`, and the unchanged main/matrix scope.

- [x] **Step 2: Run the documentation tests and verify RED**

```bash
pytest -q tests/test_run_script.py -k 'valid_shell_syntax_and_help or readme_documents_smoke_training_length'
```

Expected: assertions fail because neither document exposes the setting yet.

- [x] **Step 3: Document the setting**

Add `SMOKE_MAX_LENGTH=1024` to `run.sh --help`. In the README smoke section, explain that smoke uses 1024 for both datasets to reduce activation memory, show `SMOKE_MAX_LENGTH=768 MODE=smoke bash run.sh`, and state that main/matrix retain formal 4096/8192 lengths.

- [x] **Step 4: Run the documentation tests and verify GREEN**

Run the Step 2 command. Expected: both tests pass.

### Task 3: Full Verification

**Files:**
- Verify: `run.sh`
- Verify: `README.md`
- Verify: `tests/test_run_script.py`

**Interfaces:**
- Consumes: all changes from Tasks 1 and 2.
- Produces: evidence that the fix is regression-safe and the repository remains cleanly formatted.

- [x] **Step 1: Run static checks**

```bash
bash -n run.sh
git diff --check
```

- [x] **Step 2: Run the full test suite**

```bash
pytest -q
```

Expected: all tests pass; environment-dependent skips remain skips.

- [x] **Step 3: Inspect the resulting dry-run commands**

```bash
MODE=smoke bash run.sh --dry-run
MODE=main bash run.sh --dry-run
```

Expected: smoke training commands contain `model.max_length=1024`; main `M` commands contain 4096 for CAIL and 8192 for CMDL.

- [x] **Step 4: Commit implementation**

```bash
git add run.sh README.md tests/test_run_script.py docs/superpowers/plans/2026-10-02-smoke-training-memory.md
git commit -m "fix: bound smoke training sequence length"
```
