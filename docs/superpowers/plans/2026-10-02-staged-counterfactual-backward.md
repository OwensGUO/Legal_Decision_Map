# Staged Counterfactual Backward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent experiment M from retaining three Qwen autograd graphs while preserving its total gradient, optimizer schedule, and loss logs.

**Architecture:** Split each M microbatch into a supervised forward/backward stage and a paired-counterfactual forward/backward stage inside the existing Accelerate accumulation context. Convert completed loss breakdowns to detached scalar dictionaries for reporting, sum those dictionaries, and keep one optimizer call after both backward stages.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Accelerate, pytest.

## Global Constraints

- Preserve `gradient(supervised + paired) == gradient(supervised) + gradient(paired)` within numerical tolerance.
- Perform no more than one `optimizer.step()` call per microbatch.
- Keep all backward calls inside the existing `accelerator.accumulate(predictor)` context.
- B3 and experiments without typed pairs retain one forward and one backward per microbatch.
- Preserve JSONL/TensorBoard loss field names and values as the sum of both stages.
- Do not change sequence-length defaults, loss weights, checkpoint formats, or prediction behavior.

---

### Task 1: Detached Loss Reporting Helpers

**Files:**
- Modify: `tests/test_training.py`
- Modify: `src/legal_landscape/training/train.py`

**Interfaces:**
- Produces: `_detached_loss_values(losses: Any) -> dict[str, float]`.
- Produces: `_sum_loss_values(*parts: dict[str, float]) -> dict[str, float]`.
- Loss keys are exactly `charge`, `article`, `sentence`, `invariant`, `boundary`, `response`, `factor`, and `total`.

- [x] **Step 1: Write failing helper tests**

Add tests that create two simple loss namespaces, require detached scalar conversion, and require key-wise addition such that `total` and every named component are summed.

- [x] **Step 2: Verify RED**

```bash
pytest -q tests/test_training.py -k 'detached_loss_values or sum_loss_values'
```

Expected: import failure because both helpers are absent.

- [x] **Step 3: Implement minimal helpers**

Define one module-level tuple of loss field names. `_detached_loss_values` must call `detach()` before `float()`. `_sum_loss_values` must reject zero parts and return a new dictionary without mutating inputs.

- [x] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: all helper tests pass.

### Task 2: Staged Backward Training Loop

**Files:**
- Modify: `tests/test_training.py`
- Modify: `src/legal_landscape/training/train.py:1075-1165`

**Interfaces:**
- Consumes: `_detached_loss_values` and `_sum_loss_values` from Task 1.
- Produces: two `accelerator.backward(...)` calls for a typed-pair microbatch and one for an ordinary microbatch.
- Produces: one merged `loss_values: dict[str, float]` used by progress, JSONL, and TensorBoard.

- [x] **Step 1: Write the failing gradient-equivalence test**

Using real Torch tensors and `compute_typed_losses`, compute a combined supervised-plus-pair gradient on one scalar parameter. Recreate the parameter, backpropagate supervised and pair losses separately without clearing gradients, and assert the staged gradient equals the combined gradient with `torch.testing.assert_close`.

- [x] **Step 2: Extend the loop fixture and write failing orchestration tests**

Record every `Accelerator.backward` call, allow the fixture to choose experiment `M` or `B3`, and return distinct supervised and pair loss values. Assert:

- M performs two backward calls followed by one optimizer call for a one-microbatch run;
- B3 performs one backward call followed by one optimizer call;
- M writes a merged total and named components equal to the sum of staged values.

- [x] **Step 3: Verify RED**

```bash
pytest -q tests/test_training.py -k 'staged_backward or gradient_equivalence'
```

Expected: M currently performs one combined backward call, so the orchestration test fails.

- [x] **Step 4: Implement supervised-stage backward**

Move paired forward computation after the ordinary supervised loss. Validate and backpropagate the supervised loss immediately, snapshot its detached values, then delete references to its output and breakdown before starting pair forwards.

- [x] **Step 5: Implement paired-stage backward and merged reporting**

For a pair batch, compute only paired loss arguments, validate and backpropagate it, snapshot and add its detached values to the supervised values, and release pair references. Advance the pair sampler after successful paired backward. Leave the single optimizer/zero-grad calls after both stages. Replace later uses of `losses` with `loss_values`.

- [x] **Step 6: Verify GREEN and related training tests**

```bash
pytest -q tests/test_training.py -k 'staged_backward or gradient_equivalence'
pytest -q tests/test_training.py
```

Expected: focused and complete training tests pass.

### Task 3: Full Verification and Local Integration

**Files:**
- Verify: `src/legal_landscape/training/train.py`
- Verify: `tests/test_training.py`
- Verify: `docs/superpowers/plans/2026-10-02-staged-counterfactual-backward.md`

**Interfaces:**
- Consumes: all Task 1 and Task 2 behavior.
- Produces: a locally committed main branch ready for server acceptance testing.

- [x] **Step 1: Run static checks**

```bash
ruff check src/legal_landscape/training/train.py tests/test_training.py
git diff --check
```

- [x] **Step 2: Run the full test suite**

```bash
pytest -q
```

- [x] **Step 3: Inspect the final diff against the design constraints**

Confirm there are two backward calls only when a pair batch exists, one optimizer call remains, no checkpoint schema changes exist, and loss logs use merged detached values.

- [x] **Step 4: Commit**

```bash
git add src/legal_landscape/training/train.py tests/test_training.py docs/superpowers/plans/2026-10-02-staged-counterfactual-backward.md
git commit -m "fix: stage counterfactual training backward passes"
```

- [ ] **Step 5: Server acceptance**

First run `SMOKE_MAX_LENGTH=512 MODE=smoke PROGRESS=never bash run.sh`. If that succeeds, run the default `MODE=smoke PROGRESS=never bash run.sh`. Server acceptance is external to local automated verification and must not be claimed until those runs complete.
