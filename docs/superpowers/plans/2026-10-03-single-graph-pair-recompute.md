# Single-Graph Pair Recompute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the full typed-counterfactual objective while keeping at most one full predictor autograd graph resident on each GPU.

**Architecture:** Run the parent branch without gradients, differentiate the pair loss through small detached parent-output leaves and the live counterfactual graph, then replay the parent's RNG state and recompute only the parent graph to inject the saved output gradients. Keep the existing supervised backward, gradient-accumulation boundary, optimizer schedule, loss logging, and generated counterfactual data unchanged.

**Tech Stack:** Python 3.11, PyTorch autograd/RNG APIs, Hugging Face Accelerate, pytest, Ruff.

## Global Constraints

- Do not modify phase-6 counterfactual generation, pair validation, stored texts, loss definitions, weights, sequence-length defaults, checkpoint schemas, or prediction behavior.
- Never retain parent and counterfactual model graphs at the same time.
- Replay CPU and current-device CUDA RNG for the parent recomputation, then restore the post-counterfactual RNG state even when recomputation fails.
- Parent-leaf gradients obtained through `accelerator.backward` are already scaled; inject them with direct `torch.autograd.backward` without additional Accelerate scaling.
- Reduce parent-output requirements across ranks before branching so heterogeneous local pair types cannot produce mismatched DDP collective sequences.
- Preserve one optimizer step and one zero-grad call at the existing accumulation boundary.

---

### Task 1: Scoped RNG Replay

**Files:**
- Modify: `src/legal_landscape/training/train.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `_RngSnapshot`, `_capture_rng_state(torch_module, device)`, and `_replay_rng_state(torch_module, snapshot)`.
- Guarantees: replay installs the captured CPU/current-CUDA-device state and restores the caller's later state in a `finally` block.

- [x] **Step 1: Write the failing RNG replay tests**

Add real-PyTorch tests (guarded by the existing Torch availability marker) that capture a state, advance a dropout-bearing computation, replay the state to reproduce the same result, and prove the random stream after the context matches the stream that existed on context entry. Include an exception case to prove restoration occurs in `finally`.

- [x] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_training.py -k 'rng_replay'`

Expected: collection or assertion failure because the RNG snapshot/replay helpers do not exist.

- [x] **Step 3: Implement the minimal scoped RNG helpers**

Add `contextmanager`, an immutable `_RngSnapshot` dataclass, CPU cloning through `torch.get_rng_state`, and current-device CUDA cloning through `torch.cuda.get_rng_state(device)` only when the device type is CUDA. Restore the state captured on context entry from a `finally` block.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest -q tests/test_training.py -k 'rng_replay'`

Expected: all selected tests pass, or skip only when the local environment has no PyTorch.

### Task 2: Output-Gradient Injection

**Files:**
- Modify: `src/legal_landscape/training/train.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `_zero_output_anchor(outputs)` and `_backward_recomputed_parent(torch_module, outputs, *, charge_gradient, sentence_gradient)`.
- Consumes: a complete predictor output mapping containing `charge_logits`, `article_logits`, `penalty_type_logits`, `sentence_by_charge`, `sentence_months`, and `factor_logits`.
- Guarantees: every independent predictor head participates in DDP reduction; only charge and sentence receive nonzero saved gradients.

- [x] **Step 1: Write a failing real-Torch gradient-equivalence test**

Build a tiny shared-parameter model with charge, article, penalty, sentence, and factor heads. Compare the parameter gradient from the original joint parent/counterfactual graph against the detached-parent-leaf plus parent-recompute algorithm using literal pair-loss arithmetic. Assert all parameter gradients match.

- [x] **Step 2: Write a failing no-parent-gradient test**

Construct a pair loss that depends only on the counterfactual output and assert the helper contract permits skipping parent recomputation when both parent leaf gradients are `None`.

- [x] **Step 3: Run focused tests and verify RED**

Run: `pytest -q tests/test_training.py -k 'single_graph_recompute'`

Expected: failure because the output-anchor and recomputed-parent backward helpers do not exist.

- [x] **Step 4: Implement the minimal output-gradient helpers**

Make `_zero_output_anchor` sum zero-valued reductions of each output tensor. Make `_backward_recomputed_parent` call `torch.autograd.backward` once with real gradients for `charge_logits` and `sentence_months`, and `zeros_like` gradients for the independent output heads that otherwise have no contribution.

- [x] **Step 5: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_training.py -k 'single_graph_recompute'`

Expected: all selected tests pass, or skip only when PyTorch is unavailable.

### Task 3: Training-Loop Single-Graph Orchestration

**Files:**
- Modify: `src/legal_landscape/training/train.py:1120-1165`
- Modify: `tests/test_training.py:229-590`

**Interfaces:**
- Consumes: Task 1 RNG helpers and Task 2 output-gradient helpers.
- Produces: typed-pair training where parent graph-free forward, counterfactual backward, and parent recomputation execute in that order.

- [x] **Step 1: Extend the fake runtime and write failing orchestration tests**

Make the fake tensors support detached gradient leaves and record whether predictor calls occur under `no_grad`. Add assertions that M performs: supervised backward, graph-free parent call, counterfactual call, pair backward, parent recompute call, direct output-gradient injection, and one optimizer step. Add a boundary-only fake loss case asserting no parent recompute occurs when both parent leaf gradients are absent.

- [x] **Step 2: Run the orchestration tests and verify RED**

Run: `pytest -q tests/test_training.py -k 'single_graph or m_staged or boundary_only'`

Expected: ordering/call-count assertions fail because the current loop retains both pair graphs and never recomputes the parent.

- [x] **Step 3: Replace the pair section with single-graph recomputation**

Capture RNG before the parent call; run parent under `torch.no_grad`; create charge/sentence gradient leaves; delete the full parent output; run counterfactual with gradients; compute and validate the unchanged typed losses; add the counterfactual zero anchor; call `accelerator.backward` once; save detached parent-leaf gradients; delete counterfactual state; replay RNG and recompute/backpropagate the parent only when a saved parent gradient exists; then keep the existing sampler advance and optimizer schedule.

- [x] **Step 4: Run orchestration and training tests and verify GREEN**

Run: `pytest -q tests/test_training.py`

Expected: every training test passes; Torch-specific tests may skip if PyTorch is unavailable.

### Task 4: Project Verification and Local Main Commit

**Files:**
- Verify: `src/legal_landscape/training/train.py`
- Verify: `tests/test_training.py`
- Verify: `docs/superpowers/specs/2026-10-02-single-graph-pair-recompute-design.md`
- Verify: `docs/superpowers/plans/2026-10-03-single-graph-pair-recompute.md`

**Interfaces:**
- Produces: a reviewed local `main` commit ready for manual server upload.

- [x] **Step 1: Run static and full-suite verification**

Run: `python -m py_compile src/legal_landscape/training/train.py tests/test_training.py`

Run: `ruff check src/legal_landscape/training/train.py tests/test_training.py`

Run: `pytest -q`

Expected: compile and Ruff exit 0; all available tests pass.

- [x] **Step 2: Review the diff and repository state**

Run: `git diff --check`

Run: `git diff -- src/legal_landscape/training/train.py tests/test_training.py docs/superpowers/plans/2026-10-03-single-graph-pair-recompute.md`

Confirm no generated counterfactual code, loss definitions, sequence defaults, prediction code, or unrelated files changed.

- [ ] **Step 3: Commit the verified change on local main**

Run: `git add src/legal_landscape/training/train.py tests/test_training.py docs/superpowers/plans/2026-10-03-single-graph-pair-recompute.md`

Run: `git commit -m "fix: recompute counterfactual parent graph"`

- [ ] **Step 4: Re-run post-commit verification**

Run: `pytest -q`

Run: `git status --short --branch`

Expected: tests pass and local `main` is clean and ahead of `origin/main`; do not push because the server workflow is manual upload.

- [ ] **Step 5: Provide the server acceptance command without claiming GPU success**

First run on the server:

```bash
SMOKE_MAX_LENGTH=512 MODE=smoke PROGRESS=never bash run.sh \
  2>&1 | tee smoke-single-graph-512.log
```

Only after 512 completes with `exit_status=0`, run the default 1024-token smoke configuration.
