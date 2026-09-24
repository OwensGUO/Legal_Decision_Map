# Conditional Legal Decision Landscape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a CPU-testable and server-ready typed-counterfactual legal prediction research project.

**Architecture:** Frozen domain objects and dataset adapters feed auditable preprocessing, factor extraction, typed counterfactual generation, a backbone-independent multitask model, and independent evaluation modules. Heavy Qwen, vLLM, and quantization dependencies are lazy-loaded so pure logic and dry-runs remain runnable on CPU.

**Tech Stack:** Python 3.12, PyTorch 2.10.0, vLLM 0.19.1, Transformers 5.5.3, PEFT 0.18.1, bitsandbytes 0.49.2, Accelerate 1.13.0, HTTPX, PyYAML, NumPy, pytest, Ruff.

## Global Constraints

- Never modify or copy raw datasets into the repository.
- YAML server defaults are overridden by environment variables, then CLI.
- Default commands must not start full generation or training.
- CAIL-small is the single-defendant main experiment; CMDL-small is the multi-defendant extension.
- Do not use CMDL-big unless complete train/validation/test files exist.
- Preserve multi-label charges and keep life/death sentences out of month regression.
- Qwen3.6 realizes supplied interventions only and uses no external API.
- FlashAttention is optional; SDPA is the default.
- The server is Ubuntu 20.04 with immutable NVIDIA driver 535.171.04 and CUDA toolkit 12.2.
- The allocated physical GPUs are exactly 4, 5, 6, and 7.
- `flash-linear-attention` is optional and must not make the base installation fail.

---

### Task 1: Package, configuration, and domain schemas

**Files:** Create `pyproject.toml`, `requirements.txt`, `.gitignore`, `src/legal_landscape/config.py`, `src/legal_landscape/data/schema.py`, `src/legal_landscape/factors/schema.py`, and YAML configs; test in `tests/test_config_schema.py`.

**Interfaces:** Produce `CaseUnit`, `LegalFactors`, `InterventionSpec`, `load_config(path, env, overrides)`, JSON conversion helpers, and sentence eligibility predicates.

- [x] Write tests proving dataclass validation, finite-term eligibility, JSON round-trip, and CLI > environment > YAML precedence.
- [x] Run `pytest -q tests/test_config_schema.py` and confirm missing interfaces fail.
- [x] Implement the minimal validated dataclasses and recursive configuration merge.
- [x] Re-run the test file and refactor only while green.

### Task 2: Data adapters, sanitization, and manifests

**Files:** Create `src/legal_landscape/data/{cail,cmdl,sanitize,manifest}.py`, `tests/test_data.py`, and `scripts/audit_data.py`.

**Interfaces:** Produce `iter_cail`, `iter_cmdl`, `sanitize_text`, `build_manifest`, `assert_group_split_integrity`, and a safe audit CLI.

- [x] Write fixtures and tests proving one CAIL row maps to one unit, one CMDL row maps per defendant with shared group ID, redactions are audited, and hashes/counts are correct.
- [x] Run `pytest -q tests/test_data.py` and confirm interface failures.
- [x] Implement streaming adapters and conservative/strict regex sanitation.
- [x] Re-run tests and add local real-data smoke tests limited to two rows.

### Task 3: Factors and typed interventions

**Files:** Create `src/legal_landscape/factors/{extract,validate}.py`, `src/legal_landscape/counterfactual/interventions.py`, and `tests/test_factors_interventions.py`.

**Interfaces:** Produce `extract_factors`, `validate_factors`, and `propose_interventions` with explicit `rule_id`, `changed_fields`, and conditional rank directions.

- [x] Write tests with literal Chinese fact examples for amount, surrender, restitution, confession, role, and all three intervention types.
- [x] Verify tests fail because behavior is missing.
- [x] Implement deterministic extraction and bounded intervention rules for the configured charge boundaries.
- [x] Verify tests pass without applying global unconditional sentencing rules.

### Task 4: Counterfactual prompts, clients, persistence, and validation

**Files:** Create `src/legal_landscape/counterfactual/{prompts,generate,validators}.py`, `tests/test_counterfactual.py`, `scripts/generate_counterfactuals.py`, and `configs/cf/qwen36_27b.yaml`.

**Interfaces:** Produce `CounterfactualGenerator`, `MockGenerator`, `VLLMHTTPGenerator`, `validate_generation`, and resumable `generate_records`.

- [x] Write tests proving deterministic JSON mock output, provenance fields, leakage/identity/change checks, near-duplicate rejection, retry limits, and resume behavior.
- [x] Run the focused tests and confirm expected missing-feature failures.
- [x] Implement prompt versioning, HTTP request/response parsing, validators, and append-safe JSONL persistence.
- [x] Re-run tests and CLI dry-run/help checks.

### Task 5: Heads, marginalization, and typed losses

**Files:** Create `src/legal_landscape/models/{heads,predictor,losses}.py` and `tests/test_models_losses.py`.

**Interfaces:** Produce `MultiTaskHeads`, `LegalLandscapePredictor`, `marginalize_sentence`, `compute_typed_losses`, and `LossBreakdown`.

- [x] Write dummy-backbone tests proving probability sums, soft marginalization, invariant-only routing, no flip consistency, correct rank direction, and zero losses for empty masks.
- [x] Run focused tests and observe failures before production code.
- [x] Implement the smallest PyTorch modules and explicit mask-based loss router.
- [x] Re-run focused and cumulative tests; refactor while green.

### Task 6: Training, baselines, and experiment matrix

**Files:** Create `src/legal_landscape/training/train.py`, `scripts/train_model.py`, model/experiment YAML files, and `tests/test_training.py`.

**Interfaces:** Produce `load_backbone`, `build_experiment`, `run_train`, NaN diagnostics, safe dry-run, resume, and named B0-B5/M/A1-A6 configurations.

- [x] Write tests for dummy training, experiment toggles, config inspection without weights, NaN diagnostics, and refusal to launch without explicit execution.
- [x] Verify the tests fail for missing training behavior.
- [x] Implement lazy Transformers/PEFT/bitsandbytes loading and a CPU dummy path.
- [x] Verify the focused test and training CLI dry-run.

### Task 7: Metrics, grouped bootstrap, and evaluation CLI

**Files:** Create `src/legal_landscape/evaluation/{static_metrics,counterfactual_metrics,cmdl_case_metrics,bootstrap}.py`, `scripts/evaluate_model.py`, and `tests/test_evaluation.py`.

**Interfaces:** Produce static/CF/CMDL metrics, `cluster_bootstrap`, `holm_adjust`, and JSON evaluation output.

- [x] Write hand-calculated metric tests including exclusion of life/death from MAE and group-level resampling.
- [x] Run focused tests and confirm missing-feature failures.
- [x] Implement deterministic NumPy/scikit-learn/SciPy-compatible calculations with a pure NumPy fallback where simple.
- [x] Re-run tests and validate JSON output from a tiny fixture.

### Task 8: Environment audit, dataset build, requirements check, and documentation

**Files:** Create `scripts/{audit_environment,build_dataset,check_requirements}.py`, `README.md`, and `tests/test_cli.py`.

**Interfaces:** Every script supports `--help` and a safe bounded mode; README supplies the twelve server workflows requested by the specification.

- [x] Write subprocess tests for all help paths, dry-run safety, config parsing, and CPU-only requirement checks.
- [x] Run the focused tests and observe expected failures.
- [x] Implement the scripts, requirements metadata, local inference launch command, server upload workflow, and experiment commands.
- [x] Run `python -m compileall src scripts`, `pytest -q`, every CLI help command, and `ruff check .`; fix only evidenced failures.

### Task 9: Conda one-command pipeline and prediction export

**Files:** Create `run.sh`; modify `src/legal_landscape/training/train.py`,
`scripts/train_model.py`, `README.md`, `tests/test_training.py`, and
`tests/test_run_script.py`.

**Interfaces:** Produce `make_static_prediction_rows`,
`make_counterfactual_prediction_rows`, post-training distributed prediction
export, and a resumable `main|smoke|matrix` shell orchestrator.

- [x] Write failing tests for prediction rows, shell syntax, help, dry-run order, and Conda validation.
- [x] Run focused tests and confirm the missing interfaces/script fail.
- [x] Implement distributed prediction export and wire evaluation paths into the training CLI.
- [x] Implement `run.sh` with strict shell mode, traps, inference health wait, phase markers, overrides, and resume.
- [x] Run shell syntax, focused tests, full pytest/unittest, compileall, Ruff, and dry-run verification.

### Task 10: Reproducible vLLM dependency and environment contract

**Files:** Modify `requirements.txt`, `scripts/check_requirements.py`,
`scripts/audit_environment.py`, `tests/test_cli.py`; create
`requirements-optional.txt`.

**Interfaces:** Produce `requirement_report(cpu_only: bool) -> tuple[dict, bool]`,
strict non-zero exits for requested dependency failures, and environment output
that distinguishes physical GPU identity from CUDA-visible local rank.

- [x] Write failing subprocess tests proving a requested missing package makes
  the checker exit non-zero and CPU-only mode does not require vLLM or CUDA.
- [x] Run `pytest -q tests/test_cli.py` and confirm the new failure-status test
  fails because the current checker ignores training and inference failures.
- [x] Pin the approved vLLM/PyTorch/Transformers/Accelerate/PEFT/bitsandbytes
  stack in `requirements.txt`; move only `flash-linear-attention==0.5.0` to
  `requirements-optional.txt`.
- [x] Implement strict dependency reporting and expose Python, glibc, driver,
  runtime CUDA, visible GPU count, and device names without importing optional
  FLA as a core requirement.
- [x] Run `pytest -q tests/test_cli.py` and `ruff check` on the changed files.

### Task 11: Correct Qwen3.5 loading and memory-safe representations

**Files:** Modify `src/legal_landscape/training/train.py`,
`src/legal_landscape/models/heads.py`, `tests/test_training.py`, and
`tests/test_models_losses.py`.

**Interfaces:** Extend `InspectedModelConfig` with an architecture kind and text
backbone path; produce `select_model_loader(inspected)` and
`extract_text_backbone(model, inspected)`; preserve independent sigmoid charge
probabilities while returning normalized `charge_weights` for sentencing.

- [x] Add a failing config-inspection test using literal text-only and nested
  multimodal Qwen3.5 `config.json` fixtures. Assert the first selects causal LM
  and the second selects image-text conditional generation with a language
  backbone path.
- [x] Add a failing predictor test asserting independent charge probabilities
  may sum above one while `charge_weights.sum(-1)` equals one.
- [x] Run the two focused tests and verify failures are caused by the missing
  architecture distinction and old probability assertion.
- [x] Implement loader selection using the inspected architecture, load
  multimodal checkpoints with the correct Transformers auto class, freeze
  visual parameters, and expose only the language backbone to the predictor.
- [x] Remove `output_hidden_states=True`; consume `last_hidden_state` directly
  and raise a targeted error if the selected backbone does not expose it.
- [x] Run focused Torch tests and the full model/training test files.

### Task 12: Deterministic checkpoint resume

**Files:** Modify `src/legal_landscape/training/train.py` and
`tests/test_training.py`.

**Interfaces:** Produce `DeterministicEpochSampler` with
`state_dict() -> dict[str, int]` and `load_state_dict(state)`, and persist epoch,
sample offset, seed, input identity, dependency versions, and config summary in
`training_progress.json`.

- [x] Write a failing CPU test that consumes part of an epoch, saves sampler
  state, restores a new sampler, and asserts its remaining literal index order
  exactly equals the uninterrupted order.
- [x] Run the focused test and confirm it fails because the sampler is absent.
- [x] Implement the sampler and replace `shuffle=True`/`skip_first_batches` with
  explicit sampler state restoration for original and counterfactual loaders.
- [x] Extend checkpoint metadata and reject resume when the seed or input
  identity differs.
- [x] Run training tests, including the dummy optimizer step and checkpoint
  metadata test.

### Task 13: Clustered confidence intervals and paired Holm comparisons

**Files:** Modify `src/legal_landscape/evaluation/bootstrap.py`,
`scripts/evaluate_model.py`, `tests/test_evaluation.py`, and `tests/test_cli.py`.

**Interfaces:** Produce `bootstrap_metric_set(rows, metric_fn, names,
iterations=2000, seed=42)`, `paired_cluster_test(candidate, reference,
metric, group_key="group_id")`, and CLI options `--bootstrap-iterations`,
`--bootstrap-seed`, `--reference-input`, and `--primary-endpoints`.

- [x] Write failing hand-checked tests showing every row from a sampled group
  stays together, paired inputs must have identical group IDs, intervals are
  emitted for scalar metrics, and exactly five raw p-values receive Holm
  adjustment.
- [x] Run focused tests and observe missing-interface failures.
- [x] Implement deterministic group bootstrap, paired two-sided bootstrap
  p-values over group-level metric differences, and finite-value handling.
- [x] Integrate intervals into every evaluation kind; emit paired comparison
  and Holm fields only when `--reference-input` is supplied.
- [x] Run focused tests and a real tiny-JSONL CLI integration test.

### Task 14: Driver-535-safe one-command orchestration

**Files:** Modify `run.sh`, `README.md`, `tests/test_run_script.py`; create
`scripts/probe_gpu_stack.py`.

**Interfaces:** `probe_gpu_stack.py` returns non-zero on CUDA/BF16/NF4 failure
and prints JSON diagnostics. `run.sh` supports `INSTALL_FLA`,
`VLLM_ENFORCE_EAGER`, and `VLLM_DISABLE_P2P`, performs a real chat-completion
probe, and writes `outputs/run-summary.json`.

- [x] Write failing script tests using executable fixture commands to prove
  GPU IDs default to `4,5,6,7`, optional FLA is not imported by default,
  conservative vLLM flags appear, data audit precedes dataset build, and a
  failed generation probe aborts before counterfactual generation.
- [x] Run `pytest -q tests/test_run_script.py` and confirm behavioral failures.
- [x] Implement the standalone GPU probe and conservative vLLM environment:
  `NCCL_P2P_DISABLE=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
  `--disable-custom-all-reduce`, `--enforce-eager`, and
  `--language-model-only`; never set `VLLM_ENABLE_CUDA_COMPATIBILITY`.
- [x] Add a bounded real JSON chat request after health succeeds and print the
  vLLM log tail on startup/probe failure.
- [x] Add data audits, phase status recording, optional FLA install/probe, and
  final machine-readable summary without weakening process-group cleanup.
- [x] Update README commands to use physical GPUs 4-7 consistently and explain
  driver-535/PTX limitations and smoke-first operation.
- [x] Run shell syntax, script tests, dry-run main/smoke/matrix expansion, and
  CLI help checks.

### Task 15: Cumulative cleanup and acceptance

**Files:** Modify only files identified by failing checks, including existing
Ruff violations in counterfactual and test modules.

**Interfaces:** No new interfaces; this task proves the approved design is
internally consistent on the local CPU environment and leaves explicit server
gates for GPU-only behavior.

- [x] Run `python -m compileall -q src scripts` and fix syntax errors.
- [x] Run `pytest -q` and the Torch test suite with the available Conda Python;
  fix only failures caused by the approved behavior.
- [x] Run `ruff check .` and fix all reported violations.
- [x] Run `bash -n run.sh`, `MODE=smoke bash run.sh --dry-run`, and
  `MODE=matrix bash run.sh --dry-run`.
- [x] Re-read the design and requirements line by line, record any GPU-only
  checks that remain unverified locally, and update README usage accordingly.
