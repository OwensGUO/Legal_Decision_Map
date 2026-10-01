# CUDA 13 and Qwen3.8 Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the project around the Ubuntu 24.04/CUDA 13 server and make Qwen3.8-27B BF16 the default, reproducible counterfactual generator while retaining Qwen3.6 as an explicit comparison.

**Architecture:** Keep one Python 3.12 Conda environment for preprocessing, vLLM inference, QLoRA training, and evaluation. Pin the CUDA 13-compatible package set, select the generator through a small `run.sh` interface, record the selected model in isolated artifacts, and make GPU/P2P capability probes decide whether high-performance collectives are safe.

**Tech Stack:** Bash, Python 3.12, PyTorch 2.13.0/cu130, vLLM 0.30.0, Transformers 5.15.0, Accelerate 1.15.0, PEFT 0.21.1, bitsandbytes 0.50.0, pytest, Ruff.

## Global Constraints

- Target Ubuntu 24.04 x86-64, NVIDIA driver 580.173.02, CUDA 13.0, and RTX 4090 physical GPU IDs `4,5,6,7`.
- Use Qwen3.8-27B BF16 at `/data/cguo/Qwen3.8-27B` as the default generator.
- Retain Qwen3.6-27B BF16 at `/data/cguo/Qwen3.6-27B` only as an explicit comparison generator.
- Keep Qwen3.5-9B at `/data/cguo/Qwen3.5-9B` as the NF4 QLoRA backbone.
- Default datasets are `/data/cguo/datasets/CMDL` and `/data/cguo/datasets/CAIL2018`.
- Never modify source models or datasets; all artifacts stay below `OUTPUT_ROOT`.
- Disable Qwen thinking and require JSON output for every counterfactual request.
- Do not install flash-linear-attention in the base environment; version 0.5.2 remains opt-in and must pass a real kernel probe.
- Preserve `GPU_IDS`, `NUM_PROCESSES`, local-only serving, fail-fast probes, resume behavior, and process-group cleanup.

---

### Task 1: Pin the CUDA 13 dependency contract

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-optional.txt`
- Modify: `scripts/check_requirements.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: package versions declared in the approved design.
- Produces: `EXPECTED_GPU_VERSIONS: dict[str, str]` matching the authoritative pins and a requirements file installable inside Python 3.12 Conda.

- [x] **Step 1: Change the requirement-version test to the CUDA 13 stack**

```python
def test_requirement_version_check_accepts_cuda13_stack_and_rejects_old_stack() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "check_requirements.py"))
    version_mismatches = module["version_mismatches"]
    expected = {
        "torch": "2.13.0",
        "vllm": "0.30.0",
        "transformers": "5.15.0",
        "accelerate": "1.15.0",
        "peft": "0.21.1",
        "bitsandbytes": "0.50.0",
    }
    assert version_mismatches({**expected, "torch": "2.13.0+cu130"}, expected) == {}
    assert version_mismatches({**expected, "vllm": "0.19.1"}, expected) == {
        "vllm": {"expected": "0.30.0", "actual": "0.19.1"}
    }
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_cli.py::test_requirement_version_check_accepts_cuda13_stack_and_rejects_old_stack`

Expected: FAIL because `EXPECTED_GPU_VERSIONS` still describes torch 2.10/vLLM 0.19.1.

- [x] **Step 3: Replace the dependency pins and checker expectations**

`requirements.txt` must contain these GPU pins:

```text
setuptools>=77.0.3,<81
torch==2.13.0
vllm==0.30.0
transformers==5.15.0
accelerate==1.15.0
peft==0.21.1
bitsandbytes==0.50.0
safetensors>=0.6.2
```

Keep the existing application/test dependencies, and update the header to describe Ubuntu 24.04, driver 580.173.02, CUDA 13.0, and the cu130 PyPI wheel. Set `flash-linear-attention[cuda]==0.5.2` in `requirements-optional.txt` and state that it is opt-in.

Update `EXPECTED_GPU_VERSIONS` to the six versions asserted in Step 1.

- [x] **Step 4: Verify the dependency contract**

Run: `pytest -q tests/test_cli.py::test_requirement_version_check_accepts_cuda13_stack_and_rejects_old_stack`

Expected: PASS.

Run: `python scripts/check_requirements.py --dry-run --limit 4`

Expected: JSON dry-run output and exit code 0.

- [x] **Step 5: Commit the dependency contract**

```bash
git add requirements.txt requirements-optional.txt scripts/check_requirements.py tests/test_cli.py
git commit -m "build: pin CUDA 13 model stack"
```

### Task 2: Add pairwise GPU peer-access diagnostics

**Files:**
- Modify: `scripts/probe_gpu_stack.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `probe_peer_access(torch: Any, device_count: int) -> dict[str, Any]` with `pairs` and `all_pairs_accessible`.
- Produces: CLI `--output PATH`, which writes the same JSON object printed to stdout.
- Consumes later: `run.sh` reads `.peer_access.all_pairs_accessible` from the probe JSON.

- [x] **Step 1: Add a failing pure peer-access test**

```python
def test_gpu_probe_reports_pairwise_peer_access() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "probe_gpu_stack.py"))

    class FakeCuda:
        @staticmethod
        def can_device_access_peer(source: int, target: int) -> bool:
            return {0, 1} == {source, target}

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda()})()
    assert module["probe_peer_access"](fake_torch, 3) == {
        "pairs": {"0->1": True, "0->2": False, "1->0": True,
                  "1->2": False, "2->0": False, "2->1": False},
        "all_pairs_accessible": False,
    }
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_cli.py::test_gpu_probe_reports_pairwise_peer_access`

Expected: FAIL because `probe_peer_access` is absent.

- [x] **Step 3: Implement peer diagnostics and JSON output**

Add:

```python
def probe_peer_access(torch: Any, device_count: int) -> dict[str, Any]:
    pairs = {
        f"{source}->{target}": bool(torch.cuda.can_device_access_peer(source, target))
        for source in range(device_count)
        for target in range(device_count)
        if source != target
    }
    return {"pairs": pairs, "all_pairs_accessible": bool(pairs) and all(pairs.values())}
```

Store this under `payload["peer_access"]` after device-count validation. Add `--output` as a `Path`; after probing, write the formatted JSON there as well as stdout. The output directory must be created by the caller, not by the probe.

- [x] **Step 4: Verify peer diagnostics**

Run: `pytest -q tests/test_cli.py::test_gpu_probe_reports_pairwise_peer_access tests/test_cli.py::test_environment_and_requirement_dry_runs_are_read_only`

Expected: PASS.

- [x] **Step 5: Commit the GPU probe**

```bash
git add scripts/probe_gpu_stack.py tests/test_cli.py
git commit -m "feat: report GPU peer access"
```

### Task 3: Add Qwen3.8 generator configuration and provenance isolation

**Files:**
- Create: `configs/cf/qwen38_27b.yaml`
- Modify: `configs/cf/qwen36_27b.yaml`
- Modify: `scripts/generate_counterfactuals.py`
- Modify: `tests/test_counterfactual.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: Qwen3.8 generator config with BF16-local model identity, TP4, 8,192 context, temperature 0.2, top-p 0.9, disabled thinking, and the existing validator thresholds.
- Changes CLI default: `scripts/generate_counterfactuals.py --config` defaults to `configs/cf/qwen38_27b.yaml`.
- Preserves: `VLLMHTTPGenerator` payload field `chat_template_kwargs={"enable_thinking": False}`.

- [x] **Step 1: Add failing config-default and payload tests**

Extend the CLI test so `parser().parse_args([]).config` equals `configs/cf/qwen38_27b.yaml`. Add a config load assertion:

```python
loaded = load_config(ROOT / "configs/cf/qwen38_27b.yaml")
assert loaded["generator"]["model_path"] == "/data/cguo/Qwen3.8-27B"
assert loaded["generator"]["thinking"] is False
assert loaded["generator"]["tensor_parallel_size"] == 4
```

Keep the existing HTTP generator assertion that `enable_thinking` is false.

- [x] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_counterfactual.py tests/test_cli.py -k 'qwen38 or http_generator'`

Expected: FAIL because the Qwen3.8 config and new default do not exist.

- [x] **Step 3: Create Qwen3.8 config and update defaults**

Copy the validated intervention boundaries and validation settings from the Qwen3.6 config. Change only model identity/path and add comments that BF16 precision is controlled by the local checkpoint/vLLM. Update Qwen3.6's default path to `/data/cguo/Qwen3.6-27B`.

- [x] **Step 4: Verify configuration and client behavior**

Run: `pytest -q tests/test_counterfactual.py tests/test_cli.py -k 'qwen38 or http_generator'`

Expected: PASS.

- [x] **Step 5: Commit generator configuration**

```bash
git add configs/cf/qwen38_27b.yaml configs/cf/qwen36_27b.yaml scripts/generate_counterfactuals.py tests/test_counterfactual.py tests/test_cli.py
git commit -m "feat: make Qwen3.8 the primary generator"
```

### Task 4: Realign one-command orchestration and performance defaults

**Files:**
- Modify: `run.sh`
- Modify: `tests/test_run_script.py`

**Interfaces:**
- Consumes: `probe_gpu_stack.py --output <path>` and its `peer_access.all_pairs_accessible` value.
- Produces environment interface: `GENERATOR_MODEL=qwen38|qwen36`, `QWEN38_PATH`, `QWEN36_PATH`, and `P2P_POLICY=auto|enable|disable`.
- Produces isolated counterfactual paths: `${OUTPUT_ROOT}/counterfactuals/${GENERATOR_MODEL}/{cail,cmdl}.jsonl`.
- Produces run summary fields: `generator_model`, `generator_path`, and `p2p_policy`.

- [x] **Step 1: Update dry-run tests for new defaults**

The smoke dry-run test must assert:

```python
assert "torch==2.13.0" in output
assert "/data/cguo/Qwen3.8-27B" in output
assert "--enable-prefix-caching" in output
assert "--enforce-eager" not in output
assert "counterfactuals/qwen38" in output
```

Add a second invocation with `GENERATOR_MODEL=qwen36` and assert it uses `/data/cguo/Qwen3.6-27B`, `configs/cf/qwen36_27b.yaml`, and `counterfactuals/qwen36`.

Add invalid-selector coverage asserting `GENERATOR_MODEL=unknown` exits 2 before any phase.

- [x] **Step 2: Run run-script tests and verify RED**

Run: `pytest -q tests/test_run_script.py`

Expected: FAIL on the old paths, package version, eager default, and absent selector.

- [x] **Step 3: Implement model selection and server defaults**

Set these defaults:

```bash
QWEN35_PATH="${QWEN35_PATH:-/data/cguo/Qwen3.5-9B}"
QWEN36_PATH="${QWEN36_PATH:-/data/cguo/Qwen3.6-27B}"
QWEN38_PATH="${QWEN38_PATH:-/data/cguo/Qwen3.8-27B}"
CAIL_ROOT="${CAIL_ROOT:-/data/cguo/datasets/CAIL2018}"
CMDL_ROOT="${CMDL_ROOT:-/data/cguo/datasets/CMDL}"
GENERATOR_MODEL="${GENERATOR_MODEL:-qwen38}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
P2P_POLICY="${P2P_POLICY:-auto}"
```

Resolve `GENERATOR_PATH`, `GENERATOR_CONFIG`, and the default served-model name in a `case`. Reject unknown values. Replace hard-coded Qwen3.6 references with the resolved values and isolate output paths by generator.

Add `--enable-prefix-caching`. Append `--enforce-eager` only when explicitly set to 1. In real execution, create `OUTPUT_ROOT`, write the GPU probe to `${OUTPUT_ROOT}/gpu-probe.json`; for `auto`, parse `all_pairs_accessible` and enable P2P/custom all-reduce only when true. `disable` must set `NCCL_P2P_DISABLE=1` and append `--disable-custom-all-reduce`; `enable` must do neither.

Update the environment banner to torch 2.13/vLLM 0.30/CUDA 13 and add selected generator/P2P values to `run-summary.json`.

- [x] **Step 4: Preserve the failed-generation safety test**

Update its fixture to create Qwen3.8's `config.json` and pass `QWEN38_PATH`. Run:

`pytest -q tests/test_run_script.py::test_failed_real_vllm_probe_aborts_before_generation`

Expected: PASS and no counterfactual generation phase.

- [x] **Step 5: Verify all orchestration modes**

Run:

```bash
bash -n run.sh
MODE=smoke bash run.sh --dry-run
MODE=main bash run.sh --dry-run
MODE=matrix bash run.sh --dry-run
GENERATOR_MODEL=qwen36 MODE=smoke bash run.sh --dry-run
```

Expected: every command exits 0; matrix emits exactly 78 training commands.

- [x] **Step 6: Commit orchestration changes**

```bash
git add run.sh tests/test_run_script.py
git commit -m "feat: optimize CUDA 13 Qwen3.8 pipeline"
```

### Task 5: Document fresh Conda installation and complete acceptance

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-legal-landscape-design.md`
- Modify: `docs/superpowers/specs/2026-10-01-cuda13-qwen38-environment-design.md`

**Interfaces:**
- Produces operator commands for a fresh `legal-landscape-cu130` Conda environment and smoke-first operation.
- Records that CUDA 13 GPU execution cannot be proven on the local macOS CPU environment.

- [x] **Step 1: Replace obsolete environment documentation**

Document this fresh environment flow:

```bash
conda create -n legal-landscape-cu130 python=3.12 -y
conda activate legal-landscape-cu130
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
MODE=smoke bash run.sh
```

Explain Qwen3.8 BF16 primary/Qwen3.6 comparison selection, the exact default paths, P2P auto/fallback behavior, CUDA Graph fallback, optional FLA, and the 100-case generator quality guard.

- [x] **Step 2: Update the durable project design**

Replace driver-535/vLLM-0.19/Qwen3.6-primary statements with the approved CUDA 13 stack and Qwen3.8 primary model. Mark the 2026-10-01 environment design status as implemented only after verification succeeds.

- [x] **Step 3: Run complete local verification**

Run:

```bash
python -m compileall -q src scripts
pytest -q
PYTHONPATH=src /opt/anaconda3/envs/myenv/bin/python -m unittest tests.test_models_losses tests.test_training -v
ruff check .
bash -n run.sh
MODE=smoke bash run.sh --dry-run
MODE=main bash run.sh --dry-run
MODE=matrix bash run.sh --dry-run
```

Expected: zero failures, zero Ruff violations, zero shell errors, and 78 matrix training commands.

- [x] **Step 4: Perform plan/spec consistency checks**

Run:

```bash
rg -n 'driver 535|535\.171|CUDA 12\.2|torch==2\.10\.0|vllm==0\.19\.1|/data/chenguo' requirements.txt requirements-optional.txt run.sh README.md scripts configs docs/superpowers/specs
rg -n 'T[B]D|T[O]DO|FIX[M]E' docs/superpowers/specs/2026-10-01-cuda13-qwen38-environment-design.md docs/superpowers/plans/2026-10-01-cuda13-qwen38-environment.md
```

Expected: the first command finds obsolete values only where historical context is explicitly labeled; the second finds no placeholders.

- [x] **Step 5: Commit documentation and acceptance evidence**

```bash
git add README.md docs/superpowers/specs/2026-09-24-legal-landscape-design.md docs/superpowers/specs/2026-10-01-cuda13-qwen38-environment-design.md docs/superpowers/plans/2026-10-01-cuda13-qwen38-environment.md
git commit -m "docs: document CUDA 13 Qwen3.8 workflow"
```
