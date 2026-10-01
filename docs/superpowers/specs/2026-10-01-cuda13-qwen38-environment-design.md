# CUDA 13 and Qwen3.8 Environment Realignment

**Date:** 2026-10-01

**Status:** Implemented; local acceptance passed on 2026-10-01; server GPU acceptance pending

**Scope:** Dependency stack, server defaults, Qwen3.8 counterfactual generation,
GPU probes, and operator documentation. Dataset semantics and the Qwen3.5-9B
training objective remain unchanged.

## 1. Target environment

The supported server is:

- Ubuntu 24.04 x86-64;
- Python 3.12 in a fresh Conda environment;
- NVIDIA driver 580.173.02 with CUDA 13.0 capability;
- four allocated RTX 4090 GPUs with physical indices `4,5,6,7`;
- local-only model and dataset files.

Default paths are:

| Resource | Default path |
|---|---|
| Qwen3.8-27B BF16 generator | `/data/cguo/Qwen3.8-27B` |
| Qwen3.6-27B BF16 comparison generator | `/data/cguo/Qwen3.6-27B` |
| Qwen3.5-9B training backbone | `/data/cguo/Qwen3.5-9B` |
| CMDL | `/data/cguo/datasets/CMDL` |
| CAIL2018 | `/data/cguo/datasets/CAIL2018` |

Every path remains overridable through environment variables. The source
datasets and model directories stay read-only.

## 2. Model decision

Qwen3.8-27B BF16 is the primary counterfactual generator. Qwen3.6-27B remains
available as an explicitly selected comparison model and is not silently mixed
with Qwen3.8 outputs. Qwen3.5-9B remains the NF4 QLoRA training backbone.

The primary generator runs with:

- four-way tensor parallelism;
- `--language-model-only` because the task is text-only;
- an 8,192-token serving limit;
- thinking disabled through `chat_template_kwargs.enable_thinking=false`;
- JSON structured output;
- a fixed served-model name and recorded model revision;
- existing retry, validation, resume, and provenance behavior.

BF16 is preferred over FP8 or four-bit weights because this model creates the
paper's derived dataset. Avoiding quantization removes a source of generation
drift and makes the method easier to report. Quantized checkpoints are outside
the primary experiment and may be evaluated separately later.

## 3. Dependency stack

The environment is rebuilt rather than upgraded in place as
`legal-landscape-cu130`. The pinned GPU stack is:

| Package | Version | Reason |
|---|---:|---|
| Python | 3.12 | Existing project and server target |
| torch | 2.13.0 | Required by vLLM 0.30.0 CUDA wheel |
| vLLM | 0.30.0 | Current CUDA 13.0 release with Ubuntu 24.04 artifact |
| transformers | 5.15.0 | Qwen3.8 capable and aligned with the vLLM release line |
| accelerate | 1.15.0 | Current stable Python 3.12 distributed launcher |
| peft | 0.21.1 | Current stable QLoRA adapter stack |
| bitsandbytes | 0.50.0 | CUDA 13 and NF4 support |

The remaining utilities stay narrowly pinned or bounded. vLLM is installed
from its CUDA 13 PyPI wheel; no locally installed CUDA toolkit is used to build
vLLM. `pip check`, exact version checks, a CUDA kernel, BF16 matmul, and an NF4
operation are mandatory gates.

`flash-linear-attention[cuda]==0.5.2` remains optional. It is not part of the
first smoke run. The current `--probe-fla` imports `fla` and runs only a generic
Triton addition kernel; this is a preliminary compatibility probe, not proof
of a working FLA attention kernel. Keep FLA uninstalled/disabled and not
approved for project use until an actual model-relevant FLA operation executes
successfully on the target GPU with the target stack. Neither local tests nor
the current automated probe satisfies that gate. Use a separate validation
environment for preliminary optional-package investigation, and keep the
Transformers implementation as the project fallback until approval. The
`INSTALL_FLA=1` interface triggers installation (when `INSTALL_DEPS=1`) and the
preliminary check; it does not enforce the actual FLA operation gate and must
remain unset for project runs until that gate succeeds.

## 4. Runtime and performance policy

The old driver-535 compatibility comments and defaults are removed. The new
driver natively satisfies CUDA 13 and must not use forward-compatibility
libraries.

The vLLM service defaults to CUDA graphs for performance instead of
`--enforce-eager`. Eager mode remains an environment-variable fallback for
diagnostics. Prefix caching is enabled because counterfactual requests share a
large prompt prefix. The service remains bound to `127.0.0.1`.

RTX 4090 cards have no NVLink, and PCIe peer access depends on motherboard and
IOMMU topology. The environment probe records pairwise peer accessibility.
`P2P_POLICY=auto` enables custom all-reduce and NCCL P2P only when
`gpu-probe.json` contains the JSON Boolean `true` at
`peer_access.all_pairs_accessible`. Missing, malformed, false, numeric, or
string values use `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`.
`P2P_POLICY=disable` forces that fallback; `enable` explicitly overrides the
automatic decision. Auto dry-runs show the fallback pending a real probe.
A launch failure is not silently retried with different numerical settings.

The service must pass both `/health` and a real non-thinking JSON completion
before counterfactual generation starts. The vLLM log tail is printed on
failure.

## 5. Configuration and interfaces

`run.sh` uses Qwen3.8 defaults while retaining an explicit Qwen3.6 selector.
The public interface is:

- `GENERATOR_MODEL=qwen38` by default;
- `GENERATOR_MODEL=qwen36` for the comparison generator;
- `QWEN38_PATH` and `QWEN36_PATH` for the two local checkpoints;
- `INFER_MODEL_NAME` derived from the selected generator unless explicitly
  overridden;
- `VLLM_ENFORCE_EAGER=0` by default, with `1` as a diagnostic fallback;
- `P2P_POLICY=auto|enable|disable`, with auto as the default.

A new `configs/cf/qwen38_27b.yaml` records Qwen3.8 sampling and provenance.
The existing Qwen3.6 configuration remains intact. Qwen3.8 uses deterministic
per-request seeds, temperature `0.2`, top-p `0.9`, JSON mode, and disabled
thinking to preserve continuity with the earlier experiment while model choice
is the only intended generator change.

Generated files retain the actual selected model name and revision. Both
generators are isolated under `${OUTPUT_ROOT}/counterfactuals/${GENERATOR_MODEL}/`
and `${OUTPUT_ROOT}/runs/${GENERATOR_MODEL}/`. The latter includes training,
checkpoints, predictions, and evaluation, preventing accidental resume or
mixing downstream results across models. Processed datasets stay shared under
`${OUTPUT_ROOT}/processed/`. Root-level diagnostics and `run-summary.json`
describe the latest invocation and include its selected generator.

## 6. Validation and acceptance

Local CPU acceptance covers:

- exact dependency declarations and requirement-checker expectations;
- default paths and Qwen3.8/Qwen3.6 selection;
- dry-run smoke, main, and matrix command expansion;
- disabled-thinking JSON payloads;
- safe P2P fallback behavior;
- all existing unit tests, compile checks, Ruff, and shell syntax.

Server-only acceptance is ordered and fail-fast:

1. verify Ubuntu, driver, Python, four visible physical GPUs, and package
   versions;
2. execute CUDA, BF16, NF4, and pairwise P2P probes;
3. start Qwen3.8-27B BF16 with TP4 and complete a real JSON request;
4. run a bounded mock pipeline;
5. run a bounded real Qwen3.8 counterfactual batch;
6. run `MODE=smoke` end to end;
7. start the main experiment only after all smoke gates pass.

Before committing to the full derived dataset, a fixed sample of approximately
100 parent cases should be generated once by Qwen3.6 and once by Qwen3.8 using
the same seeds. Report JSON validity, full validator pass rate, retries, target
factor realization, non-target drift, and near-duplicate rejection. Qwen3.8 is
the declared primary model; this pilot is a quality guard, not post-hoc model
selection on downstream test performance.

The local macOS CPU environment cannot prove CUDA 13 wheel installation,
CUDA/BF16/NF4 kernels, real P2P topology, FLA, CUDA graphs, vLLM TP4, actual
model loading, memory use, throughput, or distributed checkpoint recovery.
These remain server-only acceptance gates. The 100-case quality guard is an
operator-run pilot, not automatically completed by the smoke script.

Local acceptance on 2026-10-01 passed `python -m compileall -q src scripts`,
`ruff check .`, `bash -n run.sh`, and smoke/main/matrix dry-runs. `pytest -q`
reported 101 passed and 10 Torch-dependent skips with the default Python 3.13
interpreter. The required Python 3.12 interpreter at
`/opt/anaconda3/envs/myenv/bin/python` passed all 21 loss/training unittests
with local torch 2.12.1 and no CUDA. The matrix expanded to exactly 78 training
commands. The obsolete-value and plan/spec placeholder scans returned no
matches. These results verify local contracts, not the pinned target GPU stack.

## 7. Error handling and reproducibility

Any package mismatch, unavailable GPU, failed kernel, missing model file,
failed vLLM request, or invalid generator selection terminates before expensive
generation. The run summary records the selected generator, model path, GPU IDs,
P2P policy, mode, experiment/seed selections, timestamps, and exit status.

The Conda environment creation commands and exact installation procedure are
documented in the README. The generated requirements remain the authoritative
project dependency list; operators do not mix the former CUDA 12 environment
with this CUDA 13 environment.
