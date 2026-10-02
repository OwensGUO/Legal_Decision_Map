# Smoke Training Memory Design

## Context

`MODE=smoke bash run.sh` reaches the real Qwen3.5-9B QLoRA training path. The smoke
mode currently reduces dataset limits and optimizer steps, but it retains the production
sequence lengths: 4096 tokens for CAIL and 8192 tokens for CMDL. Experiment `M` performs
the ordinary-case, counterfactual-parent, and counterfactual-child forward passes before
backpropagation. On a 24 GiB GPU, these retained computation graphs can exhaust memory even
with NF4 quantization, batch size 1, and gradient checkpointing.

The observed failure occurred during the third forward pass. A data-parallel Accelerate
process loads one complete model on each GPU, so using four GPUs does not combine their memory
for one process.

## Goal

Make smoke mode a reliable, low-resource end-to-end validation while preserving the sequence
lengths used by `main` and `matrix` experiments.

## Approaches Considered

1. **Use a smoke-only sequence-length limit (recommended).** Default smoke training to 1024
   tokens and allow an environment-variable override. This directly reduces activation memory,
   keeps experiment `M` and all three forward paths active, and leaves production experiments
   unchanged.
2. **Run smoke with experiment `B3`.** This avoids counterfactual pair forward passes, but it no
   longer validates the full `M` training path and would weaken the smoke test.
3. **Restructure loss computation and backward passes.** This could reduce peak graph retention
   at production lengths, but it changes training semantics and is disproportionate to the
   purpose of a one-step smoke run.

## Design

- Add `SMOKE_MAX_LENGTH`, defaulting to `1024` only when `MODE=smoke`.
- In `max_length_for_experiment`, keep the existing B1 limit of 512. For all other real-model
  experiments in smoke mode, return `SMOKE_MAX_LENGTH` for both CAIL and CMDL.
- Keep current non-smoke behavior unchanged: B1 uses 512, B2 uses 4096, Qwen experiments use
  4096 for CAIL and 8192 for CMDL.
- Validate `SMOKE_MAX_LENGTH` as a positive integer when smoke mode is selected. Invalid values
  must fail before launching expensive pipeline work and must produce a clear error.
- Document the new default and the override syntax in `README.md` and the `run.sh --help` output.

## Data Flow

In smoke mode, the environment value is resolved once during mode initialization. Each training
command obtains its maximum length through `max_length_for_experiment`, then passes the effective
value as the existing `--set model.max_length=...` override to `scripts/train_model.py`. No Python
training interface changes are required.

## Tests

Tests will exercise the real `run.sh --dry-run` command construction and verify:

- smoke mode passes `model.max_length=1024` for both datasets;
- `SMOKE_MAX_LENGTH` overrides the smoke default;
- main mode retains 4096 for CAIL and 8192 for CMDL;
- an invalid smoke override fails with a clear validation message;
- help and README text expose the new setting.

The focused shell-script tests will be run first, followed by the full automated test suite and
Bash syntax validation.

## Non-goals

- Changing production training sequence lengths.
- Changing QLoRA, gradient-checkpointing, or distributed-training semantics.
- Guaranteeing that every possible 1024-token configuration fits every GPU; the override remains
  available for machines with different memory constraints.
