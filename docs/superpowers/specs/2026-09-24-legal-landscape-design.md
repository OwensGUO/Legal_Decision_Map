# Conditional Legal Decision Landscape Design

## Objective

Build a Python 3.12 research codebase for typed counterfactual legal decision
landscape learning. The repository must be testable without GPUs or model
weights, while the same interfaces must support Qwen3.6-27B generation and
Qwen3.5-9B QLoRA training on GPUs 4, 5, 6, and 7 of the target eight-RTX-4090
server.

The fixed server constraints are Ubuntu 20.04, NVIDIA driver 535.171.04, and a
locally installed CUDA 12.2 toolkit. The driver cannot be upgraded. Every newer
CUDA runtime or JIT kernel must therefore pass an executable server probe rather
than being accepted from version numbers alone.

## Scope and data policy

- CAIL-small maps to `CAIL2018/exercise_contest/{data_train,data_valid,data_test}.json`.
- CMDL-small maps to `CMDL/small/{train,valid,test}_small.jsonl`.
- CMDL-big is ineligible unless complete official train, validation, and test
  files are detected. The currently observed lone `train_big-002.jsonl` is not
  sufficient.
- Source datasets are read-only. Derived artifacts retain source paths and
  stable case/group identifiers and are written only under configured output
  directories.
- Multi-label charges are preserved. Life and death sentences never receive
  fabricated month values.

## Architecture

### Configuration and command safety

YAML files provide server defaults. Environment variables override YAML and
explicit CLI arguments override both. Heavy operations require an explicit
execution flag; otherwise generation and training commands perform bounded
dry-runs. Every command exposes help and bounded `--limit` behavior.

### Domain and data layer

Frozen dataclasses define `CaseUnit`, `LegalFactors`, and `InterventionSpec`.
CAIL and CMDL adapters normalize source rows into `CaseUnit` objects. CMDL
creates one unit per target defendant but reuses the same `group_id` for every
defendant from a case. Stable IDs derive from dataset, split, source path, and
line number. A manifest records path, size, SHA-256, and line count.

The sanitizer produces raw, conservative, and strict variants plus structured
redaction spans. Conservative mode removes outcome-bearing court/prosecution
phrases; strict mode additionally masks charge strings, cited articles, and
sentencing recommendations. Sanitization never mutates source files.

### Factors and counterfactual generation

Rule-based factor extraction supplies deterministic seed annotations for
amount, surrender, restitution, confession, and role. Validation checks types
and ranges. Interventions are constructed from factors first and carry an
explicit type, changed fields, rule ID, target charge, and conditional rank
direction.

A common generator protocol has two implementations: a deterministic CPU mock
and an HTTP client for a local vLLM OpenAI-compatible endpoint. The prompt
instructs Qwen3.6 only to realize the supplied intervention as text, disables
thinking, and requests JSON. Validators check schema, target edits, non-target
preservation, identity, label/sentence leakage, and near duplication. JSONL
output records provenance, prompt/model revisions, sampling, seed, raw
response, retries, and validation details; resume mode skips completed IDs.

vLLM serves the multimodal Qwen3.6 checkpoint in language-only mode on the four
allocated GPUs and listens on loopback by default. Conservative defaults
disable custom all-reduce, peer-to-peer NCCL, FlashInfer sampling, and CUDA
graphs. Startup succeeds only after both the health endpoint and a real JSON
chat-completion probe succeed. The script never enables vLLM's CUDA
forward-compatibility library on GeForce RTX GPUs.

### Prediction, losses, and training

The predictor accepts any text backbone returning a pooled representation.
Separate heads produce multi-label charge logits, penalty-type logits,
charge-conditional sentence predictions, and factor logits. Charge probabilities
remain independent sigmoid probabilities for multi-label prediction. Sentence
months use a separate normalized mixture formed from those sigmoid values;
only the mixture weights must sum to one. Hard-charge mode is available only
for ablation A5.

Loss routing is explicit. Original examples drive charge, penalty/sentence,
and factor supervision. Invariant pairs drive stability only; charge-flip
pairs drive the boundary loss and never charge consistency; sentence-rank
pairs drive conditional ranking. Every component returns zero when it has no
eligible examples, and the trainer raises with a diagnostic JSON file on NaN
or an impossible non-zero empty loss.

The real-model loader reads `config.json`, including nested text configuration,
before choosing a Transformers auto class. It distinguishes a text-only
`ForCausalLM` checkpoint from a multimodal `ForConditionalGeneration`
checkpoint. For the latter it loads the correct architecture, extracts the
language backbone, and freezes or discards visual modules before attaching the
task heads. The loader uses SDPA by default, optional 4-bit NF4 QLoRA, BF16,
gradient checkpointing, Accelerate-compatible distributed execution, and does
not request every layer's hidden states when only the last hidden state is
needed.

Checkpoint state includes the optimizer, scheduler or Accelerate state, random
number generators, epoch, sampler position, completed microbatches, model
configuration summary, dependency versions, and input manifest identity.
Resuming reproduces the same remaining sample order rather than constructing a
fresh shuffled order and merely skipping a count. A dummy backbone supports CPU
tests and train dry-runs.

### Evaluation

Static metrics cover multi-label macro/micro F1 and exact match plus sentence
MAE, log-MAE, and tolerance accuracy on eligible finite-term examples only.
Counterfactual metrics cover flip accuracy, invariant and monotonicity
violations, sentence pair accuracy, and non-target drift. CMDL aggregation
reports defendant and case levels and strata by defendant count. Paired
bootstrap resamples whole `group_id` clusters, defaults to 2,000 iterations,
and reports percentile 95% intervals. A reference-prediction input enables
group-paired model comparisons, raw p-values, and Holm-adjusted p-values for
five explicitly named primary endpoints. Without a reference input, evaluation
reports estimates and confidence intervals but does not invent comparison
p-values.

## Dependency and compatibility policy

The supported installation is one already activated Conda environment using
Python 3.12. The reproducible primary stack is vLLM 0.19.1, PyTorch 2.10.0,
Transformers 5.5.3, Accelerate 1.13.0, PEFT 0.18.1, and bitsandbytes 0.49.2.
vLLM's manylinux 2.31 wheel is used unchanged and supplies its matching PyTorch
stack. The project does not attempt an implicit source build.

Driver 535 supports the CUDA 12 major-family compatibility floor, but newer PTX
or JIT-generated kernels may still fail. Installation success is therefore not
acceptance. Before model work the server checks a CUDA tensor operation, BF16,
the four visible devices, bitsandbytes NF4 capability, and dependency versions.
The vLLM phase additionally performs a real text-generation request. Failures
stop with diagnostics that include the driver, PyTorch CUDA runtime, visible
GPU mapping, and relevant service log tail.

`flash-linear-attention` is an optional acceleration extra, not a core
requirement. Its Triton kernels must pass a dedicated probe before use. Without
it, Transformers uses its supported fallback path. This preserves installability
on driver 535 and complies with the requirement that fast attention kernels not
be mandatory.

## Testing and acceptance

Tests use tiny JSONL fixtures, the deterministic mock generator, and a dummy
PyTorch backbone. Tests must prove split integrity, sentence eligibility,
typed loss routing, rank direction, probability normalization, manifest
hashing, sanitizer auditability, validator behavior, metrics, bootstrap
clustering and paired inference, configuration precedence, environment failure
modes, and CLI safety. Final local acceptance
runs compileall, pytest, Ruff, every CLI help path, and CPU-only configuration
checks. GPU kernels, 4-bit loading, vLLM serving, and full training remain
explicitly unverified until run on the server, where the smoke pipeline is a
required gate before `main` or `matrix`.

## Error handling and observability

Input errors identify the source file and line. Generation failures are
recorded and retried only up to the configured bound. JSONL writes are atomic
per record and resumable. Training logs JSONL and TensorBoard locally. No
external API or account service is used.

## One-command Conda orchestration

`run.sh` requires an already activated Conda environment with Python 3.12. It
installs or changes dependencies only when `INSTALL_DEPS=1` is explicitly set;
`INSTALL_FLA=1` separately opts into the optional linear-attention accelerator.
The default physical GPU allocation is exactly `4,5,6,7`, exposed to vLLM and
Accelerate through `CUDA_VISIBLE_DEVICES` while child processes use local ranks
0 through 3.

The pipeline audits the environment and data, builds both small datasets,
starts Qwen3.6 vLLM on the four allocated GPUs, performs a real generation
probe, generates resumable typed counterfactuals, stops vLLM to release GPU
memory, trains, exports static and counterfactual predictions, runs clustered
bootstrap evaluation, and writes a machine-readable run summary. `smoke`
bounds every stage and is mandatory on a new server environment; `main` runs
the declared primary experiment set; `matrix` expands to all baselines,
ablations, and three seeds. Signal and error traps stop only the vLLM process
group started by this script.
