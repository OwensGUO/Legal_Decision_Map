# Single-Graph Pair Recompute Design

## Context

The staged-backward implementation successfully releases the ordinary supervised graph before
counterfactual training, but a real 24 GiB GPU still exhausts memory during the counterfactual
forward. The new traceback points to `cf_output = predictor(...)`: the parent graph is resident,
and allocating the second pair graph exceeds memory even at 512 tokens. B3 succeeds under the same
model and length, establishing that one graph fits while two do not.

## Goal

Train the full typed-counterfactual objective while allowing at most one full Qwen autograd graph
to reside on a GPU at any time. Preserve gradients through both parent and counterfactual branches,
the existing optimizer/accumulation schedule, loss values, and generated counterfactual texts.

## Approaches Considered

1. **Single-graph branch recomputation (recommended).** Evaluate the parent once without a graph,
   differentiate the loss through the counterfactual graph and small detached parent-output leaves,
   then recompute the parent and inject the captured output gradients. This preserves the full
   objective with one resident model graph at the cost of one extra parent forward.
2. **Detach the parent branch permanently.** This fits memory but removes parent-side gradients from
   invariant and ranking losses, changing the learning objective and potentially model quality.
3. **CPU-offload saved autograd tensors.** This keeps one parent forward but adds large transfers,
   depends more heavily on autograd internals, and is less predictable than explicit recomputation.
4. **Continue reducing sequence length.** M already fails at 512 while B3 passes; the evidence
   isolates pair-graph concurrency rather than input length as the remaining bottleneck.

## Training Algorithm

The existing supervised stage remains unchanged: ordinary forward, supervised loss, immediate
backward, detached logging snapshot, and graph release.

For a typed pair batch:

1. Capture the current CPU RNG state and the RNG state for the pair input's CUDA device.
2. Run the parent through the wrapped predictor under `torch.no_grad()`.
3. Detach the parent charge logits and sentence prediction, mark those small tensors as gradient
   leaves, and release all parent model outputs. No parent model graph remains.
4. Run the counterfactual through the predictor with gradients enabled.
5. Compute the pair loss using the parent leaves and live counterfactual outputs. Add a zero-valued
   graph anchor for otherwise-unused counterfactual heads so DistributedDataParallel observes every
   predictor parameter in this backward.
6. Call `accelerator.backward` on the pair loss. This accumulates counterfactual model gradients and
   produces loss gradients on the parent leaves, already carrying Accelerate's accumulation/scaler
   factor.
7. Detach and preserve the parent-leaf gradients, then release the counterfactual graph.
8. Temporarily restore the RNG snapshot, recompute the parent with gradients, and backpropagate the
   preserved output gradients through it. Include zero gradients for unused parent heads so the DDP
   reducer completes. Restore the post-counterfactual RNG state when recomputation exits.
9. Execute the existing single optimizer step and zero-grad calls after both branches.

## RNG and Dropout Correctness

The QLoRA configuration uses LoRA dropout 0.05. The first graph-free parent evaluation and the
gradient-bearing recomputation must therefore use identical random masks. A scoped RNG-replay
context will:

- snapshot CPU and current-device CUDA state before the first parent evaluation;
- save the current post-counterfactual states when recomputation begins;
- install the original parent states for recomputation;
- restore the post-counterfactual states on exit, including exceptional exits.

This reproduces the parent output used to compute the loss without rewinding randomness for future
microbatches. RNG state is handled per DDP process and only for that process's model device.

## Gradient Semantics

Let `p(theta, rng)` be parent outputs, `c(theta)` counterfactual outputs, and `L(p, c)` pair loss.
The algorithm first obtains `dL/dc` and `dL/dp` while backpropagating through `c(theta)`, then
recomputes the same `p(theta, rng)` and applies `dL/dp`. The accumulated model gradient is:

```text
(dL/dc)(dc/dtheta) + (dL/dp)(dp/dtheta)
```

which is the chain-rule gradient of the original joint graph. No parent gradient is dropped.

## DDP and Mixed Precision

`accelerator.backward` applies gradient-accumulation scaling and any active mixed-precision loss
scaling before the parent-leaf gradients are read. Those already-scaled output gradients are passed
directly to `torch.autograd.backward` during parent recomputation, so they must not be scaled a
second time. Zero anchors/zero output gradients ensure all predictor heads participate in each DDP
backward without changing numerical gradients.

## Quality Impact

Phase-6 counterfactual generation is untouched: model, prompts, sampling, validation, and stored
texts do not change. The downstream training objective is preserved. Expected differences are
limited to ordinary BF16/CUDA floating-point effects; the intended trade-off is additional compute
time rather than reduced data or model quality.

## Tests

- RNG replay with a dropout-bearing tiny model reproduces parent outputs and restores the global RNG
  stream after recomputation.
- A real tiny Torch model compares original joint-graph gradients with single-graph recomputation
  gradients for both parent and counterfactual branches.
- Fake-runtime orchestration verifies the first parent call is graph-free, the parent is recomputed,
  pair loss is backpropagated once, parent output gradients are injected once, and the optimizer is
  called once.
- Boundary-only pair batches skip parent recomputation when both parent-output gradients are absent.
- Existing supervised staging, loss logging, progress, checkpoint, prediction, and full-suite tests
  remain green.
- Server acceptance uses `SMOKE_MAX_LENGTH=512` first and default 1024 only after 512 passes.

## Non-goals

- Modifying generated counterfactual text or its quality controls.
- Changing loss definitions, weights, sequence-length defaults, checkpoint schemas, or prediction.
- Adding CPU activation offload unless single-graph recomputation still fails on the target server.
