# Staged Counterfactual Backward Design

## Context

The real `MODE=smoke` run still exhausts a 24 GiB GPU after reducing the training sequence
length from 4096 to both 1024 and 512 tokens. A controlled run with the same model, data limits,
precision, quantization, and 512-token limit succeeds when `EXPERIMENTS=B3` is selected.

The differentiating behavior is the `M` training loop. It currently performs the ordinary-case,
counterfactual-parent, and counterfactual-child forwards before a single backward call. All three
autograd graphs therefore coexist. B3 performs only the ordinary-case forward and passes.

## Goal

Reduce experiment `M` peak training memory without changing its mathematical objective, optimizer
step count, gradient-accumulation schedule, loss weights, or output formats.

## Approaches Considered

1. **Stage the backward passes (recommended).** Backpropagate the supervised loss immediately,
   release its graph, then construct and backpropagate the paired counterfactual loss. Since
   differentiation is linear, the accumulated parameter gradient is identical to differentiating
   their sum, while at most the two pair graphs coexist.
2. **Keep reducing the sequence length.** Both 1024 and 512 already fail for `M`; this weakens the
   smoke workload without addressing simultaneous graph retention.
3. **Skip typed counterfactual training in smoke mode.** B3 proves this would run, but it would stop
   smoke mode from testing the full `M` path.
4. **Offload saved autograd tensors to CPU.** This preserves semantics but adds transfer overhead
   and complexity. It remains a fallback only if two pair graphs still exceed the target GPU.

## Design

Within each existing `accelerator.accumulate(predictor)` block:

1. Run the ordinary-case forward.
2. Compute only the supervised charge, article, sentence, and factor losses.
3. Validate that loss breakdown and call `accelerator.backward` immediately.
4. Store detached scalar loss values for reporting, delete references to the ordinary output and
   loss graph, and proceed to the pair batch.
5. When a counterfactual pair batch exists, run parent and child forwards, compute only invariant,
   boundary, and response losses, validate them, and call `accelerator.backward` a second time.
6. Add the detached supervised and paired loss values for progress, JSONL, and TensorBoard output.
7. Execute `optimizer.step()` and `optimizer.zero_grad()` exactly once, in their current positions.

B3 and experiments without typed pairs continue to perform one forward and one backward. The
existing Accelerate accumulation context remains the sole authority for deciding when gradients
synchronize and when an optimizer step counts as complete.

## Gradient Semantics

For parameters `theta`, supervised loss `Ls`, and paired loss `Lp`, the current update uses:

```text
gradient = d(Ls + Lp) / d(theta)
```

The staged implementation uses:

```text
gradient = dLs / d(theta)
gradient += dLp / d(theta)
```

These are mathematically equivalent. Both backward calls remain inside the same accumulation
context and occur before the single optimizer step. No loss is detached before its own backward;
only reporting snapshots are detached.

## Failure Handling

Each partial loss is validated before its backward call. Invalid supervised or paired losses write
the existing `loss_diagnostic.json` format and abort before the optimizer step. Pair sampler state
advances only after both pair forwards and paired loss construction succeed, matching the current
successful-batch behavior.

## Tests

- A real small Torch model compares parameter gradients from combined backward and staged backward
  and requires numerical equality.
- The training-loop fixture verifies experiment M performs two backward calls per microbatch but
  one optimizer step, while B3 still performs one backward call.
- Loss reporting tests verify the logged total and named components are the sum of detached staged
  breakdowns.
- Existing checkpoint, progress, prediction, and full-suite tests must remain green.
- Server acceptance first runs `SMOKE_MAX_LENGTH=512 MODE=smoke`; only after it passes should the
  default 1024-token smoke configuration be tested.

## Non-goals

- Changing formal sequence lengths or the smoke length default.
- Changing the counterfactual objective or dropping gradients through either pair branch.
- Adding CPU autograd offload before staged backward is tested on the target server.
