# RDR-035: Accept an exact warm-service runtime before approximate kernels

- Status: accepted
- Date: 2026-09-01
- Decider: researcher through the explicit goal to retain the quality-passing
  released rCM endpoint and next combine exact system optimization, dense FP8,
  and fused attention in measured order
- Supersedes: PLAN-056 successor-pending state only

## Context

EXP-047 established the released four-step rCM model as a high-quality H200
incumbent. It reached `10.135x` denoiser speedup and passed every registered
quality and diversity guard, but reached only `2.181x` warm end-to-end speedup.
After NFE reduction, its `3.177s` denoiser is no longer the dominant cost:
text encoding, VAE decode, and serialization consume about `22.55s`.

The frozen EXP-047 harness calls `clear_umt5_memory()` after every request.
The official rCM helper therefore destroys the global UMT5 encoder and reloads
it on the next request. This is valid for a memory-constrained batch script but
is not the intended lifetime of a warm H200 service. It is also an exact
optimization opportunity: model weights can remain resident while every new
positive prompt is still encoded normally. The fixed negative prompt used by
both native UniPC controls may be encoded once and reused exactly.

Existing measurements make dense FP8 attention a secondary rather than first
candidate. F81 FA3 FP8 attention reached `1.51x` locally, but self-attention is
only about `53.88%` of the now `3.177s` rCM denoiser, so its optimistic warm
end-to-end saving is only about `0.59s`. Exact runtime overhead must move first
if the `2.5x` target is to become reachable.

## Decision

Accept L-030 / C-030 / EXP-052 / G-031 as the new mainline integration Gate.
Test only a persistent UMT5 lifetime and exact fixed-negative embedding reuse:

- each distinct positive prompt must still execute the complete tokenizer and
  UMT5 forward;
- no positive-prompt embedding cache is permitted in the primary endpoint;
- teacher20, native4, and rcm4 receive the same resident-runtime policy;
- the Wan, rCM, VAE, scheduler, attention backend, dtypes, prompts, seeds, and
  serialization path remain unchanged from EXP-047;
- embedding and generated-video equality must pass before F81 timing;
- model initialization is reported separately and excluded only under the
  explicitly named warm-service contract.

EXP-052 is staged. A text-only four-distinct-prompt screen first determines
whether avoiding UMT5 reconstruction can save the `5.5s/request` common-cost
threshold implied by the EXP-047 Amdahl equation. Only a positive screen may
advance to exact F17 generation and then the matched three-method F81 Pareto.

## Consequences

- L-030 becomes the sole active mainline; L-026 remains the immutable measured
  incumbent rather than an active experiment.
- Failure of exact equality, H200 memory coexistence, or the text-time screen
  closes EXP-052 before expensive F81 generation.
- A passing exact runtime becomes the required baseline for any later FP8 or
  attention-fusion Gate. FP8 may not be compared against the older reload-per-
  request teacher while rCM uses the resident runtime.
- A positive result is a systems result, not a new rCM algorithm or quality
  claim. The EXP-047 quality evidence transfers only because the computation is
  required to remain exact.
- BCM, BCCB, Butterfly, and post-hoc low-rank whole-block residuals remain
  closed. They are not revived by this runtime decision.

## Revisit condition

Revise the ordering only if the resident text screen saves less than `5.5s`
per request, cannot coexist with the frozen H200 model, or changes any output.
The next decision must then use a measured VAE/serialization profile before
opening an approximate denoiser kernel Gate.
