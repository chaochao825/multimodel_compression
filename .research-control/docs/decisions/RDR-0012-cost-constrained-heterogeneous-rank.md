# RDR-0012: Authorize a cost-constrained heterogeneous-rank oracle

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue bounded,
  theory-driven experiments when failures add discriminating insight
- Supersedes: none

## Context

EXP-011 found a valid rank boundary rather than a uniform-rank deployment
path. At 5% additional exact support, additive rank 16 reached only
1.514% / 3.147%, whereas rank 32 reached 0.639% / 1.229%. A post-hoc
validation-frozen diagnostic promoted only the hard heads and reached
0.950% / 1.298% at average rank 22.67, but its optimistic centered-tail work
was about 68.5% of dense attention. This suggests stable head heterogeneity,
while leaving unresolved whether intermediate ranks can cross the quality gate
below the 60% work ceiling required before training a generator.

Training a router or Q/K LoRA before answering this capacity-and-cost question
would be expensive and scientifically ambiguous. Conversely, using only ranks
16 and 32 can overstate the cost because the optimal allocation may lie between
them.

## Decision

Authorize one target-exposed, validation-frozen heterogeneous-rank oracle on
the same Layer-14 step-9 cell. For base-only support and the three EXP-011
expanded-support budgets, evaluate adaptive ranks 0 through 44 in increments
of four plus exact dense fallback. Use the EXP-011 validation-selected support
selector at each budget. Select a per-head allocation on validation under an
explicit optimistic work model, then execute the frozen density, selector, and
head allocation once on the independent test identity.

The work model is a necessary-condition model for a centered shared-anchor
tail:

```
work(delta, r) = 0.25 + delta
               + ((r + 1) * 144 + r * 128) / (2 * 64 * 128)
```

for `r > 0`; rank zero pays only exact support, and dense fallback costs 1.0.
It excludes routing, coefficient prediction, normalization, launch, memory,
and fusion overhead. Therefore a pass only authorizes a fresh-data, QKV-frozen
adaptation Gate; it is not a speed result.

## Consequences

- A test deployment pass at no more than 60% optimistic work permits a fresh
  prompt/seed adaptation Gate for a pooled-Q/K/V router, centered latent basis,
  shared normalization, and heterogeneous dense fallback.
- A validation pass that fails to transfer identifies head-role instability
  and requires fresh-data certification before any training claim.
- Failure below 60% parks posterior low-rank sparse-linear acceleration on this
  cell. Future progress must change the defect or use dense FP8/BF16 execution,
  rather than add another static support or rank heuristic.
