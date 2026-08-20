# RDR-0011: Authorize a bounded block-innovation oracle

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue theory-driven
  experiments when failures provide additional insight
- Supersedes: none

## Context

EXP-010 closed regular-support rank shaping as a valid null. Target-exposed
mass-plus-spectral-swap improved the 25% support adaptive rank-16 error from
2.120% / 3.872% to 1.803% / 2.986%, but no density through 25% reached the
1% / 2% deployment boundary. Rank 32 reached 0.791% / 1.275% at 25%, showing
that the remaining obstacle is a high-dimensional value innovation rather than
basis generation alone.

Simply adding more support, rank, rotations, or fixed structure would repeat a
closed function class. A different exact identity remains untested. For base
support output `Y_S = N_S / Z_S`, the omitted output defect decomposes as

```
Y - Y_S = sum_{b not in S} (N_b - Y_S Z_b) / Z_all.
```

Each term is a hardware-regular block atom and may be high rank. A small number
of such atoms can therefore correct directions that no rank-16 matrix can
represent, while preserving a regular exact-compute path.

## Decision

Authorize one target-exposed side probe on the same Layer-14 step-9 cell. Keep
the 25% pooled-Q/K support fixed, add at most 2.5%, 5%, or 10% contiguous-64
innovation blocks, and compare mass, atom-energy, and residual-tail-greedy
selection. The selected exact atoms are added under the shared dense partition,
then an adaptive rank-16 tail is recomputed. Compare against simply expanding
and renormalizing the sparse support with the same blocks.

Validation selects one innovation family per budget; the frozen family is run
once on the independent test identity. Dense targets and exact partition are
permitted only because this Gate measures the function-class ceiling. No
selector, partition estimator, latent generator, rollout, kernel, or speed
claim is authorized.

## Consequences

- A strict pass permits a separate Gate for a Q/K/V-sketch innovation selector
  and centered signed latent basis with frozen QKV.
- A 1% / 2% boundary permits a low-cost adaptation comparison but not a kernel.
- A null parks bounded block innovation on this cell and assigns it to dense
  FP8/BF16 fallback unless a future learned-representation result changes the
  defect itself.
