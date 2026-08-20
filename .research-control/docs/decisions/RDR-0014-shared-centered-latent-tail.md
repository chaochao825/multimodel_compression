# RDR-0014: Authorize one shared centered-latent tail Gate

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue high-information
  theory and method probes rather than abandon the direction after one failure
- Supersedes: none

## Context

EXP-013 showed that rank-conditioned swaps within 35% contiguous-64 support do
not lower the test-exposed quality tier below 61.33% optimistic dense work. The
support objective is therefore no longer the highest-information variable on
this cell.

The inherited tail cost, however, was charged independently for every 64-query
tile. A K/V-generated output basis can be materialized once per attention head
and reused by multiple query tiles. An older three-tile pilot found that shared
adaptive rank 16 was worse than independent rank 16, but test records required
shared rank 32.75 on average and 55 at worst for a 1% record gate. Under a
shared-basis cost model, rank may increase while amortized work decreases.

Adaptive SVD alone is an uninformative endpoint because rank 128 spans the full
output channel space. The useful question is whether a constrained basis made
from `r+1` K/V centroids, one shared anchor plus `r` signed differences, tracks
the shared adaptive subspace at ranks no greater than 64.

## Decision

Authorize one two-stage target-exposed Gate on the same Layer-14 step-9 cell.
First measure shared adaptive rank growth over deterministic nested sets of
2/4/8/16 query tiles. Then optimize a single centered K/V latent basis shared
by 4/8/16 tiles on validation. Select the minimum-work `(M, rank)` satisfying
the registered local quality and adaptive-recovery gates, freeze it, and run
the same function class once on test.

The latent queries and output coefficients read the current dense defect. This
is a function-class and amortization oracle, not a deployable predictor. The
cost model is deliberately optimistic and must be reported with every result.

## Consequences

- A frozen configuration passing test at no more than 55% optimistic work may
  authorize fresh captures and a Q/K/V-conditioned latent-query/coefficient
  predictor with an explicit 5% overhead budget.
- A pass only between 55% and 65% is a representation boundary and does not
  authorize training or kernel work.
- Failure of the centered family while shared adaptive SVD passes identifies a
  basis-generation boundary; failure of both closes shared low-rank tails on
  this cell and assigns it to FP8/BF16 dense fallback.
- This decision does not reopen fixed BCM/BCCB/Butterfly output fitting or
  additional regular-support search.
