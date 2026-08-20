# RDR-0010: Authorize a rank-shaped regular-support oracle

- Status: accepted
- Date: 2026-08-11
- Decider: researcher through the explicit request to continue bounded,
  theory-driven experiments when failures provide a new mechanism insight
- Supersedes: none

## Context

EXP-009 changed the bottleneck. Its signed K+V softmax-difference basis recovered
99.02% of adaptive captured defect energy at rank 16, but adaptive rank 16 itself
was only 2.120% aggregate / 3.872% worst-head. Rank 32 approached quality at
1.148% / 1.880% but required 106.25% of dense QK+AV in an optimistic cost model.

The current 25% support was selected by pooled Q/K relevance. It was never
optimized to leave a low-rank output defect. Continuing to change the basis on
that fixed target has low information value. The remaining structural question
is whether regular sparse support can absorb high-rank value events so the
residual becomes rank-16 compressible.

## Decision

Authorize one target-exposed `side_probe` on the same validation and test
captures. Supports remain unions of contiguous 64-token blocks. Compare pooled
Q/K proxy, exact attention mass, dense-reference value influence, and a bounded
gradient-guided discrete swap oracle that directly minimizes post-rank-16
residual energy.

Evaluate fixed densities 12.5%, 18.75%, and 25%. Select one swap initialization
per density on validation, then execute the frozen algorithm once on the test
identity. Target exposure is permitted only because this Gate measures the
support-family ceiling; no selected support is deployable.

This decision permits one implementation, tests, one execution, one pre-outcome
engineering repair, and one report. It does not authorize a router, latent
generator, QKV tuning, rollout, kernel, or speed claim.

## Consequences

- A strict pass at density at most 18.75% permits a centered/shared-anchor
  latent-basis Gate with a transferable support proxy.
- A pass only at 25% is a density boundary and requires a lower-cost one-pool
  basis before deployment work.
- If no regular support reaches adaptive rank-16 1% / 2% even at 25%, park
  support-plus-rank-16 correction on this cell and move to learned sparse-linear
  routing or dense FP8.
