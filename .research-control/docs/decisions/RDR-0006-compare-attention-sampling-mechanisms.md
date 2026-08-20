# RDR-0006: Authorize comparative attention-sampling diagnostics

- Status: accepted
- Date: 2026-08-11
- Decider: researcher through the explicit request to defer engineering and
  validate the method in multiple ways against same-type alternatives
- Supersedes: none

## Context

EXP-005 refuted one tile-level centroid plus PPS control-variate estimator. It
did not identify whether failure came from 64-query proposal sharing,
with-replacement duplicates, numerator error, partition-function error, or the
entire block-sampling function class. Several related methods avoid one or more
of these restrictions: HyperAttention separates heavy entries and residual
sampling; MagicPIG and LARA use query-dependent proposals; KDEformer treats
partition estimation separately; VSA, SpargeAttn, and SLA use content-dependent
critical support rather than a frozen tile proposal.

## Decision

Authorize one bounded, offline comparative mechanism probe on the same four
frozen Wan F81 Layer-14/step-9 captures. It may use dense contributions for
clearly labelled target-leaking ceilings and error-factorial diagnostics. It
must compare equal registered block work and report both arithmetic work and
64x64 execution-union inflation.

This decision authorizes one new script, tests, one execution, one pre-outcome
engineering repair, and one analysis. It does not authorize model generation,
training, threshold tuning on held-out outcomes, rollout, a CUDA/H200 kernel,
or a measured speed claim.

## Consequences

- Peer-inspired methods are mechanism-aligned proxies, not paper-faithful
  reproductions and not direct quality/speed comparisons to published systems.
- A positive per-query oracle with dense 64x64 union is a granularity boundary,
  not deployability evidence.
- An exact-Z-only success is a partition boundary and can motivate a separate
  KDE/normalization Gate.
- Failure of all target-leaking variants through 50% work stops the block
  sampling family under the strict local objective.
