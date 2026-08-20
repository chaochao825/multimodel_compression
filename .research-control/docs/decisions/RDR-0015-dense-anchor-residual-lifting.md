# RDR-0015: Authorize dense-anchor residual lifting

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the request to continue simple, theory-driven
  probes after a positive function-class result
- Supersedes: none

## Context

EXP-014 passed its registered function-class Gate with a centered K/V rank-64
basis shared by eight query tiles: validation was 0.884% / 1.858% and frozen-
configuration test was 0.977% / 1.958% at 48.39% optimistic work. It also showed
that sixteen-tile sharing is too broad, establishing a finite reuse scale.

The pass is not deployable because each test group re-optimized latent queries
against the dense group defect and obtained coefficients by dense-defect
projection. Training a predictor immediately would not distinguish whether the
basis itself can be recovered from legal runtime information.

One dense 64-query anchor tile per eight-tile group provides an exact runtime
residual `D_anchor = Y_dense - Y_sparse`. Its row space is a content-specific
output basis with no teacher leakage. If that basis transfers, coefficients for
the remaining queries can be lifted from anchor `(Q, coefficient)` pairs by a
small online regression or kernel interpolation.

## Decision

Authorize one two-stage training-free Gate. Stage A compares fixed-first,
Q-medoid, and target-exposed best anchor tiles under oracle coefficients. The
deployment path is frozen to Q-medoid, shared across all heads. Stage B uses
only Q, sparse output, anchor residual, and anchor coefficients to fit a fixed
set of online mean/nearest/softmax/ridge predictors on validation; one family
and hyperparameter tuple is then frozen for test.

Dense non-anchor defects are evaluation-only. The anchor dense calculation is
part of the candidate algorithm and must be included in arithmetic cost.

## Consequences

- A test pass below 50% optimistic work is the first train-free local operator
  witness and may authorize fresh-cell replication before kernel work.
- If anchor basis transfers but all legal coefficient predictors fail, the
  next step is a small frozen-QKV coefficient adapter, not more BCM/support
  search.
- If only target-exposed best anchors pass, the bottleneck is anchor routing.
- If no anchor basis passes, the EXP-014 positive result requires whole-group
  target information and has no anchor-based train-free path.
