# RDR-0013: Authorize one support-manifold cost-shaping oracle

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue bounded,
  theory-driven experiments when failures add discriminating insight
- Supersedes: none

## Context

EXP-012 established that intermediate per-head ranks are useful, but the
validation-frozen deployment allocation crossed from 0.991% / 1.802% to
1.191% / 2.125% on test at 58.57% optimistic work. More importantly, a
post-hoc test-exposed allocation still needed 61.33% work. A uniform four-rank
safety margin passed quality only at 65.21%, leaving no credible runtime margin.

The support used by EXP-012 was inherited from EXP-011: mass or atom energy
selected blocks for a uniform-rank additive experiment. It was never optimized
for the heterogeneous tail cost that now determines the decision. This is a
narrow revival condition for regular support, not permission to enumerate more
geometric patterns.

## Decision

Authorize one target-exposed function-class oracle at exactly 35% contiguous-
64 support. For every registered rank action, start from the EXP-012 pooled-Q/K
plus mass support and perform a fixed bounded gradient-guided swap search that
minimizes that action's post-rank residual. Compute the exact validation Pareto
allocation, freeze the twelve rank actions, and execute them once on test while
allowing only the same target-exposed rank-conditioned support algorithm.

This asks whether support can shape the residual manifold enough to lower tail
rank. It differs from maximizing attention mass or reconstructing the complete
attention matrix. It remains an oracle: dense targets guide every swap, and no
router, basis generator, rollout, kernel, or speed claim is authorized.

## Consequences

- Passing at no more than 55% optimistic work permits a fresh-data QKV-frozen
  adaptation Gate with explicit overhead budget.
- Passing only between 55% and 60% is a representation boundary; a subsequent
  generator must first demonstrate lower feature cost before kernel work.
- If even test-exposed rank adaptation needs more than 60%, park support-
  manifold shaping on this cell and use dense FP8/BF16 fallback.
