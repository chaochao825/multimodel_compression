# RDR-0019: Authorize fresh temporal marginal-defect transport screen

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue high-information,
  theory-driven exploration after valid negative results
- Supersedes: none

## Context

EXP-014 established a target-exposed shared output-subspace witness, while
EXP-015/016/038/039 successively rejected one-anchor observation, equal-budget
distributed observation, static per-head transfer, and same-step LOAO
self-certification. The remaining unexplored source of current-content
information is the denoising trajectory itself.

This is not a generic feature-cache claim. SVD-Cache already predicts the
principal subspace of full DiT features and reuses their residual; SpectralCache
already combines feature-frequency decomposition with temporal scheduling and
cumulative error budgets. The narrower hypothesis here is that recomputing the
current sparse-critical attention branch changes the residual dynamics: the
remaining marginal `AV` defect may have a more coherent output-channel subspace
than either the full feature or a cross-identity static defect basis.

## Decision

Authorize one fresh-identity mechanism screen at Wan F81 Layer 14, conditional
branch, sampling steps 7 through 10. For each causal pair, recompute the
target-exposed 35% exact support and compare a previous-step rank-64 defect
basis, fixed-total-rank previous-plus-current-anchor innovation, a two-step
chordal predictor, a same-step calibration basis, current anchors alone, and a
full-output subspace-reuse baseline.

Validation selects one causal candidate shared over heads and target steps. The
candidate is frozen before the fresh test identity is loaded. Support and all
non-anchor coefficients remain target-exposed, so the Gate tests representation
and transport only.

## Consequences

- A frozen test pass at 1% / 2%, at least 20% improvement over the static
  same-step basis, and at most 50% local optimistic work authorizes a separate
  target-free coefficient and refresh-schedule Gate.
- Failure of the adaptive current rank-64 floor closes marginal low-rank
  capacity on these cells. Failure of causal candidates with a passing adaptive
  floor closes train-free temporal defect transport at the tested rank/support.
- A pass does not establish novelty or deployment. Any continuation must compare
  against paper-faithful SVD-Cache, sparse attention, and dense FP8 on full
  rollout and measured H200 latency.
- No BCM/BCCB/Butterfly rescue, post-outcome rank/support tuning, predictor,
  rollout, kernel, or speed claim is authorized in this Gate.

## Resource boundary

- Four fresh prompt/seed identities, four conditional QKV cells each, one layer,
  and one valid H200 capture plus one valid analysis execution.
- Capture artifacts go to `/opt/data/wangmeiqi`; the nearly full root filesystem
  must not receive the tensor payloads.
- At most four H200-hours total and one pre-outcome engineering repair.

## Pre-run implementation lock

- Config: `89649243019ae789`
- Prompts: `4811d37ed1a25889`
- Capture: `379265f7ba79c235`
- Probe: `ba0fe22259894b36`
- Core: `423a254a4f7cf3b8`
- Core test: `cca170d5af1c6e17`
- Probe test: `fc5f6c4c0e7861ba`
- Capture test: `99430185f1eb8978`
- Sixty-two new and inherited remote tests pass before formal capture.

## Closure

Closed as `adaptive-capacity-null`. Fresh validation adaptive rank 64 reached
1.380% aggregate and 2.829% worst head-step error, so no causal candidate was
selected and staged test remained unopened. Previous-step overlap nevertheless
averaged 0.877 versus 0.646 for static calibration; only a new cost-bounded
heterogeneous rank frontier may use that signal.
