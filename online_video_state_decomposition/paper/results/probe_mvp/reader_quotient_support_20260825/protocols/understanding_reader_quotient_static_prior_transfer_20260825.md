# Reader-Quotient Static-Prior Transfer Protocol

Date: 2026-08-25
Status: frozen follow-up authorized by the support-oracle `GO`

## Decision question

Can a calibration-only, query-agnostic native-reader sensitivity prior recover
a useful fraction of the transductive Fisher-support oracle while preserving
the exact `PCA-r256 + s4` state payload?

## Split

- Calibration: 40 samples, eight per task, from the old formal tasks
  `object_existence`, `state_change`, `scene_transition`, `action_sequence`,
  and `moving_direction`.
- Evaluation: 40 samples, eight per task, from the five transfer tasks.
- All 40 samples used by the preceding transductive oracle are excluded before
  evaluation sample selection.
- Calibration may estimate one fixed Fisher prior. Evaluation samples cannot
  update, select or normalize that prior using reader gradients.

## Prior and equal-rate candidates

For calibration Fisher diagonals `G_n`, estimate

\[
\bar G_{p,d}=\frac{1}{N F}\sum_{n,f}G_{n,f,p,d},
\]

where `p` is the `8x8` visual-token position and `d` is the native hidden
channel. Evaluation support scores use only the current codec residual and the
frozen prior.

1. `euclidean_s4`: existing residual energy.
2. `position_s4`: Euclidean energy times the position marginal of `G_bar`.
3. `channel_s4`: channel marginal weighted residual energy.
4. `separable_s4`: outer product of position and channel marginals.
5. `static_fisher_s4`: full fixed `G_bar` weighted residual energy.
6. `mixed_static_s4`: equal normalized mixture of Euclidean and static-Fisher
   scores.

Every candidate stores the same four FP16 residual vectors and four int16
indices per frame. The prior is global model metadata and its byte size must be
reported separately.

## Endpoints and decisions

Primary endpoint: aggregate candidate-distribution KL reduction relative to
`euclidean_s4`. Secondary endpoints are full-vocabulary KL, P95 candidate KL,
candidate top-1 agreement, support overlap, task consistency and scorer cost.

- `GO`: one preregistered full/static-mixed method reduces aggregate candidate
  KL by at least `25%`, has P95 ratio at most `1.0`, and top-1 agreement drops
  by no more than `1pp`.
- `WEAK`: reduction is `10%` to `25%` with no tail or top-1 regression.
- `NULL`: reduction is below `10%`.
- `ADVERSE`: aggregate or P95 KL worsens by more than `10%`, or top-1 agreement
  falls by more than `1pp`.
- `INVALID`: sample overlap, evaluation-time gradient use, payload mismatch,
  non-finite metrics or changed full-feature logits.

`GO` authorizes strong-reader replication and low-rank/quantized prior storage.
`WEAK` authorizes at most one query-conditioned low-cost scorer. `NULL` or
`ADVERSE` rejects static Fisher transfer but does not erase the transductive
oracle result.

## Claim boundary

This remains a first-token functional-distortion experiment on a weak frozen
LLaVA reader. It cannot establish answer accuracy, TTFT speedup or superiority
to video-token selection systems.
