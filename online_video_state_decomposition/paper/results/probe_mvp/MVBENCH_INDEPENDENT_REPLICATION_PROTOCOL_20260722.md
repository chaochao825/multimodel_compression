# MVBench Independent Replication Protocol

Frozen before reading any result from the 300-sample run on 2026-07-22.

## Objective

Test whether the previously designed bounded query-conditioned memory and
grid/sparse residual route transfer to the final untouched MVBench reserve.
This is a LLaVA-1.5 mechanism replication, not a comparison with the
different-backbone StreamingBench systems.

## Frozen Split

The evaluation contains all 60 remaining reserve samples from each of five
tasks, for 300 paired samples total. It has zero overlap with the original
calibration set, original evaluation set, and prior 200-sample confirmation.
The split manifest SHA256 is
`720423b308f97aecdacd1bb9c70b13ae08920a38e3abb75147e63e71486821f2`.
Its direct source and parent hashes are retained in the manifest.

## Frozen Method

- Backbone: local LLaVA-1.5-7B, 32 sampled frames.
- Reader state: 16 projected feature frames, eight selected visual frames.
- Policies: `exact_recent` and frozen `learned_recent_query_topk`.
- Codec: rank-256 PCA with SHA256
  `39f7623b3ae95db93403ef00c63589d901a2d9011dd4a23b8fc185d76d20012f`.
- Variants: full, latent-only, fixed sparse s4, spatial grid 2x2, and the
  frozen grid/sparse route.
- Route: compare both FP16-stored candidates and retain the lower
  reconstruction-error state with ratio 1.0. This remains an error oracle,
  not a deployable low-cost router.

## Decisions

The primary preservation comparison is routed versus full state under the
learned reader. It passes only if the one-sided 95% Clopper-Pearson upper bound
on full-correct/routed-wrong events is at most 2%. At `n=300`, at most one such
loss is allowed; two losses yield an upper bound of 2.084% and fail.

The primary memory comparison is learned routed state versus exact-recent
routed state. Report paired accuracy gain, exact McNemar p-value, and a paired
confidence interval. No selector, route, threshold, prompt, or metric may be
retuned after results are observed.

Secondary analyses compare fixed sparse, grid-only, latent-only, routed, and
full states; localize failures by task; and report steady/cold-start bytes plus
writer and cached-read timing. Task-level findings remain descriptive unless
multiple-comparison correction is applied.

## Claim Boundary

Passing would independently support representation preservation and the
bounded query-memory mechanism on this backbone. It would not establish a
cheap router, end-to-end streaming latency, cross-backbone superiority, BCCB
benefit, or online-video state of the art.
