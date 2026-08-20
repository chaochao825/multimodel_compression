# PLAN-035: Pre-attention value-risk block routing for faster Sage variants

- Status: completed (`support-null`)
- Owner: researcher and Agent
- Gate: exposed mechanism-and-cost screen after DIAG-034 proxy null
- Claims: none; all DIAG-030 identities have appeared in prior diagnostics
- Candidate line: one-pass regular mixed-V precision inside faster Sage kernels
- Lane: explore
- Resource cap: immutable DIAG-030 captures, one implementation repair, one
  exclusive H200 pass under 90 minutes, and no rollout or custom CUDA kernel

## Decision to unlock

Determine whether a support known **before** full attention can repair the
small quality miss of faster Sage variants while retaining a conservative
whole-attention latency envelope of at least `1.5x` versus FA3 BF16.

The candidate keeps all QK support dense. For each 64-query tile and head it
selects 13 of 256 regular 128-key blocks, then treats only those V blocks as a
high-precision residual path:

`Y = Y_sage + P_Omega (V - V_q)`.

Unlike DIAG-032 probability-mass selection, the deployable support must be
available before the main kernel begins. This would permit one fused online
softmax pass in which selected V blocks use BF16 and all remaining V blocks use
FP8, without a second QK pass or an online low-rank tail.

## Scope and evidence boundary

- Reuse the 96 hashed DIAG-030 captures for Layer `[0,29]`, steps
  `[2,7,12,17]`, both CFG branches, 16 fixed 64-query tiles, 12 heads, 128
  channels, and 32,760 tokens.
- Preserve the original calibration/evaluation labels for within-run policy
  simulation, but treat every result as globally exposed. No transfer,
  generalization, rollout, or deployment claim is permitted.
- The exact FP32 probability and target defect may be used only for diagnostic
  references and output scoring. The primary router may read current Q/K/V,
  quantized V, position, layer, step, and branch, but not dense probability,
  Sage output, dense output, or any defect.

## Frozen Sage bases

1. `thread_smooth`: per-thread INT8 Q/K, smooth-K, per-channel FP8 V,
   `fp32+fp32` PV. Historical isolated speed `1.589031x`.
2. `warp_smooth`: per-warp INT8 Q/K with smooth-K. Historical isolated speed
   `1.628335x`; it narrowly missed DIAG-027 quality.
3. `thread_no_smooth`: per-thread INT8 Q/K without smooth-K. Historical speed
   `1.611407x`; it also narrowly missed DIAG-027 quality.

All three must use the installed Sage SM90 worker and the same actual output
extraction as DIAG-027/032. Historical speed is a cost reference only; this
plan does not claim a new measured kernel.

## Frozen support families

All selectors choose exactly 13 blocks per head/query tile.

1. `exact_mass`: top block mass from FP32 dense probability. Diagnostic upper
   reference; unavailable before attention.
2. `v_residual`: top block RMS of `V - V_q`. Query independent, pre-attention,
   and one-pass eligible.
3. `centroid1`: one mean Q vector per query tile and one mean K vector per key
   block; rank blocks by their centroid logit.
4. `centroid4`: split query and key blocks into four contiguous groups each,
   compute 16 centroid logits, and rank by their stable log-sum-exp.
5. `centroid4_vrisk`: primary candidate. Add `log(RMS(V-V_q)+1e-8)` to the
   `centroid4` log-mass score.

The four-by-four router uses 16 centroid interactions instead of 8,192 token
pairs per tile pair, a QK arithmetic ratio of `0.1953125%` before pooling and
top-k. It can select any long-range regular block and makes no local-window or
periodic assumption.

## Frozen correction variants

For each selector and Sage base report:

- `exact_p`: add the selected `P(V-V_q)` correction. This isolates support
  quality but is not deployable evidence.
- `modeled_online_p`: use the registered running-max E4M3 numerator and FP32
  denominator model from DIAG-032 on selected blocks. This is the primary
  one-pass numerical proxy, but still omits actual Sage INT8-QK probability
  exposure and cannot establish kernel equivalence.

Also record exact-mass support overlap, exact probability mass recall, V-error
energy recall, selector arithmetic, and the correction's signed alignment with
the actual Sage defect.

## Quality, policy, and cost gates

Quality guards remain aggregate `<=1%`, worst head `<=2%`, and worst tile
`<=2%` for every layer-step cell.

Primary Stage-A pass requires `centroid4_vrisk + modeled_online_p` to pass all
eight cells for either `warp_smooth` or `thread_no_smooth` without BF16-head
fallback. `exact_p` passing while `modeled_online_p` fails is a probability
model boundary, not a method pass.

As a secondary exposed diagnostic, use calibration identities only to rank
risk heads and freeze the smallest action among 0, 1, 2, or all 12 BF16 heads
that meets `0.85` times every quality threshold. Apply the frozen table to the
evaluation labels. This table cannot become confirmation evidence; it only
localizes residual risk and estimates whether fresh captures are warranted.

For one-pass mixed V, define a conservative latency envelope by charging the
selected fraction against the full observed FA3-BF16 minus Sage latency gap:

`T_mix = T_sage + density * (T_bf16 - T_sage) + T_router`.

Charge router arithmetic at its QK ratio and report both zero-overhead and
charged envelopes. Any cell requiring the existing split 1/2-head precision
path inherits its measured speed `1.4907x / 1.4290x`, which cannot satisfy the
`1.5x` primary speed gate.

## Outcomes

- `one-pass-block-pass`: primary modeled correction passes all 8 cells for a
  faster Sage base and conservative latency envelope is `>=1.5x`. Authorize a
  custom mixed-V Sage kernel and fresh-capture validation.
- `probability-model-boundary`: pre-attention support with exact P passes, but
  modeled online P fails. Inspect actual Sage online-softmax statistics before
  any kernel implementation; do not add another predictor family.
- `risk-island-boundary`: primary fails, but the frozen 0.85-margin policy has
  zero false-safe evaluation cells and no more than two BF16 heads in every
  cell. Retain only if a fused cost envelope could still reach `1.5x`.
- `support-null`: exact-P centroid/value-risk support materially trails
  exact-mass and no primary/secondary route meets quality. Stop regular V-block
  routing.
- `cost-null`: numerical quality is possible but even the zero-router
  conservative envelope is below `1.5x`. Stop before CUDA.
- `invalid`: leakage, changed methods/gates, nonfinite output, artifact or
  identity mismatch.
- `engineering-failure`: implementation cannot complete within one repair and
  the resource cap; preserve evidence and stop.

## Stop rules

Do not change block sizes, selected density, centroid count, score formula,
Sage variants, correction probability model, safety margin, quality gates, or
outcome thresholds after the first formal capture is evaluated. Do not train,
load model weights, write CUDA/Triton, capture fresh identities, or run video
rollout in this plan.

## Frozen implementation

- Config SHA-256:
  `799a6e926598f4aae487a7020eaf09ac6ff7b37e1d3a2b0757f7c2f89314b348`.
- Router core SHA-256:
  `897be9864248ab07d826fcd6d0403b4c24c64fc82051f0dd97f99d795ba21162`.
- Core tests SHA-256:
  `9e40c01c47f64006280dca3bb123cdeed7ee3e5abb8c27bb6f8a6758d3df791c`.
- Probe SHA-256:
  `1387c7f397964d43a3d556a1d37812726f618571d53e14fb83e5a1303d6c9a02`.
- Runner SHA-256:
  `1f9ec309265f9f003e39866bcbf643bd4a85fd19afd73fe9b8e1159ded977759`.
- Reused DIAG-032 loader/core, operand helper/core, and query-tile helper
  SHA-256:
  `a17d529d569b8818d38476ec459f0a96710fe08af0f026756f7d03171ddaf6f3` /
  `be275347dd9fdd530100fcf69c14ca86419d9927c4db1801548040d0be8c8bb1` /
  `76c9d0591046fea918bb84011ebc84d4046d01358ae283cd0a543c177c263b74` /
  `62a798b1c0097c5dcf7eaa3ba5fc070253c68563c98101b30068245e033ad85c` /
  `e924b16afe4cf90dffaa57b6f1a16abaaa55d24bad743158823b894c142a74e4`.
- Remote tests passed `8/8`. A one-capture H200 engineering smoke produced
  6,336 finite rows covering all three bases and 11 methods. Smoke output is
  engineering evidence only and cannot change the frozen outcome mapping.

## Required closure

Archive frozen source hashes, tests, per-head/tile records, cell and policy
summaries, support-recall diagnostics, cost envelope, decision, manifest, one
visualization, and one Chinese report. Classify the result before opening any
kernel or fresh-capture gate.

## Closure

- Closed UTC date: `2026-08-12`.
- Validity class: valid, fully exposed mechanism-and-cost diagnostic.
- Outcome: `support-null`.
- Formal runtime: `165.8 s`; tests `8/8`; foreign GPU PID count `0`.
- Primary `centroid4_vrisk + modeled_online_p` passed only `2/8` cells for
  warp-smooth and `3/8` for thread-no-smooth.
- On thread-no-smooth, exact mass and centroid1 each passed `5/8`; mean
  aggregate errors were `0.869% / 0.944%`, versus `1.065%` for the primary.
- Centroid1 recovered `85.3%` exact top-k indices and `56.1%` exact probability
  mass. V-risk recovered only `5--6%` V-error energy at 5.08% density and did
  not improve centroid support.
- Exact-P and modeled-online-P errors differed by less than approximately
  `0.0002` percentage points in the aggregate averages; the registered
  probability model was not the failure source.
- Exact-mass support still failed L0/S2, L0/S7, and L29/S2 for the best faster
  base. L29/S2 worst-tile error remained approximately `2.10%` after V
  correction, establishing a non-selector residual boundary.
- No-fallback charged speed envelopes were `1.540x / 1.575x / 1.560x`, but the
  0.85-margin policy required one full-BF16 L0/S2 cell for every base. Policy
  harmonic envelopes fell to `1.396x / 1.399x / 1.405x`.
- Supports: one-centroid Q/K pooling is a useful long-range coarse support
  feature, and no-smooth Sage is a stronger edge-layer base in this atlas.
- Does not support: a deployable 5% pre-attention V-block mixed-precision path
  satisfying both full-cell quality and `1.5x` attention cost.
- Unknown: actual mixed-V kernel behavior, fresh transfer, and rollout; none
  was executed.
- Action: reject further V-block selector refinement. Retain centroid1 only as
  a router feature and move the next bounded diagnostic to Q/K scale
  granularity plus online-softmax interaction.
- Raw records SHA-256:
  `5b5af84a7084fdd4a5ef9ad52915768695ba4db18406a2243eb85cd2d877a99d`.
- Decision SHA-256:
  `e864c954e0550d0cabbef2397665277a82e27a6166dc89a8622f767b8415d16d`.
- Manifest SHA-256:
  `0fbcf7277339c52e24fbe48bdfeb947334ed05668f122ab88c08a68cf006e6de`.
- Report SHA-256:
  `b6b10250d95896674674ddf0001b291ec9dfcae3c321c9d88ddcaa93687b4d13`.
- Figure PNG SHA-256:
  `1acd40e4a3b8b4e6304babf0d7307471f148b1ac49e49582e800f3a4909068f6`.
