# PLAN-034: Foldable V-channel quantization-aware rotation

- Status: completed (`proxy-null`)
- Owner: researcher and Agent
- Gate: small QAT after DIAG-033 basis-transfer null
- Claims: none; this is a captured-QKV adaptation diagnostic
- Candidate line: semantics-preserving V/O coordinate rotation for Sage FP8 PV
- Lane: explore
- Resource cap: immutable DIAG-030 captures, one implementation repair, and
  one exclusive H200 pass under 90 minutes

## Decision to unlock

Determine whether a calibration-only orthogonal V-channel rotation can reduce
the actual Sage FP8-PV defect on held-out Layer-0/29 cells while preserving the
unquantized attention function exactly and adding no deployed attention branch.

For each layer and head, use a transform shared across every step, CFG branch,
prompt, seed, and token:

`V' = V T`, `Y' = Attention(Q,K,V') T^T`.

When `T T^T = I`, exact arithmetic gives `Y' = Attention(Q,K,V)`. `T` and
`T^T` can therefore be folded into the V and O projections before deployment.
This plan tests capture-level numerical transfer only; folding, model rollout,
and end-to-end speed remain outside the claim.

## Immutable split and scope

- Reuse DIAG-030 captures: layers `[0,29]`, steps `[2,7,12,17]`, both CFG
  branches, 16 fixed 64-query tiles, 12 heads, 128 channels, 32,760 tokens.
- Calibration identities: `s00_p00_seed20260860`, `s01_p01_seed20260860`.
- Evaluation identities: `s02_p00_seed20260861`, `s03_p01_seed20260861`,
  `s04_p02_seed20260861`, `s05_p03_seed20260862`.
- Train/freeze every transform and prospective precision-island policy before
  loading any evaluation tensor payload. Evaluation metadata may be read.

## Frozen transform families

1. `identity`: actual stock Sage.
2. `hadamard`: fixed normalized Walsh-Hadamard V rotation and inverse; no
   training. This repeats the fixed isotropization baseline in the new atlas.
3. `cayley_r4`: per-layer/per-head rank-4 Cayley rotation.
4. `cayley_r8`: per-layer/per-head rank-8 Cayley rotation.

For rank `r`, parameterize a skew matrix

`A = U V^T - V U^T`,

where `U,V` have width `r`, initialize `U` deterministically and `V=0`, and use

`T = (I - A)(I + A)^{-1}`.

This initializes exactly at identity and keeps `T` orthogonal up to numerical
error. Both factors are learned, but `V=0` gives a nonzero initial gradient for
the second factor. Parameter counts are 12,288 and 24,576 per layer for r4/r8;
the dense folded transform is not executed online.

## Frozen calibration objective

- Train each rank and layer independently with Adam, deterministic seed, two
  epochs, learning rate `0.03`, no weight decay, gradient norm cap `1.0`.
- Process all 16 calibration captures per layer in fixed order. Concatenate
  four registered query tiles per update, yielding four updates per capture
  and 128 updates per layer/rank.
- Forward quantization is the same per-(batch,head,channel) E4M3 roundtrip as
  DIAG-031/032. Backpropagation uses a straight-through payload gradient.
- Minimize dense-output relative MSE using exact FP32 probabilities and the
  original unrotated dense output as teacher. No evaluation tensor, actual
  evaluation Sage output, rollout, or downstream label enters training.
- Stage-A guard: on the complete calibration proxy, learned r8 must reduce
  V-only output error energy by at least 10% versus identity and maximum
  `||T T^T-I||_F/sqrt(d)` must be `<=1e-4`. Otherwise stop as `proxy-null`
  without loading evaluation payloads.

The proxy omits QK/PV kernel interaction. Final classification uses the actual
Sage SM90 worker on calibration and evaluation captures.

## Prospective precision-island baseline

After transforms are frozen, evaluate calibration actual-Sage errors and, for
each layer-step, select the smallest action among 0, 1, or 2 BF16 heads that
passes all three calibration quality guards. Head order is calibration error
energy only. If none passes, freeze that cell as full BF16 fallback. Freeze the
action table and hash it before loading evaluation.

Apply the frozen action table to `cayley_r8` on evaluation. BF16 head output is
the original dense reference for that head; selection never reads evaluation.
Inherited native H200 local speed proxies are `1.5906x`, `1.4907x`, and
`1.4290x` for 0/1/2 BF16 heads. Rotation cost is excluded only because the
deployment hypothesis folds it into V/O weights; no speed claim is allowed
until that fold is implemented and benchmarked.

## Metrics and outcomes

Report aggregate, worst-head, and worst-tile relative L2 for every
split/layer/step/method; per-identity and per-head errors; proxy training curves;
orthogonality; transform distance from identity; calibration action table; no
false-safe count; and the inherited arithmetic speed proxy.

Full BF16 fallback has local speed proxy `1.0x`; it is never counted as a
low-precision quality pass. The campaign speed proxy is the harmonic mean over
the eight frozen layer-step actions because latency, not throughput, composes
across sequential cells.

Quality guards remain aggregate `<=1%`, worst head `<=2%`, worst tile `<=2%`.

- `foldable-rotation-pass`: r4 or r8 actual Sage passes every evaluation cell
  without BF16 heads. Authorize weight folding and a fresh rollout gate.
- `foldable-island-pass`: the frozen r8 island policy has zero false-safe cells,
  passes all evaluation cells, and its mean local speed proxy is `>=1.4x`.
  Authorize folding plus actual grouped-kernel timing before rollout.
- `rotation-boundary`: no method passes, but r8 reduces aggregate actual-Sage
  evaluation error energy by at least 20% in both layers without a new adverse
  worst tile. Retain only as a QAT baseline; do not develop a kernel.
- `proxy-null`: Stage A fails. Do not load held-out tensors; stop rotations.
- `rotation-null`: all learned methods miss the boundary. Stop V/PV residual
  tail work and retain heterogeneous Sage/BF16 plus external trained baselines.

Invalid, contradictory, adverse, leakage, and engineering-failure outcomes
retain precedence.

## Stop rules

Stop after Stage A null or one valid evaluation pass. Do not change learning
rate, epochs, ranks, tile grouping, transform granularity, action budget,
quality gates, or method order after training starts. Do not add sparse/tail,
BCM/BCCB/Butterfly/cache/temporal predictors, train original Wan weights, write
CUDA, or run rollout in this plan.

## Frozen implementation

- Config SHA-256: `8efa30b4f876afc4113275b6ed8a082f08fa373cd0a652e4c9fdce59485930db`.
- Transform core SHA-256:
  `b56c6b8dc85a3619b69e989d433d871fe2ce2d84cf7b11a0676974e9ad9286bf`.
- Core tests SHA-256:
  `81c501bc743697a7d7169dd05eb79cd6630ddebb987614d9da0938f3b8a581d7`.
- Probe SHA-256: `6159cc4f46456d3cec9e08c68da1ee1ca6ff08bd72cc1d4e868120702d894ee6`.
- Runner SHA-256: `2f03a83f475c29bb78cd1d85bfac5a1a2d1938e26383c0e4cb8c672df7359047`.
- DIAG-032 loader / DIAG-031 helper / summary core / dense evaluator SHA-256:
  `a17d529d569b8818d38476ec459f0a96710fe08af0f026756f7d03171ddaf6f3` /
  `76c9d0591046fea918bb84011ebc84d4046d01358ae283cd0a543c177c263b74` /
  `be275347dd9fdd530100fcf69c14ca86419d9927c4db1801548040d0be8c8bb1` /
  `e924b16afe4cf90dffaa57b6f1a16abaaa55d24bad743158823b894c142a74e4`.
- DIAG-030 capture provenance SHA-256:
  `f56604ea92906b1ea864d5002bb2bcef528d34b26fa4fe35b362bc414ce9adb2`.
- Remote pure-function tests passed `9/9`. A one-capture calibration-only H200
  smoke produced finite nonzero STE gradients, exact Cayley orthogonality, and
  a finite transformed Sage output of shape `[1,64,12,128]`; it is engineering
  evidence only and cannot change the frozen gate.

## Required closure

Archive frozen source hashes, training traces, transform bank and freeze
receipt, action table and freeze receipt, actual-Sage records, summaries,
decision, one visualization, and one Chinese report. Update `STATUS.md` and
open at most one fold/rollout gate only if the frozen outcome authorizes it.

## Closure

- Closed UTC date: `2026-08-12`.
- Validity class: valid calibration-only Stage-A diagnostic.
- Outcome: `proxy-null`.
- Formal runtime: probe `66.58 s`; runner approximately `80.9 s`; tests `9/9`.
- Leakage guard: transform bank froze before evaluation and
  `evaluation_payloads_loaded = false`; no held-out payload was read.
- Layer-0 best error-energy reduction was fixed Hadamard at `0.6325%`;
  Cayley r4/r8 reached `0.5216% / 0.5078%`.
- Layer-29 all transforms worsened identity; Cayley r4/r8 changes were
  `-0.4652% / -0.4213%`.
- Maximum orthogonality error remained approximately `4e-7`, while rank-8
  transforms moved a normalized distance `0.3338 / 0.3055` from identity.
- Matched epoch-2 loss changes were centered around zero and STE gradient
  norms remained approximately `3e-7--5.6e-7`; the frozen optimizer did not
  show stable descent.
- Supports: shared foldable orthogonal V rotation does not materially improve
  this per-channel E4M3 proxy under the registered objective.
- Does not support: a claim that every rotation-aware quantizer or nonorthogonal
  foldable equalizer must fail.
- Unknown: actual-Sage held-out quality, folded-model quality, rollout, and
  H200 speed; none was executed after the Stage-A null.
- Action: reject further Cayley/Householder/rank/epoch expansion under this
  line. Narrow the next gate to observable regular V-residual blocks plus
  separately calibrated risk-head precision islands.
- Decision SHA-256:
  `bfc7c2f57f079f775ea7dd18f3a9ea01dbc6834d945234677877ca0a01455d65`.
- Manifest SHA-256:
  `1df0eb90c3a957e6deca80772c2bfddfe5998554efc9cc739d33878e13f64e1b`.
- Report:
  `worldfoundry_hybrid_residual/results/WAN_FOLDABLE_V_ROTATION_QAT_DIAG034_20260812.zh-CN.md`.
- Figure:
  `worldfoundry_hybrid_residual/results/foldable_v_rotation_qat_f81_diag034_v1/foldable_v_rotation_diag034.png`.
