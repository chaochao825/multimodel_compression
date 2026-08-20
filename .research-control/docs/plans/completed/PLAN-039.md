# PLAN-039: Released Sol-Attn strict-quality H200 Pareto gate

- Status: completed
- Owner: researcher and Agent
- Gate: released-kernel quality and latency reproduction
- Claims: none; all DIAG-030 identities are globally exposed
- Candidate: pinned Sol-Attn `cute_sm90` forward kernel
- Lane: explore
- Resource cap: one isolated environment, at most two environment attempts,
  one H200-hour on physical GPU3, no model download, no fresh capture

## Decision to unlock

Determine whether the released Sol-Attn function class has at least one fixed
operating point that passes the existing strict local `AV` guards on every
selected Wan2.1-1.3B F81 capture while delivering at least `1.5x` measured
whole-attention speed versus local FA3 BF16 on H200.

## Frozen identity and protocol

- Candidate repository: `https://github.com/NVlabs/Sana.git`.
- Branch/commit/tree: `sol-engine`,
  `5dd502af9938d924be206c332ad1e911b4a925a1`,
  `730cef4dc0fe6be2e4b17997fec295f730afa541`.
- Protocol:
  `worldfoundry_hybrid_residual/configs/sol_attn_released_kernel_pareto_diag037_v3.json`.
- Protocol revision 2 was frozen before any candidate result existed. A source
  audit found that the released Wan integration applies one global Morton3D
  order around the complete block stack; revision 1 incorrectly excluded that
  official function-class component and is retained only as history.
- Pre-result dual-backend smoke then found that the registered Torch 2.9 FA3
  binary could not load in the required Torch 2.10 Sol environment. Revision 3
  records an ABI-compatible rebuild from the identical frozen FA3 Hopper source
  manifest. No Sol output was evaluated before either correction; candidate,
  data, layout, tau, guards, timing boundary, and resource caps are unchanged.
- Data: the 96 immutable DIAG-030 evaluation captures for four identities,
  layers `[0,14,29]`, steps `[2,7,12,17]`, and both CFG branches.
- Candidate source is read-only. Harnesses, tests, logs, and environment files
  live outside the candidate checkout.

## Frozen methods

- Evaluate exactly `tau = [-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0]`.
- Use the released `diag` threshold, `kv_splits=1`, no sink tokens, contiguous
  BF16 BTHD Q/K/V, and head dimension 128.
- Apply the source-identical global Morton3D order for grid `(21,30,52)` once
  before both Sol and FA3. Inverse-order outputs before evaluating the original
  DIAG-030 query identities. This layout is fixed and is not a searched method.
- Set strict dispatch and assert `get_sol_attn_backend(...) == "cute_sm90"`.
  Triton or dense fallback is invalid evidence, not a slower candidate.
- Compute a source-faithful exact-block density audit outside the timed region.
  Density is diagnostic and never substitutes for measured wall time.

## Evaluator and guards

- Quality reference: the same FP32 tiled dense softmax evaluator and 16
  deterministic 64-query tiles used by DIAG-030.
- For every capture, require aggregate/worst-head/worst-tile relative L2 at
  most `1% / 2% / 2%`, finite output, and unchanged shape/dtype semantics.
- FA3 BF16 control must remain below `0.3%` aggregate relative L2.
- Latency baseline: actual local FA3 BF16 full attention on the registered
  Layer-14 step-12 conditional capture, using the same preordered Q/K/V as Sol.
- Time the complete public Sol API including pooled K/V and threshold
  preprocessing, with two warmups and seven alternating forward/reverse
  synchronized repetitions. The model-level Morton order is outside both
  per-attention timings, matching the released integration. Require median
  speedup `>=1.5x`.

## Fairness and leakage boundary

- This is a released-operator diagnostic, not a prospective model claim. All
  captures have appeared in prior diagnostics; no calibration or transfer
  claim is permitted.
- Do not use output errors to change tau, threshold type, KV split, block size,
  sink, query tiles, or latency identity after execution begins.
- Do not compare FLOPs against measured FA3 latency. Do not omit Sol routing or
  preprocessing from the primary time.
- The local 20-step/shift-5/guidance-5 QKV setting differs from the published
  50-step Sol-Engine configuration. Report it as a local operator reproduction,
  not a paper end-to-end reproduction.

## Outcome mapping

- `released-kernel-pass`: at least one fixed tau passes every quality guard and
  measured `>=1.5x`. Authorize a fresh prospective rollout protocol, not an
  automatic rollout.
- `speed-boundary`: at least one fixed tau passes quality, but every passing
  point is below `1.5x`. Close strict-quality Sol integration unless a separately
  accepted exact-system optimization changes the dense baseline.
- `quality-boundary`: at least one point reaches `1.5x`, but no fixed tau passes
  every quality guard. Preserve the Pareto curve as a strong baseline and do
  not tune another exposed proxy.
- `strict-null`: no point reaches either a universal quality pass or `1.5x`.
  Close this released-baseline line under the frozen shape and protocol.
- `environment-boundary`: complete artifacts cannot produce `cute_sm90` within
  two isolated attempts and 20 GiB. Preserve source readiness; make no method
  claim.
- `invalid`: source identity, capture hashes, backend, evaluator, GPU isolation,
  or frozen protocol differs.

## Stop rules

- Run unit tests and one small-shape backend/correctness smoke before loading
  the atlas.
- Stop immediately on Triton/dense fallback, modified source, nonfinite output,
  wrong capture identity, GPU overlap, or resource-cap breach.
- Do not add tau points, exact covariance, an alternate/native token layout,
  dynamic per-cell tau, dense guards, new captures, model weights, or rollout
  inside this Gate.
- Once the seven-point table is complete, close the Gate; do not start a new
  method family from the same exposed outputs.

## Closure

- Closed date: `2026-08-12`.
- Outcome: `quality-boundary`.
- Nearest quality point: tau `-2.0`, `92.26%` exact blocks, `40/96` passing
  captures, `0.826% / 1.526% / 7.296%` global aggregate/head/tile error, and
  `1.354x` H200 whole-attention speedup.
- First speed-gate point: tau `-1.5`, `83.85%` exact blocks, `34/96` passing
  captures, `1.279% / 2.425% / 8.042%` error, and `1.542x` speedup.
- No fixed tau passed every `1% / 2% / 2%` capture guard and `>=1.5x` speed.
- Validity: all source/data/code hashes, strict `cute_sm90` dispatch, FA3
  control, finite-output guard, dual idle checks, and zero-overlap guard passed.
- Boundary: all identities were globally exposed; no prospective transfer,
  rollout, model coverage, or end-to-end claim is permitted.
- Action: preserve the complete public Pareto and park further tuning on this
  exposed atlas.
- Evidence:
  `worldfoundry_hybrid_residual/results/WAN_SOL_ATTN_RELEASED_KERNEL_PARETO_DIAG037_20260812.zh-CN.md`.
