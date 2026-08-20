# PLAN-026: Screen RoPE-commuting gauge equalization for FP8 attention

- Status: completed
- Owner: researcher and Agent
- Gate: fully exposed pre-gate diagnostic
- Claims: none
- Candidate line: exact-invariant dense mixed-precision attention
- Lane: explore
- Resource cap: one implementation, one repair, and one exclusive H200 pass under one hour

## Decision to unlock

Determine whether the existing FA3 FP8 error is dominated by avoidable input
dynamic-range mismatch rather than irreducible FP8 attention error. Test an
exact BF16 reparameterization before opening any new rollout or trained sparse
operator campaign.

For post-RoPE Q/K and a positive diagonal matrix whose adjacent entries are
equal within every RoPE complex pair,

\[
Q' = QD,\qquad K' = KD^{-1},\qquad Q'K'^\top=QK^\top.
\]

Because each two-channel block of `D` is a scalar multiple of identity, it
commutes with Wan's adjacent-pair 3D RoPE rotation and can be folded into the
Q/K projections. A fixed normalized Hadamard transform of V is likewise exact
when its inverse is folded into the output projection:

\[
AVW_O=(AVH)(H^\top W_O).
\]

The transforms therefore change FP8 quantization coordinates, not dense BF16
attention semantics.

## Frozen data and split

- Reuse immutable EXP-040 post-RoPE Q/K/V captures only.
- Cell: Wan2.1-T2V-1.3B F81, Layer 14, step 9, conditional branch.
- Identities: `s00/s01` calibration, `s02` validation, `s03` test.
- Evaluate the 16 deterministic 64-query tiles and all 12 heads used by
  DIAG-024. No tile, head, or identity may be changed after outcomes are read.
- All identities are exposed; results are diagnostic and cannot establish
  deployment transfer or video quality.

## Registered methods

1. `fa3_bf16`: actual FA3 BF16 tile output against an FP32 dense reference.
2. `fa3_fp8_stock`: actual FA3 FP8 with its existing full-tensor dynamic amax
   scales for Q/K/V.
3. `fake_fp8_q`, `fake_fp8_k`, `fake_fp8_v`, and `fake_fp8_qk`: component
   diagnostics using the same E4M3 round trip and FP32 attention.
4. `fa3_fp8_pair_gauge`: calibration-only per-head, adjacent-pair amax gauge,
   clipped once to `[1/8, 8]`, followed by the unchanged tensorwise FA3 scales.
5. `fa3_fp8_v_hadamard`: fixed normalized 128-channel Walsh-Hadamard V basis,
   with the inverse applied before evaluation.
6. `fa3_fp8_pair_gauge_v_hadamard`: the two foldable transforms combined.
7. `fa3_fp8_channel_gauge`: unrestricted per-channel calibration gauge as a
   non-foldable diagnostic upper bound; it cannot authorize deployment.

No scale objective, clamp, transform, split, tile, or method sweep is allowed.
The pair and channel gauges use only calibration amax statistics. The actual
FP8 methods must call the installed FA3-v3 kernel with full-sequence global
Q/K/V scales, even though only registered query tiles are evaluated.

## Validation and fairness

- Verify pair-gauge QK outputs in FP32 and V-Hadamard output algebra using the
  same FP32 attention probabilities with FP64 linear accumulation are invariant
  before quantization within `1e-5` relative L2. Record the pure-FP32
  `A(VH)H^T` reassociation difference separately rather than hiding it.
- Keep the FP32 reference, BF16 FA3 baseline, query tiles, softmax scale, and
  global tensorwise FP8 scaling identical across methods.
- Report aggregate, worst head, worst tile, identity, and component errors.
- Record all code/config/data/source hashes, environment, elapsed time, and the
  installed FA3 wrapper revision.
- No local error may be converted to SSIM, VBench, rollout quality, or speed.
  Foldability is an arithmetic property, not a measured latency result.

## Outcome map

- `foldable-gauge-pass`: combined pair gauge plus V-Hadamard reaches `1%`
  aggregate and `2%` worst-head/worst-tile error on both held-out identities.
  Authorize fresh multi-cell captures and a fused/folded H200 kernel Gate.
- `equalization-boundary`: combined method cuts stock FP8 aggregate error by at
  least 50% on held-out identities but misses `1% / 2%`. Retain the transform as
  initialization for low-cost QAT; do not run a static rollout.
- `v-quantization-boundary`: V-only error dominates and V-Hadamard supplies the
  majority of improvement. Prioritize V/O folding and accumulation accuracy.
- `qk-quantization-boundary`: Q/K error dominates and pair gauge supplies the
  majority of improvement. Prioritize RoPE-pair equalization and Q/K scaling.
- `mixed-equalization-boundary`: a foldable method improves held-out stock FP8
  by at least 20% but less than 50%, with no single branch dominating. Preserve
  the mechanism only as a QAT initialization.
- `gauge-null`: no foldable method cuts held-out stock FP8 aggregate error by
  at least 20%. Terminate this train-free equalization line.
- Standard invalid, contradictory, adverse, and engineering-failure outcomes
  retain precedence.

## Stop rules

Stop after the first valid H200 diagnostic or the one allowed implementation
repair. Do not tune scales on validation/test, add learned rotations, change
FP8 format, run Wan, benchmark full rollout, or claim speed. Any QAT, learned
router, branch ratio, or sparse-linear operator requires a new prospective
Gate.

## Frozen implementation

- Config SHA-256:
  `454fa201f4d6fe18cefaacd70848aa06373ce1e27b9322f515e9001d8b5a8a75`.
- Core SHA-256:
  `e924b16afe4cf90dffaa57b6f1a16abaaa55d24bad743158823b894c142a74e4`.
- Probe SHA-256:
  `8f32cae100894954fd21584f2e15eb942184a0c213ae2bec942cb87d288887bc`.
- Test SHA-256:
  `aebe47d963f7e7173ff0187c049c1d387d730c079176a93d85a133057fa7316e`.
- Installed SageAttention FA3 wrapper SHA-256:
  `53813a192b9d64b49cf72179aadbce39c67c70c00cae2331ed273887c6a58af9`.
- Capture-index SHA-256:
  `7d12d9ad9fa19379417e943d0962222bfdc59a5df4a91b5f12a625a2e7fda947`.
- Local and remote implementation hashes match. Six core tests and one actual
  H200 FA3-v3 FP8 unequal-sequence smoke test passed before the formal run.
- The first formal attempt was invalid before any method outcome was emitted:
  Q/K gauge invariance was about `3.5e-7`, while FP32 reassociation of
  `A(VH)H^T` versus `AV` was `1.19e-5`--`1.52e-5`, marginally above the
  over-strict guard. The one allowed repair keeps this FP32 value visible and
  moves only the algebraic identity guard to FP64 linear accumulation under the
  same FP32 attention probabilities. Actual BF16/FP8 method evaluation is
  unchanged.

## Closure

Write a bound report and visualization, update `STATUS.md`, and keep claim,
candidate, experiment, and RDR registries unchanged because all captures are
fully exposed.

Closed 2026-08-12 with outcome `gauge-null`.

- Held-out stock FA3 FP8 was `5.1078%` aggregate, `9.2015%` worst head, and
  `5.7880%` worst tile versus `0.1678%` aggregate for FA3 BF16.
- Pair gauge, V-Hadamard, and their combination reached `5.1286%`, `5.1322%`,
  and `5.1530%` aggregate. The combination worsened stock error by `0.884%`
  relatively; both held-out identities failed.
- The unrestricted non-foldable channel-gauge diagnostic reached `5.1010%`,
  only `0.133%` relative improvement.
- Fake input round trips reached Q `2.1889%`, K `2.2955%`, V `0.5913%`, and
  Q+K `3.1400%`, locating the main input sensitivity in Q/K rather than V.
- The exact pair-gauge guard was at most `3.58e-7`; the separately retained
  FP32 V-Hadamard reassociation difference was at most `1.53e-5`.
- Formal H200 run exited zero after the one registered guard repair; six core
  tests passed and GPU3 was released.
- Decision SHA-256:
  `df5346fddf01e38a13a532f91392e6f0b8c1e5fef4289f67e72ec371e1939617`.
- Manifest SHA-256:
  `444e15d4e95656ea8e24c15e1907dfdc6dd21ebffbebd96b3a616601f327b942`.
- Evidence:
  `worldfoundry_hybrid_residual/results/fp8_rope_gauge_f81_l14s9_diag026_v1/`.
- Report:
  `worldfoundry_hybrid_residual/results/WAN_FP8_ROPE_GAUGE_DIAG026_20260812.zh-CN.md`.
- Required action: `TERMINATE_TRAIN_FREE_GAUGE_EQUALIZATION`; next audit the
  existing fine-grained Q/K and two-level-PV kernel frontier before QAT.
