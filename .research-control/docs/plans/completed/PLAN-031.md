# PLAN-031: Sage operand-localization diagnostic

- Status: completed
- Owner: researcher and Agent
- Gate: fixed-method operand attribution before low-cost QAT
- Claims: none; this is a diagnostic over immutable DIAG-030 captures
- Candidate line: output-risk-certified low-precision dense attention
- Lane: explore
- Resource cap: reuse DIAG-030 captures, one implementation repair, and one
  exclusive H200 pass under 45 minutes

## Decision to unlock

Identify whether the uncertified Layer-0/29 SageAttention error is dominated by
Q/K INT8, V FP8/PV arithmetic, or a non-additive interaction. The result selects
exactly one bounded adaptation surface: Q/K, V/PV, or joint QKV. It does not
authorize rollout, a custom kernel, or a speed claim.

## Immutable data and scope

- Reuse the 144 SHA-256-registered QKV captures from DIAG-030. Do not run Wan or
  modify the capture split.
- Evaluate layers `[0,14,29]`, sampling steps `[2,7,12,17]`, both CFG branches,
  two calibration identities, and four evaluation identities.
- Use the same 16 deterministic 64-query tiles, FP32 dense attention reference,
  and aggregate/head/tile relative-L2 definitions as DIAG-030.
- All methods and attribution rules below are frozen before evaluation. No
  method, scale, threshold, head, layer, or step is selected from evaluation.

## Fixed methods

1. `dense_fp32`: existing tiled FP32 reference.
2. `fa3_bf16`: numerical guard only.
3. `sage_qk_int8_pv_fp16`: installed SageAttention 2.2.0 CUDA worker with
   per-thread Q/K INT8, smooth-K, original V, and FP32 PV accumulation. This is
   the real-kernel Q/K-path diagnostic, not a deployment candidate.
4. `dense_v_fp8_roundtrip`: original Q/K with a mathematical reconstruction of
   Sage's unsmoothed per-channel E4M3 V quantizer (`scale=amax/448`) followed by
   the same FP32 dense evaluator. This isolates V quantization and makes no
   latency claim.
5. `sage_qk_int8_pv_fp8`: the DIAG-027--030 deployed candidate with per-thread
   Q/K INT8, smooth-K, per-channel FP8 V, and `fp32+fp32` PV accumulation.

For every tile, retain output defect tensors

\[
d_Q=Y_{\mathrm{QK\ INT8,PV\ FP16}}-Y_D,\quad
d_V=Y_{\mathrm{V\ FP8\ roundtrip}}-Y_D,\quad
d_F=Y_{\mathrm{Sage\ full}}-Y_D.
\]

Record relative norms, squared-energy fractions, cosine similarities, and the
unexplained interaction

\[
d_I=d_F-d_Q-d_V.
\]

The components are not assumed orthogonal; attribution must not add scalar
errors or call cancellation an improvement.

## Prospective classification

Classify each `layer x step` cell from calibration only using aggregate defect
energy over identities, branches, tiles, heads, and channels:

- `qk-dominant`: `||d_Q||/||d_F|| >= 0.80` and
  `||d_F-d_Q||/||d_F|| <= 0.50`.
- `v-pv-dominant`: `||d_Q||/||d_F|| <= 0.50` and
  `||d_F-d_Q||/||d_F|| >= 0.80`.
- `mixed-coupled`: every other finite case.

Also record whether `d_Q+d_V` predicts the full defect with residual ratio
`||d_I||/||d_F|| <= 0.35`. This is a diagnostic of additive error modeling,
not a quality gate.

Evaluation tests only transfer of the frozen class labels. A label transfers
when the evaluation ratios satisfy the same inequalities; `mixed-coupled`
transfers when neither dominance rule fires. No label may be changed after
evaluation.

## Latency and outcome map

- Benchmark FA3 BF16, QK-INT8/PV-FP16, and full Sage at native 32,760 tokens on
  the registered DIAG-030 latency capture with two warmups and five alternating
  repetitions. The V roundtrip is excluded because it is a diagnostic graph.
- `localized-qk`: at least 8/12 labels transfer, both Layer-0 and Layer-29 have
  one dominant class in at least 3/4 steps, and that class is `qk-dominant`.
  Open one Q/K-only scale/QAT gate while retaining FP8 PV.
- `localized-v-pv`: the same conditions hold for `v-pv-dominant`. Open one
  V-scale/smoothing/QAT gate while retaining INT8 Q/K.
- `localized-joint`: at least 8/12 labels transfer, but Layer-0/29 are mainly
  `mixed-coupled`. Open one joint, rank-bounded QKV error-shaping gate.
- `nontransferable-attribution`: fewer than 8/12 labels transfer. Do not tune
  on evaluation; require broader calibration or full BF16 fallback.
- Invalid, contradictory, adverse, and engineering-failure outcomes retain
  precedence.

## Stop rules

Stop after one valid fixed-method pass or the one allowed engineering repair.
Do not train, alter quantization scales, change precision islands, add BCM,
low-rank, sparse, rotation, cache, or temporal predictors, or run a video
rollout in this plan.

## Frozen implementation

- Config SHA-256: `55381e23da5f0dbe7149697275bc1d6eb695b0f518a0d50815f1f0a3c7948606`.
- Attribution core SHA-256: `62a798b1c0097c5dcf7eaa3ba5fc070253c68563c98101b30068245e033ad85c`.
- Core test SHA-256: `49d43fd7d2f80659225370d94f6f63e48ee97e8b13c2c42e1a91a999c5e9cfa8`.
- Probe SHA-256: `76c9d0591046fea918bb84011ebc84d4046d01358ae283cd0a543c177c263b74`.
- Runner SHA-256: `8fb787c16e9f164fa9213249a28396f2b171f97e76839284c196f7abec14e3dd`.
- Dense evaluator SHA-256: `e924b16afe4cf90dffaa57b6f1a16abaaa55d24bad743158823b894c142a74e4`.
- DIAG-030 capture provenance SHA-256:
  `f56604ea92906b1ea864d5002bb2bcef528d34b26fa4fe35b362bc414ce9adb2`.
- Remote pure-function tests passed `5/5`; a real H200 single-capture smoke
  produced 192 finite head-tile rows. The smoke classified Layer-0 step-2 cond
  as `v-pv-dominant`, but remains exposed engineering evidence and cannot
  change the frozen formal classification.
- H200 preflight measured approximately `14.3 / 17.0 / 9.0 ms` for FA3 BF16,
  QK-INT8/PV-FP16, and QK-INT8/PV-FP8, respectively. The FP16-PV path is
  therefore frozen as a diagnostic only.

## Required closure

Archive method/source/capture hashes, raw per-tile component metrics, frozen
calibration labels, evaluation transfer outcomes, latency records, a layer-step
attribution heatmap, and one Chinese report. Update `STATUS.md` and open at most
one adaptation gate selected by this outcome.

Closed 2026-08-12 with outcome `localized-v-pv`.

- All 144 immutable captures produced 27,648 finite head-tile records in
  63.26 seconds; tests passed `5/5` and the foreign-PID audit remained empty.
- Calibration labels transferred for `11/12` cells. Layer 29 was
  `v-pv-dominant` at all four steps, Layer 0 at three evaluation steps, and
  Layer 14 remained `mixed-coupled` throughout.
- Evaluation V-only/full-defect cosine was `0.887--0.951` on Layer 29 and
  `0.712--0.846` on Layer 0. The additive QK-plus-V model passed only `2/12`
  evaluation cells, so independent post-hoc corrections are not supported.
- Measured H200 latency was `14.354 / 17.070 / 9.026 ms` for FA3 BF16,
  QK-INT8/PV-FP16, and QK-INT8/PV-FP8. The diagnostic FP16-PV path is not a
  deployment candidate.
- Decision SHA-256:
  `24157f3521b77169bdc73c07c9a22ba66b04404b809e8d724e29c36a34b3c722`.
- Manifest SHA-256:
  `78e86d8464c6c89afcdcb71ccdee2f2613453d3437c866135da78fe45520fb87`.
- Report SHA-256:
  `2bf2bb78c498b42788130a3580b17db1f32e6c64216973f652b04684371d2b18`.
- Figure PNG/PDF SHA-256:
  `bb91b2fe26701447b3590cb10b67374336616cd284fc76ea51375e7048d2cd9f` /
  `d9c147f755cf940e84fdf46a7ab7a9b79d3482d50b238921fe288bdaf0123654`.
- Required action: test a hardware-regular FP8 PV residual correction at fixed
  low density before any QAT. Exposed smokes close centering, fixed Hadamard,
  and finer token-block scaling; a 5% correction screen on Layer 29 passed the
  aggregate guard and Layer 0 became borderline only with a rank-8 tail.
