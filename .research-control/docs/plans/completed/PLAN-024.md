# PLAN-024: Audit the operator-faithful SLA residual projection

- Status: completed
- Owner: researcher and Agent
- Gate: fully exposed pre-gate diagnostic
- Claims: none
- Candidate line: paper-faithful baseline bridge before fresh adaptation
- Lane: explore
- Resource cap: one implementation, one repair, and one exclusive H200 pass under one hour

## Decision to unlock

Determine whether the public SLA operator's core function class has enough
local Wan `AV` capacity under frozen QKV to justify fresh-data sparse-linear
adaptation. Separate three explanations for prior failures: an incorrect tail
objective, an insufficient output projection, and cross-identity transfer.

## Source authority and exact operator

Use the public `thu-ml/SLA` main-branch operator semantics as of the recorded
source revision:

1. smooth K by subtracting its token mean;
2. mean-pool Q/K into `64 x 64` blocks and retain the top 20% key blocks for
   each query block;
3. compute exact softmax attention over the retained keys, normalized within
   the sparse branch;
4. compute softmax-feature linear attention over all keys;
5. add a learned affine projection of the linear output to the sparse output.

For query row `i`:

\[
\widehat Y_i=Y_i^{\rm sparse}+W\,Y_i^{\rm linear}+b.
\]

This intentionally preserves SLA-1 branch normalization rather than silently
replacing it with the project's shared-N/Z formulation. It is an operator-level
baseline, not a full paper reproduction of model fine-tuning or fused kernels.

## Frozen data and split

- Reuse immutable EXP-040 post-RoPE Q/K/V captures only.
- Cell: Wan2.1-T2V-1.3B F81, Layer 14, step 9, conditional branch.
- Identities: `s00/s01` calibration, `s02` validation, `s03` test.
- All four identities are outcome-exposed; every result is development-only.
- Evaluate the 16 deterministic 64-query tiles registered by EXP-040 and all
  12 heads. Do not change tiles after reading outcomes.

## Registered methods

1. `sparse_only`: official 20% pooled-Q/K block map and sparse branch.
2. `sla_shared_calibration`: one affine 128-to-128 projection shared by all
   heads, fitted on `s00/s01` only.
3. `sla_per_head_calibration`: one affine projection per head, fitted on
   `s00/s01`; diagnostic extension isolating head heterogeneity.
4. `sla_per_head_loio`: leave-one-identity-out per-head projection.
5. `sla_per_head_same_identity`: target-exposed per-head capacity ceiling.
6. `residual_svd_rank16/32`: target-exposed posterior lower bounds on the
   sparse residual, reported only to locate remaining intrinsic rank.

Affine maps use one fixed training-only RMS scale, one fixed ridge fraction
`1e-4`, and an explicit bias matching the public SLA module. No feature-map,
density, block-size, support, rank, or regularizer sweep is allowed.

## Validation and fairness

- Recompute dense reference and sparse/linear outputs from the same QKV tensor.
- Verify dense tiled reconstruction against the capture dispatcher within
  `1e-4` relative L2 when a saved dense output is available.
- Every calibration and LOIO map excludes the evaluated test identity.
- Report per identity, head, tile, aggregate, worst head, and worst tile.
- Record official-source URL/revision, code/config/data hashes, package/device
  versions, elapsed time, and optimistic arithmetic ratio.
- The arithmetic ratio is not latency and excludes routing, gathers, memory,
  launches, fusion, and complete rollout.

## Outcome map

- `operator-capacity-null`: same-identity per-head projection misses 1% / 2%.
  Do not collect fresh data for this operator; retain fused FP8 and external
  paper baselines.
- `adaptation-boundary`: same-identity passes 1% / 2%, but calibration and LOIO
  fail. Authorize a fresh-capture, low-cost QKV/branch-ratio adaptation Gate.
- `head-transfer-boundary`: per-head calibration passes while the public shared
  projection fails. Any successor must budget head-conditioned parameters.
- `operator-pass`: shared calibration reaches 1% / 2% and optimistic arithmetic
  remains below 30% dense. Then benchmark the official kernel before rollout.
- Standard invalid, engineering-failure, and adverse classifications retain
  precedence.

## Stop rules

Stop after the first valid outcome or the one allowed implementation repair.
Do not add a nonlinear gate, QKV LoRA, shared-N/Z correction, value sketch,
support oracle, or a second feature map after seeing the result. Those changes
require a new prospective plan. No rollout, VBench, or speed claim is allowed.

## Frozen implementation

- Official SLA source revision: `7db4039111a9002f900be537c2b2061d26d73744`.
- Core SHA-256: `ebaf6e879007ba76e4bf017ae13e6972204f51f3231484f54ecc29a746de8c84`.
- Probe SHA-256: `b0f5861150705ca16d5f5f5406e3fa210e1414ba10b1ca2ef9c2537bbfe7c455`.
- Test SHA-256: `3af5b33755355454058c579944675bc540dd0f6bcac3676202528923d41baec1`.
- Config SHA-256: `4f0ecc19e9e6159586b4b746d93ea8d28aa01debf41dc770a6c749d1b8b16164`.
- Remote and local hashes match; the remote PyTorch environment passes all
  eight registered operator tests.
- The one allowed repair corrected a mistaken unit-test expectation for the
  partial-block smooth-K example. It did not alter implementation semantics.
- A method passes the `1% / 2%` Gate only when aggregate error is at most 1%
  and both worst-head and worst-tile aggregate errors are at most 2%.
- The probe evaluates the public operator function class in FP32 from the
  captured BF16 values. This is an optimistic capacity ceiling, not a bit-exact
  reproduction of the public BF16/Triton kernel.

## Closure

Write a bound report and visualization, update `STATUS.md`, and keep claim,
candidate, and experiment registries unchanged because this is an exposed-data
baseline diagnostic.

Closed 2026-08-12 with outcome `operator-capacity-null`.

- Same-identity per-head affine: `12.692%` aggregate, `26.419%` worst head,
  and `20.027%` worst tile; the registered `1% / 2%` Gate failed.
- Held-out per-head calibration: `77.875% / 178.997% / 85.292%`.
- Target-exposed residual SVD rank 16/32 remained at `8.155% / 5.869%`
  aggregate, locating a broad residual rather than a small channel tail.
- Optimistic arithmetic ratio was `20.528%` dense, but quality failure prevents
  any latency or speed interpretation.
- Formal H200 run exited zero; 8/8 tests and all artifact/hash checks passed.
- Decision SHA-256:
  `a1d00ad2e7dff6799e8cae51f3cd821a3a9bc6927615a924cce65284ff7616d7`.
- Manifest SHA-256:
  `3fc043fc028eabcf7a64ed2966e80d1ff555272d38da3058cd3d877a6e38d91b`.
- Evidence:
  `worldfoundry_hybrid_residual/results/operator_faithful_sla_f81_l14s9_diag024_v1/`.
- Report:
  `worldfoundry_hybrid_residual/results/WAN_OPERATOR_FAITHFUL_SLA_DIAG024_20260812.zh-CN.md`.
- Required action: `TERMINATE_SLA_OPERATOR_ADAPTATION`; retain SLA as an
  external trained baseline and keep fused FP8/BF16 dense attention as the
  system anchor.
