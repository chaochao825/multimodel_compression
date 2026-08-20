# PLAN-030: Fresh stratified precision-island transfer atlas

- Status: completed
- Owner: researcher and Agent
- Gate: calibration-only static-atlas transfer
- Claims: none until prospective evaluation passes
- Candidate line: output-risk-certified dense attention precision islands
- Lane: explore
- Resource cap: one 6-trajectory capture campaign, at most 60 GiB of QKV,
  one implementation repair, and one exclusive H200 probe under two hours

## Decision to unlock

Determine whether DIAG-029's one-head precision island reflects transferable
head-risk structure or only an exposed Layer-14 cell. A pass is required before
rollout, QAT, or custom heterogeneous-kernel development.

## Fresh split and stratification

- Model and resolution remain Wan2.1-T2V-1.3B F81 at `480x832`, 20-step UniPC,
  shift `5.0`, guide scale `5.0`.
- Capture layers `[0,14,29]`, sampling steps `[2,7,12,17]`, and both `cond` and
  `uncond` branches. Stop after step 17. This produces `144` QKV files from
  `3 layers x 4 steps x 2 branches x 6 identities`.
- Calibration identities are fixed to `s00_p00_seed20260860` and
  `s01_p01_seed20260860`.
- Evaluation identities are fixed to `s02_p00_seed20260861`,
  `s03_p01_seed20260861`, `s04_p02_seed20260861`, and
  `s05_p03_seed20260862`. The first pair isolates new-seed transfer on seen
  prompts; the second pair adds unseen prompts. No evaluation tensor may
  influence head selection, thresholds, fallback, or latency mapping.
- Quality uses 16 deterministic 64-query tiles per capture, all 12 heads, and
  the same FP32 dense reference as DIAG-027--029.

## Frozen atlas builder

- For every `layer x sampling-step` cell, share one precision policy across
  both CFG branches.
- Evaluate SageAttention 2.2.0 SM90 `per_thread`, `smooth_k=True`,
  `pv_accum_dtype=fp32+fp32` and FA3 BF16 on calibration only.
- Enumerate every BF16 risk-head subset with cardinality 0, 1, or 2. Mixed
  output error is assembled exactly from independent head outputs before the
  O projection.
- A subset is calibration-safe only when every calibration
  `identity x branch` satisfies aggregate `<=0.8%`, worst-head `<=1.6%`, and
  worst-tile `<=1.6%`.
- Select the smallest safe subset. Within the same cardinality, minimize the
  maximum normalized gate ratio; break exact ties lexicographically.
- If no subset with at most two BF16 heads is calibration-safe, freeze that
  layer-step cell to full FA3 BF16 fallback. Do not increase the head budget or
  relax a threshold after viewing evaluation data.

## Frozen latency model

- On one registered full 32,760-token evaluation capture, measure risk counts
  `0`, `1`, `2`, and `12`. Count 0 is all-head Sage; count 12 is full FA3 BF16.
- Counts 1 and 2 use contiguous safe/risk groups, actual Sage and FA3 calls,
  and timed output concatenation. Static permutations are materialized before
  timing because deployment can fold them into Q/K/V and O projections.
- Use two warmups and five alternating forward/reverse wall and CUDA-event
  repetitions. The atlas speed is FA3 median divided by the equal-cell-weighted
  latency implied by the frozen 12-cell calibration atlas.

## Evaluation and outcome map

- A low-precision-certified cell is a false safe if any frozen evaluation
  `identity x branch` exceeds aggregate `1%`, worst-head `2%`, or worst-tile
  `2%`. BF16-fallback cells are still evaluated as numerical guards.
- `atlas-transfer-pass`: zero false-safe cells, all policy outputs finite, and
  predicted sampled-atlas attention speed `>=1.4x`. Authorize one prospective
  projection-folded fused-kernel design gate, followed by rollout only after
  kernel validation.
- `atlas-quality-only`: zero false-safe cells but sampled-atlas speed `<1.4x`.
  Keep the quality certificate, stop kernel work, and prefer low-cost QAT or a
  less strict deployment objective.
- `atlas-false-certification`: at least one low-precision-certified evaluation
  cell fails. Reject the current static calibration rule; do not tune on the
  failures. Prefer QAT or exact BF16 fallback.
- `atlas-null`: no low-precision cell is certified or policy guards fail.
  Close train-free static precision islands.
- Standard invalid, contradictory, adverse, and engineering-failure outcomes
  retain precedence.

## Stop rules

Stop after one valid capture and probe or the one allowed engineering repair.
Do not change prompts, identities, layers, steps, branches, tile count,
calibration margins, head budget, tie break, or speed threshold after capture.
Do not run rollout, VBench, QAT, or write a custom CUDA kernel in this plan.

## Frozen implementation

- Config SHA-256: `dcb104e0fb36ee030a713efcd942079c6dd4847c3f34fc5e446436dc7448419d`.
- Prompt-file SHA-256: `02cb200b5fd6c5687daa56814b6d6248e77b55ad9170a319489c5d4f2fb35c7f`.
- Wan QKV capture SHA-256: `379265f7ba79c2352e91786ad356e3709e37d77f4d54aeb0e584de505d418768`.
- Atlas-core SHA-256: `87df630e3c9a0468033ac8fe576297bc4df876e0a6c75d69a28d8768b8afa7fb`.
- Atlas-core test SHA-256: `af952072cb5ddfc4630e118098e2671dd26e448adc6b7aa2466cac0530142c41`.
- Probe SHA-256: `0adc362488b9e8978fd07d5879ff778362cbd6b0c20b1ac4c3096eb896a140b7`.
- Runner SHA-256: `ae9947b2dc0edf52b5872a3b57cc187abb4e788ec4461d54881dbb96df786c01`.
- Latency-core SHA-256: `668df9dc73a059368eba7d8f9c550773672ee5c64666df051b59eea3a5e45d8d`.
- Dense-reference core SHA-256: `e924b16afe4cf90dffaa57b6f1a16abaaa55d24bad743158823b894c142a74e4`.
- Artifact-utility probe SHA-256: `8f32cae100894954fd21584f2e15eb942184a0c213ae2bec942cb87d288887bc`.
- SageAttention core SHA-256: `e475182fc55d6a683499e1eb4d8886fc06e050ed938d8563b090773eede2a99e`.
- FA3 wrapper SHA-256: `53813a192b9d64b49cf72179aadbce39c67c70c00cae2331ed273887c6a58af9`.
- Local and remote atlas tests passed `5/5`; remote shell syntax passed.
- A real H200 smoke produced 192 finite energy rows from 16 query tiles and
  finite latency outputs for BF16 risk counts `0/1/2/12`.

## Closure

Write a transfer heatmap and report, archive all capture and implementation
hashes, update `STATUS.md`, and change protected registries only if the
prospective evaluation supports a narrowly worded claim.

Closed 2026-08-12 with outcome `atlas-quality-only`.

- The runner completed in 457 seconds with 144/144 QKV captures totaling
  `43,476,415,248` bytes, per-file SHA-256 provenance, and zero foreign GPU
  PID observations.
- The calibration atlas was frozen before any evaluation capture was loaded;
  its SHA-256 is
  `c3d55b7195c651e9014a4e0d1ee848d35eaf6c06b81fd950ee0fc47d2cc10218`.
- Four Layer-14 cells were certified: head 4 at steps 2/7 and heads 0/4 at
  steps 12/17. Layer 0 and 29 cells required full BF16 fallback.
- All 96 evaluation policy summaries passed and no certified cell was false
  safe, but only `4/12` atlas cells used low precision.
- Measured risk-count speeds were `1.5906x / 1.4907x / 1.4290x / 1.0000x`
  for `0/1/2/12` BF16 heads. The frozen atlas reached only `1.1172x`, below
  the `1.4x` gate.
- Decision SHA-256:
  `29a95cfbf286e50e5fd2b4a2fcfc27f81c7de3227569958572a614a2d97e2ba0`.
- Manifest SHA-256:
  `3765af3fbbcc68089e43ba90ff3bc6dd6e07cbf718faeac6a042491c101d7de7`.
- Report SHA-256:
  `862f69d1ab01bc853ded638868edfe0d666fbcff845b1bdb830723776a39ab3f`.
- Figure PNG SHA-256:
  `772c233d9ca9667bb12f5415a90a05d0d49b9970c9f5b9d87fb7d0346ef3028f`.
- Report:
  `worldfoundry_hybrid_residual/results/WAN_PRECISION_ISLAND_TRANSFER_ATLAS_DIAG030_20260812.zh-CN.md`.
- Required action: stop fused-kernel and rollout work. Localize Layer-0/29
  Q/K-versus-V/PV quantization error before one bounded low-cost QAT gate.
