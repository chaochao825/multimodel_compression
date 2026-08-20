# PLAN-033: V-mass shaped-tail basis transfer and observability

- Status: completed
- Owner: researcher and Agent
- Gate: rank-8 basis transfer after regular V-residual shaping
- Claims: none; every tail coefficient is target-projected
- Candidate line: Sage FP8 bulk plus 5% regular V correction plus rank-bounded tail
- Lane: explore
- Resource cap: immutable DIAG-030 captures, one implementation repair, and one
  exclusive H200 pass under 60 minutes

## Decision to unlock

Determine whether the DIAG-032 `v_mass5` shaped residual admits either a
calibration-fixed or current-content-observable rank-8 output-channel basis on
held-out prompt/seed identities. A pass authorizes only a bounded coefficient
predictor gate. It does not authorize QAT, CUDA, rollout, or a speed claim.

## Immutable scope and split

- Reuse DIAG-030 captures and the DIAG-032 scope without modification: layers
  `[0,29]`, steps `[2,7,12,17]`, both CFG branches, 16 deterministic 64-query
  tiles, 12 heads, 128 channels, and 32,760 tokens.
- Calibration identities are `s00_p00_seed20260860` and
  `s01_p01_seed20260860`. Evaluation identities are
  `s02_p00_seed20260861`, `s03_p01_seed20260861`,
  `s04_p02_seed20260861`, and `s05_p03_seed20260862`.
- Exact `v_mass5` selects 13 of 256 regular 128-key blocks. This remains an
  expensive mechanism observer and is frozen only to isolate the tail basis.
- Build the calibration basis bank, write it, and hash it before loading any
  evaluation QKV tensor. Reading evaluation path metadata is allowed; reading
  evaluation tensor payloads before the freeze receipt is invalid.

## Frozen basis families

Let `R = Y_dense - (Y_Sage + C_v_mass5)` be the shaped residual. Every basis has
orthonormal rows and every reported tail uses the target-exposed optimal row
coefficients `R B^T`, so the corrected tail is `R B^T B`.

1. `adaptive_rank8`: per-sample SVD of `R`; inherited target-exposed capacity
   reference, now using mass support rather than oracle support.
2. `frozen_cell_rank8`: calibration Gram pooled by layer, step, and head across
   identities, branches, and all 16 query tiles.
3. `frozen_tile_rank8`: calibration Gram pooled by layer, step, head, and query
   tile across identities and branches.
4. `frozen_tile_rank16`: pre-registered boundary diagnostic; it cannot count as
   a rank-8 deployment pass.
5. `q_basis_rank8`: right singular basis of the current query tile.
6. `correction_basis_rank8`: right singular basis of current `C_v_mass5`.
7. `sage_basis_rank8`: right singular basis of the current Sage output.
8. `observed_basis_rank8`: right singular basis of the row concatenation of
   current Q, Sage output, and `C_v_mass5`.

The last four use no dense target to construct the basis but still use the
dense target to fit projection coefficients. Their SVD cost is unpriced and
they test observability only.

## Metrics and gates

For every split/layer/step/method, report aggregate, worst-head, and worst-tile
relative L2. Also report shaped-tail captured energy, overlap with the adaptive
rank-8 target subspace, basis source/granularity, dynamic-SVD requirement, and
static basis bytes.

Quality guards remain:

- aggregate output relative L2 `<=1%`;
- worst-head output relative L2 `<=2%`;
- worst-tile output relative L2 `<=2%`.

Outcome precedence after invalid/adverse/contradictory checks:

- `frozen-cell-rank8-pass`: the cell-pooled frozen rank-8 basis passes every
  evaluation cell. Open one coefficient-prediction gate.
- `frozen-tile-rank8-pass`: the tile-specific frozen rank-8 basis passes every
  evaluation cell. Open one tile-conditioned coefficient-prediction gate.
- `observable-rank8-pass`: at least one current-content basis family passes
  every evaluation cell. Open one low-cost basis/coordinate approximation gate;
  do not deploy online SVD.
- `frozen-rank16-boundary`: no rank-8 family passes, but frozen tile rank-16
  passes every cell. Retain only a cost-bounded rank-16 boundary candidate.
- `adaptive-capacity-regression`: mass support plus adaptive rank-8 fails any
  evaluation cell. Close the DIAG-032 hybrid result as oracle-support-specific.
- `basis-transfer-null`: adaptive rank-8 passes but every frozen/observable
  basis fails. Stop the train-free tail and proceed to one small QAT/LoRA gate.

## Cost and claim boundary

No H200 speed is measured in this diagnostic. Exact probability mass, selected
exact V correction, target-projected coefficients, online SVD, routing,
normalization, fusion, and fallback are all outside the cost claim. Rank-16 is
only a scientific boundary. A numerical pass cannot be described as a kernel,
rollout, or end-to-end result.

## Stop rules

Stop after one valid pass or the one allowed implementation repair. Do not
change density, ranks, basis granularity, identities, query tiles, quality
gates, or method order after evaluation. Do not train, write CUDA, run rollout,
add BCM/BCCB/Butterfly/cache/temporal predictors, or tune on evaluation.

## Frozen implementation

- Config SHA-256: `959300d094786765b620872e0946a484c3c811ec836c4e57101518223d761402`.
- Basis core SHA-256: `96249a0fe2b86042d002855918139413680abd47215da47db945d655e2176ff1`.
- Core test SHA-256: `2e5e05eac263d532ec6c6ed5bfb20b94866370bd029eecdf7d82ef5d638654b9`.
- Probe SHA-256: `4d3736cfa4a39be592812dba071fa2bbd0d01d29909e8385522ee5963e783650`.
- Runner SHA-256: `5b658a744c75564f5451e381c9c391c67432c4b097df9a630500397c76e14d7b`.
- DIAG-032 probe/core SHA-256:
  `a17d529d569b8818d38476ec459f0a96710fe08af0f026756f7d03171ddaf6f3` /
  `be275347dd9fdd530100fcf69c14ca86419d9927c4db1801548040d0be8c8bb1`.
- DIAG-031 helper/core and dense evaluator SHA-256:
  `76c9d0591046fea918bb84011ebc84d4046d01358ae283cd0a543c177c263b74` /
  `62a798b1c0097c5dcf7eaa3ba5fc070253c68563c98101b30068245e033ad85c` /
  `e924b16afe4cf90dffaa57b6f1a16abaaa55d24bad743158823b894c142a74e4`.
- DIAG-030 capture provenance SHA-256:
  `f56604ea92906b1ea864d5002bb2bcef528d34b26fa4fe35b362bc414ce9adb2`.
- Remote pure-function tests passed `6/6` before execution. The first formal
  attempt was stopped while still evaluating calibration identities: a source
  audit found that slicing the first eight rows of an ascending-eigenvalue
  rank-16 bank selected directions 9--16 rather than the top eight. No held-out
  payload had been loaded. The sole implementation repair makes Gram bases
  descending and adds a nested top-8/top-16 regression test; the repaired suite
  contains `7` tests. The invalid partial output is preserved under
  `/opt/data/wangmeiqi/trash/20260812-094000-diag033-launch-timeout/` and the
  subsequent correctness-stop artifact is preserved in a separate trash entry.

## Required closure

Archive source/capture hashes, calibration basis bank and freeze receipt,
per-head/tile records, cell and identity summaries, basis overlaps, decision,
one visualization, and one Chinese report. Update `STATUS.md` and open at most
one coefficient-predictor or QAT gate selected by the frozen outcome.

Closed 2026-08-12 with outcome `basis-transfer-null`.

- The repaired formal run processed 32 calibration captures before freezing a
  12,978,777-byte basis bank, then evaluated all 96 captures in 288.41 seconds.
  Tests passed `7/7`, the foreign-PID audit was empty, and the freeze receipt
  records `evaluation_payloads_loaded_before_freeze=false`.
- Adaptive rank-8 after exact `v_mass5` passed all 8 evaluation cells, so the
  DIAG-032 capacity result did not regress. Frozen cell rank-8 and frozen tile
  rank-16 each passed 7/8; frozen tile rank-8 passed 6/8; no observable rank-8
  family passed all cells.
- The unique hard cell was Layer 29 step 12. Frozen cell rank-8 reached
  `0.965% / 1.507% / 5.139%`; Sage-output rank-8 reached
  `0.782% / 1.154% / 3.652%`; adaptive rank-8 reached
  `0.325% / 0.689% / 0.419%`.
- The hard event was concentrated in held-out `s05`, head 9, primarily query
  tile 7 and tile 0. On `s05` cond head9/tile7, frozen-cell overlap with the
  adaptive basis was only `0.051`, while Sage-output overlap was `0.140`.
  Increasing the static rank to 16 did not fix the event.
- Basis bank/freeze receipt SHA-256:
  `8c5559a447d56853166b589b88a901fafffd39493eecafeae90ca1185bdb08a3` /
  `84d267f846b873b3c8bfa871eb0ebba9c6e8650d313e198e108e42b3cd7ce606`.
- Basis/cell/identity records SHA-256:
  `9ef8fd7097f104d6dead35bb2a905f9206f3c887a017f11e9af418e80400c68f` /
  `ec388a3c008b6b5cd30b5386535111da1faa6d0c18617542dcf84091fcdb84b0` /
  `ab731f39a344ab5910b6cdfba27d5b9dd22a973a68be6f546bc6f8fe1ecafa0d`.
- Decision/manifest SHA-256:
  `13efa1fe0bee22a1f0f739eb4096bda70e69a11bcc1fb69b624dffcc17d563ec` /
  `2c59dc2df6db1f62d05c52ef31b16de6f0a207cf3d5accdcaffddf0e947ac4c0`.
- Report SHA-256:
  `51c04a4bb4fbbc6f69d34ab6f9684215b67d573b4f5abf1db61910322442119f`.
- Figure PNG/PDF SHA-256:
  `5502a2ceccc64fa5d43bf70cd012093f63a43b7ab3e10600c2c19cd622ad67d0` /
  `fffd97a41cab950567ca4540c93ff495430e14f4c49267fac3384f855bed58b7`.
- Required action: stop ordinary train-free basis banks and open one small,
  foldable V/PV quantization-aware adaptation gate. Any hard-cell fallback
  must be frozen prospectively and reported separately.
