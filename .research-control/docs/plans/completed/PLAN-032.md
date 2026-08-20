# PLAN-032: Hardware-tile FP8 PV residual capacity atlas

- Status: completed
- Owner: researcher and Agent
- Gate: sparse quantization-residual capacity before selector training or QAT
- Claims: none; target-exposed methods are oracle diagnostics
- Candidate line: FP8 dense bulk plus regular high-precision PV correction
- Lane: explore
- Resource cap: immutable DIAG-030 captures, one implementation repair, and
  one exclusive H200 pass under 60 minutes

## Decision to unlock

Determine whether the actual Sage `S_FP8 x V_FP8` output defect on unsafe
Layer-0/29 cells can be reduced below the local quality boundary by correcting
at most 13 of 256 regular `64-query x 128-key` tiles per head, optionally
followed by an adaptive rank-8 output tail. A pass only authorizes a subsequent
calibration-transfer selector/basis gate.

## Immutable scope

- Reuse DIAG-030 captures and its calibration/evaluation identities without
  modification.
- Evaluate layers `[0,29]`, steps `[2,7,12,17]`, both CFG branches, all six
  identities, and the same 16 deterministic 64-query tiles.
- The 128-key block matches the SM90 PV kernel tile. Pad only the final eight
  keys with zero values and masked logits.
- Reference is the existing FP32 dense evaluator. Baseline is the actual
  per-thread smooth-K Sage SM90 worker, not a fake-quant surrogate.

## Frozen correction dictionaries

For every query tile and head, construct 256 regular candidate corrections.

1. `v_residual`: exact dense probability for a block multiplied by
   `V - V_E4M3`.
2. `pv_residual`: exact block contribution minus an online-softmax E4M3
   numerator and E4M3-V block contribution. Running max and high-precision
   denominator follow the inspected Sage kernel order, but Q/K scores remain
   FP32; this dictionary is therefore a mechanism model, not an exact kernel
   decomposition.

Apply corrections to the actual Sage output. Compare the following fixed
methods at exactly 13 blocks (`5.078125%`) unless stated otherwise:

- `stock_sage`;
- `v_all` and `pv_all`, diagnostic full-dictionary sums;
- `v_mass5`, `v_norm5`, and target-exposed `v_oracle5`;
- `pv_mass5`, `pv_norm5`, and target-exposed `pv_oracle5`;
- adaptive per-head/per-tile rank-8 of stock defect;
- adaptive rank-8 after `v_oracle5` and after `pv_oracle5`.

`mass5` ranks by exact block probability mass. `norm5` ranks by candidate
correction energy and does not read the dense target, though its current
implementation is not yet a cheap router. `oracle5` greedily chooses only
positive reductions in the actual dense-relative defect and may stop before
13 blocks. Oracle and adaptive-rank methods are target-exposed capacity bounds.

## Metrics and gates

Report aggregate, worst head, and worst tile relative L2 separately for every
split, layer, and step. Also report selected-block count, correction/full-defect
cosine, optimistic added PV work, and the gap from norm/mass to oracle.

- `sparse-capacity-pass`: one 5% sparse method reaches aggregate `<=1%`, worst
  head `<=2%`, and worst tile `<=2%` on every evaluation layer-step cell.
- `hybrid-capacity-pass`: sparse alone fails, but one sparse-plus-rank8 method
  reaches the same boundary on every evaluation cell. Authorize only frozen
  basis/router transfer testing; do not claim deployability.
- `layer29-only-boundary`: a method passes all Layer-29 cells but any Layer-0
  cell fails. Retain a heterogeneous Layer-29 candidate and assign Layer 0 to
  a different precision/QAT path.
- `capacity-null`: neither layer passes consistently. Stop sparse PV residual
  correction and proceed to low-cost quantization-aware adaptation or BF16.
- Invalid, contradictory, adverse, and engineering-failure outcomes retain
  precedence.

## Cost boundary

Five percent is frozen because the exposed two-cell screen plateaued by 5%; a
10% mass/norm correction worsened both cells due dictionary mismatch. No density
sweep is allowed. Rank-8 cost is reported only as arithmetic; basis generation,
routing, exact selected logits, normalization, and kernel integration remain
unpriced and prevent a speed claim.

## Stop rules

Stop after one valid pass or one implementation repair. Do not train, fit any
evaluation basis/router, alter density/rank after evaluation, write CUDA, run
rollout, or add BCM/BCCB/Butterfly/cache/temporal predictors.

## Frozen implementation

- Config SHA-256: `314e16fce47cfab33a3fc5c97259b388f29cb13849553cb20fa459594688d07a`.
- Capacity core SHA-256: `be275347dd9fdd530100fcf69c14ca86419d9927c4db1801548040d0be8c8bb1`.
- Core test SHA-256: `a3d299968382dd0b107e7bac87b1715c06e036d19458e7213dbfba78be63007e`.
- Probe SHA-256: `a17d529d569b8818d38476ec459f0a96710fe08af0f026756f7d03171ddaf6f3`.
- Runner SHA-256: `3e979e8edd8fbcde16169ee7e5cba9921996697d38019ec40e56f9357a6045ee`.
- DIAG-031 helper/core SHA-256:
  `76c9d0591046fea918bb84011ebc84d4046d01358ae283cd0a543c177c263b74` /
  `62a798b1c0097c5dcf7eaa3ba5fc070253c68563c98101b30068245e033ad85c`.
- DIAG-030 capture provenance SHA-256:
  `f56604ea92906b1ea864d5002bb2bcef528d34b26fa4fe35b362bc414ce9adb2`.
- Remote tests passed `5/5`. A real 16-tile Layer-29 smoke produced 2,304
  finite rows; `v_mass5` reached `0.714% / 1.408% / 1.070%`, while
  `pv_all` exposed dictionary mismatch at `5.695%` worst-head error. This
  smoke is engineering evidence only and cannot change the formal gate.

## Required closure

Archive source/capture hashes, raw summary tables, selector/oracle gaps, one
capacity heatmap, and one Chinese report. Update `STATUS.md` and open at most
one transfer or QAT gate selected by the outcome.

Closed 2026-08-12 with outcome `hybrid-capacity-pass`.

- All 96 immutable captures produced finite records in 182.34 seconds; remote
  pure-function tests passed `5/5` and the foreign-PID audit remained empty.
- `v_mass5` alone passed 4/8 evaluation cells. Adaptive rank-8 on the raw Sage
  defect also failed most cells. Target-exposed `v_oracle5_rank8` passed all
  eight cells at `0.229--0.799%` aggregate, `0.301--1.161%` worst-head, and
  `0.311--0.852%` worst-tile error.
- `v_mass5`, `v_norm5`, and `v_oracle5` were nearly identical; the evaluation
  aggregate mass-to-oracle relative gap averaged about `0.088%`. Five percent
  V correction was also nearly identical to correcting all 256 blocks. The
  next bottleneck is therefore tail transfer, not selector capacity.
- Layer-29 step-12 retained a `5.549%` worst-tile error after `v_mass5`, but the
  target-exposed hybrid reduced it to `0.418%`. Aggregate-only reporting would
  hide this heavy-tail failure mode.
- Decision SHA-256:
  `dc597ea6b1e0afdc49d6a57e90d5516e7a8964299a5d6ae1f5ee5753ddae64a4`.
- Capacity/cell/identity record SHA-256:
  `86933e78b4296db3d9ad657b5c4d4dadb0de371154d7c07a19c1516763f3e270` /
  `81c3477e4e722d27dedfaac626242b5a97d949a586fffcf0c8d891b481c81ed9` /
  `9e02baf3819470917769936f9f2bdaa8395b015cabe58b05ee9ee5e9f48ac66d`.
- Manifest SHA-256:
  `7cd971bbb885b6e53b2081548820f4529574b918bd127f1d181c38ebb9c4b390`.
- Report SHA-256:
  `1f86c735f2a09897483b44bdde1596269236aff5f42941e91103e47a8e6e060`.
- Revised figure PNG/PDF SHA-256:
  `7ed443b0c7448809487ee1fd7f5b801011dfc392bd176f0a2833c641f907d0ca` /
  `02f18827b40cb89a41cd9c03089a844bf80a9a061558a3589f1e6545825f6cc8`.
- Required action: freeze `v_mass5`, then test calibration-only and
  content-observable rank-8 output-channel bases before any coefficient
  training, QAT, kernel implementation, or rollout.
