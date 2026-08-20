# PLAN-040: Static head-group Sol/FA3 H200 feasibility gate

- Status: completed
- Owner: researcher and Agent
- Gate: grouped heterogeneous attention implementation feasibility
- Claims: none; this gate measures execution cost only
- Candidate: pinned Sol-Attn `cute_sm90` fast group plus FA3 BF16 dense group
- Lane: explore
- Resource cap: one implementation iteration, at most 0.25 H200-hour on
  physical GPU3, no model download, no fresh capture, no candidate-source edit

## Decision to unlock

Determine whether a static per-layer head partition can execute one Sol sparse
group and one FA3 BF16 dense island while retaining at least `1.5x` measured
whole-attention speed versus all-head FA3 BF16 at Wan F81 native token length.
This is a necessary engineering condition for a future prospective
heterogeneous Sol policy; it is not a quality or transfer test.

## Frozen identity and inputs

- Sol candidate repository, commit, tree, environment, FA3 rebuild, and source
  hashes are identical to PLAN-039 / DIAG-037.
- Use the immutable PLAN-039 latency identity only: Layer 14, step 12,
  conditional branch, sequence length `32760`, 12 heads, head dimension 128,
  BF16 BTHD Q/K/V, and global Morton3D order `(21,30,52)`.
- Existing output errors, pass/fail labels, and per-head risk identities are
  forbidden inputs. The exposed observation that multiple dense heads may be
  needed motivates the tested *counts* but does not select head identities.
- Candidate source remains read-only. Harnesses, logs, and result manifests
  live outside the candidate checkout.

## Frozen methods

- Baseline: one FA3 BF16 call over all 12 heads.
- Fast path: released Sol API with fixed `tau=-1.5`, `diag` threshold,
  `kv_splits=1`, no sink tokens, and strict `cute_sm90` dispatch.
- Evaluate dense-head counts `[1, 2, 3, 4]`, corresponding to fast/dense head
  counts `[11+1, 10+2, 9+3, 8+4]`.
- Use contiguous head groups and concatenate outputs in the original head
  order. Include Sol preprocessing, both attention calls, concatenation, and
  synchronization in the primary latency. Exclude only the already registered
  model-level Morton permutation, exactly as in PLAN-039.
- Static head permutation may be folded into QKV/O projection weights in a
  deployment implementation, but no such folding is credited in this test.
- Run two warmups and seven alternating forward/reverse synchronized repeats.
  Report median, mean, standard deviation, coefficient of variation, and every
  raw latency sample.

## Correctness and fairness guards

- Assert Sol dispatch is `cute_sm90`; fallback is invalid evidence.
- Assert every grouped output is finite and has the same shape and dtype as
  the all-head baseline.
- Verify the concatenated result equals the independently returned fast-group
  and dense-group outputs exactly at their corresponding head slices.
- Require zero foreign process overlap on physical GPU3 and record pre/post GPU
  inventories, environment identity, source hashes, input hash, and clock
  context.
- Require timing CV `<=5%` for the all-head baseline and every grouped point.
- Do not infer quality, model coverage, rollout quality, or end-to-end speed
  from this engineering-only benchmark.

## Outcome mapping

- `grouped-runtime-pass`: the `9+3` point reaches measured `>=1.5x` with all
  guards passing. Authorize drafting, but not running, a fresh
  calibration/evaluation-separated heterogeneous quality protocol.
- `grouped-runtime-boundary`: `11+1` or `10+2` reaches `>=1.5x`, but `9+3`
  does not. Preserve the implementation frontier and require a prospective
  quality policy to prove that at most two dense heads suffice before capture.
- `grouped-runtime-null`: no grouped point reaches `1.5x`. Close the public
  two-call heterogeneous Sol path and prioritize a faithful trained fused
  baseline instead of collecting fresh policy data.
- `engineering-failure`: the public APIs cannot execute a valid grouped shape
  within the single implementation iteration. Preserve diagnostics and make no
  latency conclusion.
- `invalid`: source/input identity, backend, timing, isolation, or correctness
  guards differ from this plan.

## Stop rules

- Run a small-shape grouped correctness smoke before the native-length timing.
- Stop immediately on fallback, nonfinite output, wrong input identity, GPU
  overlap, timing guard failure, or resource-cap breach.
- Do not inspect quality records, choose head identities, change tau, add a
  third approximate branch, tune tile policy, capture new QKV, or run rollout.
- Close after the four-point timing table; do not optimize a custom fused
  multi-group kernel inside this Gate.

## Provenance-only rerun note

- The first complete execution on `2026-08-12` produced a stable
  `grouped-runtime-null` table, but the runner omitted the preregistered GPU
  clock/P-state context even though dual idle checks, same-run FA3 baseline,
  alternating order, zero foreign overlap, and all numerical guards passed.
- Preserve that execution as an engineering diagnostic. Before treating the
  Gate as formal evidence, run exactly one provenance-only repeat with the
  identical config, payload, probe hash, method order, warmups, repetitions,
  and outcome mapping. Add only pre/post `nvidia-smi` clock, performance,
  power, and temperature records.
- This authorization was recorded before the repeat. The repeated result is
  adopted regardless of whether it improves or worsens the first table; no
  method, threshold, head count, timing boundary, or quality input may change.

## Closure

- Closed date: `2026-08-12`.
- Outcome: `grouped-runtime-null`.
- Formal FA3 BF16 baseline: `15.6741 ms` median.
- Formal grouped points: `11+1 = 1.3786x`, `10+2 = 1.3464x`,
  `9+3 = 1.2798x`, and `8+4 = 1.2419x`.
- The registered `9+3` decision point missed the `1.5x` gate by `1.7976 ms`;
  no tested grouped point reached the gate.
- Validity: strict `cute_sm90`, finite/shape/dtype/slice guards, candidate and
  payload identities, zero foreign-PID overlap, P0/clock context, and every
  timing CV below 2% passed in the formal provenance-only repeat.
- Boundary: no quality record was read and no fresh capture was made. This
  closes only the public two-call Sol/FA3 composition, not fused heterogeneous
  Attention, model quality, transfer, rollout, or end-to-end acceleration.
- Action: park this public two-call path and audit a pinned public fused dynamic
  sparse baseline before any new quality experiment.
- Evidence:
  `worldfoundry_hybrid_residual/results/WAN_STATIC_HEAD_GROUP_SOL_FA3_DIAG038_20260812.zh-CN.md`.
