# RESULT-EXP-040: Causal temporal marginal-defect subspace transport

- Status: complete
- Validity: valid exploratory adaptive-capacity null
- Date: 2026-08-12
- Gate: G-019
- Claim: C-019
- Candidate: L-019

## Registered outcome

`adaptive-capacity-null`. On the fresh validation identity, target-exposed
adaptive current rank 64 reached 1.380% aggregate and 2.829% worst head-step
relative L2 across targets 8/9/10, failing the registered 0.5%/1% capacity gate.
No causal candidate was selected and the staged test identity was not loaded.

## Supports

- Adjacent-step marginal defects contain substantially more transferable
  subspace information than cross-identity static same-step defects: projector
  overlap averaged 0.877 versus 0.646.
- Previous-step rank 64 reached 1.842%/3.639%, a 51.3% aggregate improvement
  over the matched static rank-64 basis at 3.780%/9.676%.
- The remaining failure is a capacity/rank-boundary issue, not absence of any
  temporal signal: the median rank-64/65 singular ratio was only 1.021.

## Does not support

- Fixed rank-64 marginal defect transport cannot meet the registered strict
  quality gates on the screened fresh cells.
- Current-anchor innovation and two-step chordal extrapolation do not rescue the
  rank-64 boundary.
- Target-exposed support and coefficients, omitted refresh/runtime costs, and no
  rollout or kernel prohibit any deployment or speed claim.
- The previous-dense-output diagnostic is not a paper-faithful SVD-Cache result.

## Decision

Refute `C-019` and park `L-019` at rank 64. A separate bounded Gate may test a
validation-frozen per-head temporal rank frontier through rank 96 plus dense
fallback because the temporal-vs-static signal is strong and the failure is
concentrated in a flat spectral boundary. No other rescue is authorized.

## Integrity

- Sixteen fresh cond-only QKV payloads were captured completely under
  `/opt/data`; capture and probe exited 0 with zero PID overlap.
- Sixty-two related remote tests passed and all registered hashes matched.
- The manifest loaded only calibration and validation identities; `s03` test
  tensors remained unopened.
- Local and remote formal artifact SHA-256 values match.

## Evidence

- `worldfoundry_hybrid_residual/results/temporal_marginal_defect_transport_f81_l14_exp040_v1/decision.json`
- `worldfoundry_hybrid_residual/results/WAN_TEMPORAL_MARGINAL_DEFECT_TRANSPORT_EXP040_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/temporal_marginal_defect_exp040_20260812/temporal_marginal_defect_exp040.png`
