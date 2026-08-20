# RESULT-EXP-010: Rank-shaped regular-support oracle

- Status: complete
- Validity: valid exploratory null
- Date: 2026-08-12
- Gate: G-010
- Claim: C-010
- Candidate: L-010

## Registered outcome

`null`. Validation selected `tail_swap_from_mass_top` at all three densities.
On the independent test identity, adaptive rank 16 reached:

- 12.5% support: 3.573% aggregate / 6.259% worst-head error;
- 18.75% support: 2.480% / 4.077%;
- 25% support: 1.803% / 2.986%.

No registered density reached the 1% / 2% deployment boundary, so the strict
0.5% / 1% claim is refuted. Adaptive rank 32 reached 1.135% / 1.965% at
18.75% and 0.791% / 1.275% at 25%, which is a posterior capacity diagnostic,
not an executable acceleration result.

## Supports

- Rank-aware regular-support selection improves the pooled-Q/K baseline. At
  25%, rank-16 aggregate/worst-head error improved from 2.120% / 3.872% to
  1.803% / 2.986%.
- Exact mass is the strongest transferable initialization among the registered
  choices. Spectral swaps provide additional gains mainly on a small number of
  difficult head-tiles.
- The residual-rank bottleneck lies between 16 and 32 for this cell; basis
  generation alone cannot cross it.

## Does not support

- It does not support further work on 25% regular support plus a rank-16
  correction for this cell.
- It does not establish a deployable selector: test swaps inspect the dense
  target by design.
- It does not establish kernel speed, rollout quality, or model-wide coverage.

## Decision

Park `L-010` and refute `C-010`. Do not add more fixed support families,
rank-16 rotations, BCM/BCCB experts, or kernel work to this target. A revival
must change the function class: either low-cost sparse-linear adaptation with
frozen QKV, or a separately gated bounded high-rank innovation branch whose
oracle first passes the quality and cost guards.

## Integrity

- Five EXP-010 tests passed before the valid run.
- The one registered pre-outcome repair relaxed only an over-tight FP32
  reduction-order guard from 2e-5 to 1e-4; no endpoint or method changed.
- All 432 swap traces are monotone non-increasing and all outputs are finite.
- `SUCCESS.json`, manifest, protocol, probe, core, and test hashes match.
- Runtime was 366.25 seconds on exclusive physical H200 GPU2; no latency
  benchmark was performed.

## Evidence

- `worldfoundry_hybrid_residual/results/rank_shaped_regular_support_f81_l14s9_exp010_v1/decision.json`
- `worldfoundry_hybrid_residual/results/rank_shaped_regular_support_f81_l14s9_exp010_v1/manifest.json`
- `worldfoundry_hybrid_residual/results/rank_shaped_regular_support_f81_l14s9_exp010_v1/support_summary.csv`
- `worldfoundry_hybrid_residual/figures/rank_shaped_regular_support_exp010_20260812/rank_shaped_regular_support_exp010_diagnostics.png`
