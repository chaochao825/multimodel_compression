# RESULT-EXP-005: CMAQ block-control-variate probe

## Status

- Experiment: `EXP-005`
- Claim: `C-005`
- Candidate: `L-005`
- Gate: `G-005`
- Terminal outcome: `adverse`
- Secondary capacity finding: target-leaking oracle also fails the registered
  50% diagnostic ceiling.

## Validity

- Used exactly the four registered Wan F81 Q/K/V captures, frozen
  calibration/held-out identities, three query tiles, densities, and 24 seeds.
- Five synthetic estimator tests passed remotely with PyTorch 2.10.0+cu128.
- The sole registered pre-outcome repair relaxed only the dense-parity guard
  from `2e-6` to `5e-5` after a `1.656679e-5` reduction-order difference; no
  scientific metric had been written.
- Valid execution used an idle RTX 4090 and completed in 5.203 seconds. It made
  no H200 timing, rollout, or final-video claim.

## Primary evidence

At 37.5% registered block work:

| Method | P95 aggregate | P95 worst record | Certified rows at 2% |
|---|---:|---:|---:|
| Oracle top-k | 5.496% | 7.075% | N/A |
| Oracle PPS-CV | 4.557% | 6.394% | 7.123% |
| Moment PPS-CV | 8.026% | 12.020% | 1.570% |
| Frozen PPS-CV | 10.424% | 17.219% | 1.835% |

At the 50% diagnostic ceiling, oracle top-k remained at 3.647% aggregate and
4.705% worst-record error; oracle PPS-CV remained at 3.764%/5.304%. Thus the
target-leaking function class does not approach the registered 0.5%/1% gate.

All realized denominators were positive and finite. The 99% joint confidence
radius had roughly 98%--100% empirical coverage for sampled methods, but was
too wide to certify useful coverage. The adverse classification is triggered
by catastrophic held-out tail error in deployable sampled estimators, not NaN
or a negative realized partition estimate.

## Structural diagnosis

- Child-centroid correction effective support averaged 5.706%.
- Its top 1/5/10/25% blocks captured 49.0%/76.1%/87.3%/96.5% of aggregate joint
  correction energy.
- Calibration-to-held-out proposal cosine averaged 0.878; top-12.5% overlap
  averaged 0.695.
- Despite these favorable aggregate diagnostics, query-row-specific signed
  corrections and denominator uncertainty prevented concentration.

The proposal is therefore not the dominant failure: even the current-content
oracle fails. RMT/participation statistics remain useful no-go diagnostics but
do not establish a sample-efficient residual bulk.

The registered oracle reads current joint correction norms and is favorable
for the linear Hansen--Hurwitz variance objective. It is not a combinatorial
per-query support oracle, so this evidence refutes the registered tile-level
estimator rather than proving all row-aware quadrature impossible.

## Decision effect

Refute `C-005`, stop `L-005`, and close `G-005`. Do not implement a proposal
network, controller, rollout, or H200 kernel for this tile-level estimator.
Per-query leverage or a different analytic tail is a new estimator family and
requires a fresh researcher decision and protocol.

## Evidence artifacts

- Scientific report:
  `worldfoundry_hybrid_residual/results/WAN_CMAQ_BLOCK_QUADRATURE_EXP005_20260811.zh-CN.md`
- Manifest SHA256: `c6c0ead5f2a6d8ab463de3abf7933aa79d0e3d774fa40f2745790cae6e974314`
- Summary SHA256: `2484b7bc2d6d421ab4c7ffa230b5b2da99e1bd87fb2ccbfb95600f6b07703777`
- Record metrics SHA256: `bf7cc18240e5075276ca9b1f2e6f4e435cc251d34c13536aa2dea3b281040cd1`
- Structure metrics SHA256: `af05917e33f5780fb37251ce30773f44aa61eaa574b0cf3cc5101fbc51691aa6`
