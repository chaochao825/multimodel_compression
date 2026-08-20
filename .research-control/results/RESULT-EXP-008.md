# RESULT-EXP-008: Positive-residual basis provenance

- Status: complete
- Validity: valid exploratory boundary
- Date: 2026-08-11
- Gate: G-008
- Claim: C-008
- Candidate: L-008

## Registered outcome

`adaptive-capacity-boundary`. The frozen positive rank-64 tail plus 25%
regular support had 9.395% aggregate / 19.775% worst-head test error.
Target-exposed adaptive SVD reduced this to:

- rank 8: 3.113% / 5.773%;
- rank 16: 1.862% / 3.121%;
- rank 32: 0.803% / 1.298%.

Adaptive rank 16 therefore missed the 0.5% / 1% oracle gate, while adaptive
rank 32 crossed the weaker 1% / 2% deployment-quality threshold.

Validation selected `pooled_q_weighted_v_pca` among the four QKV-only content
bases. With target-defect oracle coefficients, its test error remained:

- rank 16: 6.338% / 12.863%;
- rank 32: 5.022% / 10.218%.

Its mean overlap with the adaptive subspace was 0.468 at rank 16 and 0.555 at
rank 32. The calibration-frozen defect basis was similarly insufficient at
5.422% / 10.913% for rank 32.

## Supports

- The residual contains a strong posterior low-rank component, but rank 32 is
  required for the registered positive-bulk/regular-support target.
- Current V variance and K/V cross-moments do not expose the posterior defect
  directions. Even oracle coefficients cannot rescue their basis mismatch.
- Deterministic covariance generation is not the missing coefficient predictor;
  the generated output subspace itself is wrong.

## Does not support

- It does not refute a learned, nonseparable, value-aware basis generator.
- It does not show that rank-32 residual correction is deployable; all
  projection coefficients are target-defect oracles.
- It does not test support/basis co-training, rollout quality, or measured
  kernel speed.
- The pooled-Q weighted covariance has high arithmetic cost and is a diagnostic,
  not a recommended runtime operator.

## Decision

Park deterministic V-PCA, selected-V PCA, K-transpose-V, and pooled-Q weighted
V-covariance bases for this target. A bounded revival may test an O(rNd)
signed value-aware latent-pooling basis whose directions are selected by
learned or oracle latent queries rather than variance. Its first Gate must be a
target-exposed function-class ceiling; no coefficient generator or kernel is
justified unless that ceiling reaches 0.5% / 1% at rank at most 32.

## Evidence

- `worldfoundry_hybrid_residual/results/positive_residual_basis_provenance_f81_l14s9_exp008_v1/decision.json`
- `worldfoundry_hybrid_residual/results/positive_residual_basis_provenance_f81_l14s9_exp008_v1/manifest.json`
- `worldfoundry_hybrid_residual/results/positive_residual_basis_provenance_f81_l14s9_exp008_v1/basis_summary.csv`
- `worldfoundry_hybrid_residual/figures/positive_residual_basis_exp008_20260811/positive_residual_basis_exp008_diagnostics.png`
