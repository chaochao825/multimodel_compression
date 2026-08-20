# RESULT-EXP-038: Progressive heterogeneous function-class transfer

- Status: complete
- Validity: valid exploratory transfer boundary
- Date: 2026-08-12
- Gate: G-017
- Claim: C-017
- Candidate: L-017

## Registered outcome

`transfer-boundary`. The validation-frozen allocation reached 0.856% aggregate,
1.926% worst-head relative L2, and 57.90% optimistic work. Frozen test retained
the same 1.727x arithmetic upper bound and passed the 2% worst-head gate at
1.713%, but aggregate error rose to 1.097%, failing the registered 1% gate.

## Supports

- Calibration prior plus progressive current observations and heterogeneous
  dense fallback can reach the strict local quality/work boundary on validation.
- Failure is aggregate-risk drift rather than one catastrophic head: several
  moderate approximate heads worsen together while no frozen approximate head
  exceeds 2% on test.
- A test-exposed diagnostic using only existing evaluated actions reaches about
  0.951% / 1.862% at 62.28% optimistic work, leaving a narrow adaptive margin.

## Does not support

- Static per-head action identity does not transfer at the registered 1% / 2%
  gate.
- The post-hoc 1.60x arithmetic point is not a prospective pass and omits all
  runtime overhead.
- Target-exposed coefficients and support remain; no deployable predictor,
  measured H200 speed, rollout, or full-model claim is supported.

## Decision

Refute `C-017` and park static head actions. The only bounded continuation is a
calibration/validation-only leave-one-anchor-out risk-observability screen that
does not read the prior test identity. A positive screen may authorize fresh
identity collection for a prospective aggregate-budget certificate; a null
parks same-step residual completion on this cell.

## Integrity

- Formal run used physical H200 GPU3 after three idle confirmations, with no
  process overlap; exit code 0 and runtime 11.9 seconds.
- Thirty-four related remote tests passed; registered hashes matched.
- Local and remote formal artifact SHA-256 values match.

## Evidence

- `worldfoundry_hybrid_residual/results/progressive_heterogeneous_transfer_f81_l14s9_exp038_v1/decision.json`
- `worldfoundry_hybrid_residual/results/WAN_PROGRESSIVE_HETEROGENEOUS_TRANSFER_EXP038_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/progressive_heterogeneous_transfer_exp038_20260812/progressive_heterogeneous_transfer_exp038.png`
