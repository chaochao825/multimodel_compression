# PLAN-043: Protected attention portfolio decision hold

- Status: completed
- Owner: researcher
- Gate: none; this plan authorizes no execution
- Claims: none
- Candidate: none
- Lane: portfolio
- Resource cap: zero GPU-hours, zero fresh captures, zero rollout, zero model
  or environment changes, and zero tuning on exposed identities

## Decision required

Select exactly one next portfolio branch before opening another experiment
Gate:

1. Reproduce an official trained sparse-attention configuration prospectively.
2. Evaluate an exact fused low-precision dense path under a separately frozen
   quality and H200 cost protocol.
3. Stop the local strict-quality attention acceleration line and preserve the
   completed evidence as a bounded negative result.

This choice can alter portfolio priority and therefore remains a protected
researcher decision. PLAN-043 does not select a branch by recency, available
compute, implementation volume, or agent preference.

## Accepted decision evidence

- DIAG-037: public fused Sol-Attn has a speed witness, but no fixed tau passed
  the complete local quality and `1.5x` speed boundary.
- DIAG-038: composing separate public Sol and FA3 calls missed `1.5x`.
- DIAG-039: pinned Sparse-VideoGen SAP/EAR is operator-ready at local F81 shape.
- DIAG-040: fixed stateful SAP/EAR missed both strict local quality and complete
  preprocessing cost Gates; EAR reduced SAP error but was slower than dense.
- Earlier custom train-free BCM/BCCB, static low-rank, temporal transport,
  support, scale, and precision-island lines remain bounded by their accepted
  records and may not be silently reopened.

## Prohibited work

- Do not run GPU experiments, capture data, start rollout, download a model,
  rebuild an environment, or modify a public candidate under this plan.
- Do not tune tau, centroids, top-p, density, block ratios, BCM, low-rank
  residuals, or fallback using globally exposed identities.
- Do not create a candidate, protocol, registry row, or execution budget until
  the researcher selects a branch and a separately frozen Gate is accepted.

## Exit mapping

- Branch 1 selected: complete PLAN-043, then draft a prospective trained-sparse
  reproduction Gate without running it.
- Branch 2 selected: complete PLAN-043, then draft an exact low-precision dense
  quality/cost Gate without running it.
- Branch 3 selected: complete PLAN-043 and record the portfolio stop decision.
- No selection: keep PLAN-043 active with zero execution authorization.
- Any experiment run under PLAN-043: invalid and non-evidentiary.

## Closure

On 2026-08-12 the researcher made a protected portfolio decision outside the
three attention-only branches: stop local attention component growth as the
mainline and replace it with motion-conjugated finite-horizon flow curvature.
RDR-0021 records the accepted decision. No experiment was run and no identity
was exposed under PLAN-043.
