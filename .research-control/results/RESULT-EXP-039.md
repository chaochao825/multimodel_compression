# RESULT-EXP-039: Anchor LOAO risk-observability screen

- Status: complete
- Validity: valid exploratory observability null
- Date: 2026-08-12
- Gate: G-018
- Claim: C-018
- Candidate: L-018

## Registered outcome

`observability-null`. The identity-held-out screen reached 0.9804 pooled
Spearman, 88.89% top-quartile high-risk recall, and zero false-safe records,
but certified only 18 of 108 records (16.67%), below the registered 25%
coverage gate.

## Supports

- Observed-anchor LOAO error strongly orders pooled candidate-stage-head risk.
- The registered cross-fit maximum-ratio certificate was conservative on the
  exposed screen and produced no false-safe records.
- The refinement-stage delta retained a moderate 0.6312 Spearman signal.

## Does not support

- Same-step anchor self-error does not provide enough certified coverage to
  continue to fresh-data aggregate-budget certification under the protocol.
- Post-hoc residualization shows that the raw signal mostly recovers static
  head/action role: static calibration risk reached 0.9543 Spearman and 92.59%
  recall, whereas content-delta Spearman/recall fell to 0.4123/37.04%.
- Post-hoc quantile multipliers cannot alter the registered outcome or support
  a safety, generalization, predictor, rollout, kernel, or speed claim.

## Decision

Refute `C-018` and park `L-018`. Do not tune same-step LOAO features,
calibrators, ranks, anchors, or thresholds on this cell. A continuation must
change the observable function class and open a new Gate.

## Integrity

- The formal run used physical H200 GPU3 after three idle confirmations, with
  exclusive flock and no foreign PID overlap; probe runtime was 24.163 seconds.
- Six pure local tests and forty-eight related remote tests passed.
- Registered hashes matched and local/remote formal artifact hashes agree.
- `s03` remained prohibited and was not loaded.

## Evidence

- `worldfoundry_hybrid_residual/results/anchor_loao_risk_observability_f81_l14s9_exp039_v1/decision.json`
- `worldfoundry_hybrid_residual/results/WAN_ANCHOR_LOAO_RISK_OBSERVABILITY_EXP039_20260812.zh-CN.md`
- `worldfoundry_hybrid_residual/figures/anchor_loao_risk_exp039_20260812/anchor_loao_risk_exp039.png`
- `worldfoundry_hybrid_residual/figures/anchor_loao_signal_decomposition_exp039_20260812/anchor_loao_signal_decomposition_exp039.png`
