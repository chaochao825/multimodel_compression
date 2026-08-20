# RDR-0018: Authorize anchor leave-one-out risk-observability screen

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to continue theory-driven,
  high-information experiments without abandoning negative-result insight
- Supersedes: none

## Context

EXP-038 passed validation at 0.856% / 1.926% and 57.90% optimistic work but
missed frozen-test aggregate quality at 1.097%. Several moderate-risk heads
worsened together while worst-head error remained below 2%. Static head identity
therefore cannot certify the aggregate residual budget.

The current operator already computes exact residuals on 64/96/112 selected
micro-anchor rows. A leave-one-anchor-out (LOAO) reconstruction error can be
computed entirely from these observed rows: build the prior-plus-innovation
basis from all but one selected block, then predict the held-out observed block.
If this self-error tracks unseen-row basis error, it may support monotone
`b4 -> b6 -> b7 -> dense` refinement without reading unseen rows.

## Decision

Authorize one calibration/validation-only mechanism screen. Use `s00` with a
prior from `s01`, `s01` with a prior from `s00`, and freeze all calibrators before
evaluating `s02` with a prior from both calibration identities. The prior EXP-038
test identity `s03` is prohibited.

The primary signal is anchor-reference-normalized LOAO error. Secondary
diagnostics may report prior residual, innovation conditioning, basis drift, and
stage-delta fidelity, but they cannot replace the registered primary gates.

## Consequences

- A positive screen only authorizes fresh identity collection and a prospective
  aggregate-budget certificate Gate.
- A null parks same-step anchor self-certification on this cell; do not rescue it
  with more features, learned regressors, ranks, BCM/BCCB, or post-hoc thresholds.
- Target-exposed non-anchor projection remains a label only. No coefficient
  predictor, rollout, kernel, or speed claim is authorized.
- Existing exposure to `s02` makes this a mechanism screen, not independent
  confirmation or population generalization evidence.

## Pre-run implementation lock

- Config: `0f29814793d70b95`
- Probe: `7c044cb19f575022`
- Core: `efce1004408d0c36`
- Pure test: `e9414258b7dcf7e9`
- Tensor-isolation test: `c40236e96b6a7ac4`
- Six pure local tests pass. Forty-eight new and inherited tests pass in the
  registered remote PyTorch environment before formal execution.

## Closure

Closed as `observability-null`. The formal screen reached 0.9804 pooled
Spearman, 88.89% top-quartile recall, and zero false-safe records, but only
16.67% certified coverage versus the registered 25% gate. Post-hoc
residualization further localized most signal to static head/action role rather
than content drift. `C-018` is refuted and `L-018` is parked.
