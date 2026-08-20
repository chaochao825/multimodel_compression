# RESULT-EXP-004

- Experiment: EXP-004
- Date: 2026-08-11
- Code identity: probe `1c0de874b87def6a`; core `f16e8d4170b14c7b`
- Configuration identity: `98bd3a4eee2b93cc`
- Data identities: capture index `d8bc6f7fbb0accf4`; capture manifest
  `f959929ba46773b6`; worker records `3487b193f7128b72`
- Decision identity: `b8fc9f1fbe83357e`
- Evidence tier: prospective split-frozen offline local-output screen
- Independent unit: one training, one validation, and two held-out prompt/seed
  identities
- Validity: valid within the frozen 3-layer x 3-step x 12-head conditional scope
- Outcome class: null
- Protocol deviations: none

## Registered result

The target-leaking multi-action envelope selected 107 of 216 held-out
head-cells. It reached 0.928% aggregate and 1.994% worst local output error at
1.554x fallback-adjusted arithmetic attention speed. This establishes only a
narrow function-class ceiling; it uses held-out target errors and excludes all
router and kernel overhead.

The frozen full structure scout selected 12 of 216 records and reached 0.0015%
aggregate and 0.0068% worst error, but only 1.043x arithmetic speed. The static
layer-step-head table selected 78 records at 0.327%/1.771% error and 1.324x. The
structure-only scout reached 1.097x but produced two false-safe records and
4.713% worst error. The full scout therefore missed the 1.5x gate and was
0.281x slower than the strongest simple baseline.

## Validity checks

- The worker table contained exactly 3,888 registered action rows and every
  head-cell had all nine frozen actions.
- The capture index contained exactly 36 registered conditional QKV files and
  each file matched its declared byte size.
- Sample splits were disjoint and matched the frozen config.
- Models read only the training identity; the validation identity selected one
  threshold; test identities did not alter features, actions, regularization,
  or thresholds.
- Dense attention mass, dense AV, adaptive SVD, test errors, and head-role labels
  were absent from deployment scores.
- Seven numerical/selection unit tests passed in the frozen remote PyTorch
  environment. All features and outputs were finite, and no retry occurred.

## Claim update

- `C-004`: refuted in the registered Wan/F81, worker, split, local-output, and
  arithmetic-cost scope.
- Supports: static layer-step-head risk transfers strongly and a target-leaking
  multi-action envelope barely exceeds 1.5x.
- Does not support: a train-free structure scout, H200 acceleration, paired
  endpoint fidelity, or a population-level statistical certificate.
- Unknown: whether a directly sampled action-defect sentinel can detect rare
  flips under less than 2.33% dense-attention work, and whether low-cost learned
  routing can widen the worker frontier.

## Decision

Close `G-004` as null. Do not tune the registered scout, implement a worker
kernel, or run rollout. Any direct defect sentinel, learned router, new worker,
or precision runtime requires a fresh decision and protocol.

## Post-hoc explanation

Static calibration risk transferred with mean Spearman 0.9687 and safe-set
Jaccard 0.8193. Raw QKV structure-error correlations were often high, but after
centering on calibration layer-step-head identity their content-delta
correlations were weak or inconsistent. Structure mainly recovered fixed head
role rather than sample-specific action risk.

A post-hoc robust static table reached 1.596x on calibration+validation but
failed held-out quality at 1.060% aggregate and 6.282% worst error. The best
test-safe post-hoc static threshold reached only 1.379x. These diagnostics are
explanatory and cannot alter the registered null outcome.

## Artifacts

- Registered local result:
  `worldfoundry_hybrid_residual/results/structured_risk_scout_exp004/`
- Post-hoc analysis and figure:
  `worldfoundry_hybrid_residual/results/structured_risk_scout_exp004_analysis/`
- Chinese synthesis:
  `worldfoundry_hybrid_residual/results/WAN_STRUCTURE_AS_RISK_SCOUT_EXP004_20260811.zh-CN.md`
- Remote registered result:
  `/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723/results/structured_risk_scout_exp004/`
