# PLAN-057: Test distillation-induced low-rate state closure

- Status: complete
- Owner: researcher and Agent
- Gate: G-027
- Claims: C-027
- Candidate line: L-027
- Lane: explore
- Resource cap: one H200, at most eight GPU-hours and 30 GiB under the EXP-048 root

## Decision to unlock

Decide whether a training-native State-Closure Distillation experiment has a
measured mechanism to exploit, or whether cross-denoising hidden-state closure
should stop.

## Context and source authority

RDR-031 authorizes this side probe without superseding RDR-030. EXP-045/046 are
the post-hoc nulls; EXP-047 is the unchanged released-rCM deployment mainline.
The official rCM source commit and both checkpoint identities are inherited only
as pinned model identities, not as EXP-047 scientific outputs.

## Non-goals

- No H200 latency or energy claim under the co-resident stopped process.
- No VAE decode, VBench, video-quality comparison, training, checkpoint change,
  sparse attention, quantization, BCM/BCCB, Butterfly, or hidden-state router.
- No use of EXP-045/046 final identities or EXP-047 formal prompt-seed pairs.

## Milestones

1. Pass pure-state synthetic tests and one equivalence-checked integration smoke.
2. Capture four frozen calibration identities with sampled tokens only.
3. Freeze model-specific/shared rank bases and stagewise transition coefficients.
4. Evaluate four untouched selection identities at full token resolution.
5. Aggregate the two-by-two effects, open-loop risk, payload/cost, and close G-027.

## Validation

- Exact block outputs must be unchanged by capture.
- Calibration and selection indices are checked before model load.
- Selection reads one frozen closure artifact and performs no fitting.
- G-027 uses rank 64, model-specific bases, and the thresholds in the frozen
  EXP-048 config; shared-basis results diagnose gauge dependence.
- Every model/trajectory/layer/rank/stage/horizon row must be finite and complete.

## Stop and escalation rules

Stop on equivalence failure, split contamination, incomplete cross cells, two
integration repairs, resource exhaustion, or one valid G-027 outcome. Relative
improvement without absolute fidelity is recorded as directional evidence and
does not unlock training.

## Closure

Write RESULT-EXP-048, update claim/candidate/experiment/status records, preserve
all failed attempts, and leave EXP-047 unchanged.
