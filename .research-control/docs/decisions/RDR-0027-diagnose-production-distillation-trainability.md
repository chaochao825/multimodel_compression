# RDR-0027: Diagnose production-distillation trainability after EXP-043

- Status: accepted
- Date: 2026-08-13
- Decider: researcher through the explicit request to analyze whether opening
  distillation, fine-tuning, and training can release the accumulated method's
  potential and to test that conclusion against prior results
- Supersedes: RDR-0026 only as the active mainline selection; EXP-043 and its
  null result remain immutable

## Context

EXP-043 validly returned `capacity-null`, and every trained arm was worse than
zero correction on its held-out development identity. Post-run integrity
analysis also found that `delta_sigma^4` sampling assigned 1,976 of 2,000
updates to the final interval and only 24 total updates to the first three.
The result therefore closes the registered candidate but cannot distinguish a
bad sampler from missing interval conditioning, insufficient local capacity,
or cross-identity transfer failure.

Public few-step video generators show that training can solve the broad task,
but through substantially larger function classes and objectives: consistency
or distribution matching, adversarial/reward losses, full rollouts, synthetic
data, and often jointly trained sparse attention or quantization. The remaining
high-information question is not whether generic distillation works. It is
whether the project's tiny structured finite-jump representation has any local
trainable capacity once the known sampler defect is removed.

## Options

1. Treat EXP-043 as a general capacity refutation and stop immediately.
2. Add ranks, losses, experts, identities, and rollout to EXP-043 post hoc.
3. Run one separate development-only decomposition using balanced interval
   training, interval conditioning, stage-specific adapters, and transductive
   target-exposed controls on the already exposed identities.
4. Skip diagnosis and begin full rCM/DMD training.

## Decision

Select option 3 as EXP-044. This is a no-claim diagnostic, not a replication or
confirmation. It may reuse only `dev00_cloth` and `dev01_turntable`, both already
exposed by EXP-043. It must not open calibration, validation, or test identities.

Compare, in order, the same shared lifting under balanced sampling, a nearly
equal-parameter interval-FiLM lifting, four stage-specific liftings, and a wider
stage-specific capacity control. Evaluate both development transfer and
transductive same-identity fitting. Keep observable and privileged
target-exposed coordinates separate.

## Consequences

- C-022 remains refuted and L-022 remains parked; EXP-044 cannot repair them.
- If even the wide target-exposed transductive control misses 1% endpoint error,
  stop local finite-jump correction and use a released/full few-step student.
- If transductive capacity passes but cross-identity transfer fails, the
  bottleneck is data/observability and the next viable class is a multi-identity
  student, not a larger static matrix family.
- If interval conditioning alone produces a large transfer gain, a fresh
  prospective identity Gate may compare the corrected tiny adapter against
  equal-budget LoRA. No such claim is allowed from the exposed identities.
- BCM/BCCB/Butterfly attention paths remain closed; this diagnostic preserves
  the teacher's dense operator semantics.
