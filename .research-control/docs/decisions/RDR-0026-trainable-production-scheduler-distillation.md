# RDR-0026: Open trainable production-scheduler distillation

- Status: accepted
- Date: 2026-08-13
- Decider: researcher through the explicit instruction to abandon the strict
  training-free constraint and analyze and test distillation, fine-tuning, and
  training against the accumulated results
- Supersedes: RDR-0021 only as the current mainline selection; C-021 and
  EXP-042 remain preserved as an untested claim and a teacher engineering
  failure

## Context

EXP-042 could not define a converged continuous-ODE teacher for the pretrained
BF16 Wan denoiser. Euler and midpoint RK2 remained tens of percent apart at the
registered resolutions. This prevented every scientific arm from running, but
it did not test whether a trainable finite-jump student can learn the production
sampler's behavior.

The broader project also established two constraints. First, training does not
repair an incorrect operator class automatically: token-Butterfly Q/K LoRA at
rank 4 and 8 reduced a very large local error but remained approximately 48%
on held-out output. Second, target-exposed temporal representations can be
accurate: EXP-041 reached 0.486% aggregate and 0.874% worst error in its exact
staged boundary. A training Gate should therefore preserve Wan's dense operator
semantics and learn a finite-jump correction, rather than force attention into
BCM, BCCB, or Butterfly structure.

Related work already demonstrates that generic few-step video distillation is
viable. rCM, Causal-rCM, T2V-Turbo, VideoLCM, AnimateDiff-Lightning, and
TurboDiffusion occupy the broad consistency, distribution-matching, reward, and
few-step claims. Motion Consistency Model and other motion-aware losses also
preclude a broad novelty claim for motion supervision alone. The remaining
project-specific uncertainty is narrower: whether current-observable motion
coordinates improve the data and parameter efficiency of a frozen-backbone
production-trajectory finite-jump adapter under an equal-budget control.

## Options

1. Continue repairing a continuous ODE teacher before testing C-021.
2. Park the project and use only released rCM/TurboWan checkpoints.
3. Train a full generic few-step student without a structure-specific control.
4. Replace the teacher definition with exact anchor states from Wan's production
   20-step UniPC path and first test a frozen-backbone, equal-budget four-jump
   residual adapter with and without current-observable motion alignment.

## Decision

Select option 4. Register C-022 and L-022 as the sole mainline. L-021 is parked,
not refuted. Its continuous-flow claim may be revived only by a separately
validated float-time teacher.

The new teacher object is the production trajectory itself, not a claimed ODE
solution. For anchors `a=(0,5,10,15,20)`, save the exact UniPC states and guided
Wan predictions at the four interval starts. For interval `j`, define the
finite-jump residual relative to the same first-order Wan update:

    kappa*_j = ((z_a[j] - z_a[j+1]) / delta_j - v_a[j]) / delta_j

where `delta_j` is the corresponding shifted-flow sigma difference. The teacher
endpoint can depend on UniPC history; the student is explicitly distilling that
black-box finite jump into a Markov correction.

EXP-043 is development-only. It trains on `dev00_cloth`, screens once on
`dev01_turntable`, and never opens calibration, validation, or test. It compares
zero correction, identical unaligned lifting, current-observable motion-aligned
lifting, target-exposed motion capacity, and an exactly parameter-matched local
control. Checkpoint selection is fixed at the final step and cannot use the
screen identity.

## Consequences

- The strict no-training constraint is removed. Distillation, LoRA, and
  lightweight adapter training are admissible only under prospective equal-data,
  equal-teacher-call, equal-parameter, and equal-step controls.
- Production UniPC anchor states replace the failed continuous ODE integration
  only for C-022. They do not retroactively repair EXP-042 or prove C-021.
- Fixed BCM/BCCB/Butterfly attention remains closed. Such structure can return
  only after an equal-budget trainable capacity result, never as an extra arm
  appended after a miss.
- EXP-043 is a feasibility screen, not a quality, speed, or publication result.
  A pass authorizes a fresh split-frozen training and rollout Gate; a miss parks
  the tiny motion-conjugated adapter and makes released rCM/TurboWan the default
  few-step base.
- Claims of outperforming prior work require direct same-hardware comparison to
  a released 4-step rCM/TurboWan baseline, multi-prompt and multi-seed VBench or
  equivalent video metrics, and H200 wall-clock. Published headline numbers are
  context, not a substitute for that experiment.
