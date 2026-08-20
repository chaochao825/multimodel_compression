# RDR-0021: Make motion-conjugated flow curvature the sole mainline

- Status: accepted
- Date: 2026-08-12
- Decider: researcher through the explicit request to replace component growth
  with a new RDR and make the flow-map curvature probe the unique mainline
- Supersedes: RDR-0001 only as the current mainline selection; all accepted
  closures and evidence in RDR-0002 through RDR-0020 remain binding

## Context

The completed program falsified or bounded many local approximation classes:
fixed BCM/BCCB and Butterfly bases, static low-rank tails, scalar and grouped
temporal transitions, sparse-support shaping, train-free coefficient transport,
low-precision operand corrections, and released sparse-attention baselines under
the registered strict local gates. Continuing to add experts, ranks, routers, or
fallbacks would no longer test one clear scientific mechanism.

The remaining high-leverage alternative changes the approximation target. A
pretrained Wan sampler defines a finite-interval denoising flow map. Its exact
average velocity differs from the current instantaneous velocity by a
finite-horizon curvature remainder. Natural video also has a distinct physical
time axis with motion correspondences. The new hypothesis is that expressing
clean-video predictions in a current-observable motion-aligned coordinate system
may reduce the complexity of this curvature remainder.

This decision explicitly separates three axes: denoising time, physical video
time, and transformer depth. A video motion field is not the probability-flow
velocity. It is only a candidate coordinate transform under which the
finite-interval denoising operator may become simpler.

## Options

1. Continue the closed local attention portfolio with more BCM, sparse, low-rank,
   low-precision, cache, router, or fallback components.
2. Reproduce a generic MeanFlow, consistency, or flow-map distillation method
   without a structure-specific hypothesis.
3. Make motion-conjugated finite-horizon curvature the sole mainline and require
   a bounded representation/transfer probe before adapter training or rollout.
4. Stop the project after preserving the completed negative evidence.

## Evidence

Facts from this project:

- C-000 through C-013 and C-015 through C-019 are refuted under their registered
  classes. C-014 and C-020 are narrow representation witnesses, not deployment
  claims.
- Fixed structured bases failed even when capacity was expanded, while several
  target-adaptive low-rank ceilings were much better than frozen transfer. This
  supports basis rotation or coordinate mismatch, not generic low-rank success.
- Temporal AR/reuse/Taylor probes failed on difficult Wan modules, showing that
  adjacent denoising states are not enough in the old hidden coordinates.
- The released Sol/SAP/EAR cycle did not jointly pass strict quality and H200
  cost gates. Local attention changes therefore have limited information value
  relative to a direct NFE-reduction hypothesis.

Facts from related work:

- Flow Map Matching formalizes two-time maps; Shortcut Models and MeanFlow learn
  finite-step or average-velocity objects.
- Decoupled MeanFlow and TMD show that adapting a pretrained backbone into a
  flow-map model is plausible, but TMD already uses a semantic backbone plus a
  recurrent flow head on Wan.
- rCM/Causal-rCM already scale continuous-time consistency and MeanFlow-style
  distillation to bidirectional and autoregressive video.
- RhymeFlow already uses video-time keyframes and asynchronous denoising. Motion
  conditioning/alignment is also established in OnlyFlow and MoAlign.

Researcher judgment:

- The defensible gap is not "MeanFlow for video" or "motion-guided denoising."
  It is the narrower causal claim that a current-observable motion conjugacy
  exposes a compact, nonperiodic finite-horizon curvature operator in a frozen
  pretrained bidirectional video flow model.
- The structure is admissible only after the coordinate transform. BCCB may be
  tested only as a numerical circulant embedding or a periodic control, not as a
  semantic assumption about video boundaries.

## Decision

Select option 3. Register C-021 and L-021. L-021 is the only mainline. Close the
PLAN-043 attention portfolio hold and preserve all local-attention evidence as
completed negative or boundary evidence.

The canonical mathematical object is the finite-interval average velocity

    u*(z_t, r, t) = (z_t - z_r) / (t - r),

with boundary-preserving decomposition

    u* = v(z_t, t) + (t - r) kappa*(z_t, r, t).

The first probe compares the same structured family in raw-noisy,
clean-prediction, and current-observable motion-aligned coordinates. It must
separate a target-exposed capacity ceiling from calibration-frozen transfer and
must evaluate finite-map composition as well as endpoint error.

No GPU execution, training, capture, rollout, model change, environment change,
or experiment registration is authorized by this RDR. PLAN-044 is a zero-GPU
protocol and artifact-identity audit. An executable EXP-042/G-021 requires a
separate frozen protocol and acceptance.

## Consequences

- Fixed BCM/BCCB/Butterfly attention, static low-rank residuals, sparse-attention
  variants, quantization, cache, and mixed fallback systems are not active lines.
- The new line may compare nonperiodic lifting, Toeplitz, low-displacement-rank,
  and low-rank controls only at equal payload and only on curvature after the
  coordinate audit. Adding experts after a miss is prohibited.
- Target clean video, teacher endpoints, held-out residuals, and held-out optical
  flow may be used only in a separately labeled capacity oracle. They may not
  fit or select a deployable warp, basis, coefficient, interval, or fallback.
- If motion alignment does not improve the identical frozen structure by the
  registered margin, or if the capacity/transfer/cost gate fails, park L-021.
  A generic flow-map baseline succeeding would not rescue the structured claim;
  changing the mainline again would require another protected decision.
- If the representation and transfer gate passes, the next decision may open a
  frozen-backbone tiny-adapter training Gate, followed only then by multi-prompt,
  multi-seed rollout and H200 timing.
