# Action-DiT structured correction probe

Date: 2026-08-26

## Decision question

Does an action-generating diffusion Transformer expose a more transferable
structured residual than the previously tested video DiT and LeWM latent
transitions, because its output has a physical action-horizon ordering?

This is the first experiment in this program on an actual action diffusion
policy. The earlier LeWM experiment predicted action-conditioned visual latent
transitions and is not an action-DiT result.

## Frozen model and data

- Model: EMA weights from the PushT low-dimensional Diffusion Transformer,
  seed 0, checkpoint `epoch=0850-test_mean_score=0.967.ckpt`.
- Architecture: horizon 10, action dimension 2, hidden width 256, 8 decoder
  layers, 100 DDPM inference steps.
- Calibration samples come only from the checkpoint's training-episode split.
- Evaluation samples come only from its validation-episode split.
- The checkpoint, dataset, normalizer, architecture, and policy weights are
  read-only.
- Initial diffusion noise and scheduler randomness are paired across methods.

## Perturbation

Use per-output-channel symmetric W4 fake quantization on the condition MLP and
all decoder FFN `linear1`/`linear2` weights. Attention projections, embeddings,
normalization, biases, and the output head remain full precision. W8 is a
numerical reference. Fake quantization validates numerical behavior only and
does not support an integer-kernel speed claim.

## Candidate family

Let the frozen full-precision denoiser output be (v), the W4 output be
(v_q), and the defect be (d=v-v_q). Every deployable correction may use
only the current noisy action chunk (x_s), current W4 output (v_q), current
observation condition (o), and diffusion-step bucket (b(s)).

The comparisons are:

1. W4 without correction.
2. Bucket mean correction.
3. Bucket/channel affine output balancing.
4. Circular temporal kernel with radius 2, the action-axis BCM comparator.
5. Non-periodic Toeplitz temporal kernel with radius 2 at identical budget.
6. Bucketed reduced-rank predictor with rank 4.
7. Toeplitz plus rank-4 residual predictor.
8. Full ridge predictor as a function-class ceiling, not a deployable winner.
9. Calibration-fixed defect PCA with held-out oracle coefficients as a basis
   transfer ceiling, never as a runnable method.

All fitted parameters use calibration trajectories only. Evaluation defects,
actions, or endpoints may not update a basis, coefficient predictor, bucket,
radius, rank, or quantization rule.

## Endpoints

Primary mechanism endpoints:

- teacher-forced denoiser-output relative L2;
- fraction of W4 defect energy removed;
- P95 per-call relative defect;
- calibration-fixed rank-1/2/4/8 basis transfer.

Primary closure endpoints use complete paired 100-step sampling:

- final action-chunk relative L2 to the full-precision policy;
- P95 per-sample action relative L2;
- executed-action endpoint error;
- first-difference and second-difference trajectory errors.

The runner also reports selected-weight fraction, analytical MAC share, state
payload, and correction MACs. It does not claim H200 or integer-kernel speed.

## Outcome mapping

A correction is a mechanism-level `GO` only if, on held-out episodes:

- complete-sampling action relative L2 improves by at least 25% over plain W4;
- P95 action error does not increase;
- endpoint and second-difference errors do not increase by more than 10%;
- analytical correction work is at most 5% of one denoiser forward;
- the same method improves teacher-forced defect rather than relying only on
  stochastic cancellation.

`BOUNDARY` means a positive teacher-forced result that fails complete sampling
or has a confidence/worst-case conflict. `NULL` means no deployable candidate
reaches 25%. `ADVERSE` means the correction worsens complete sampling by at
least 10%. A GO authorizes a fresh PushT environment rollout and a pi0
replication; it does not itself establish control success or GPU acceleration.

## Stop rule

Do not tune on validation episodes. If the registered family is NULL or
ADVERSE, stop post-hoc action-output correction and prioritize scale-calibrated
PTQ, step distillation, or train-time action-trajectory structure. Do not add a
larger BCM, more buckets, or a learned router to rescue this split.
