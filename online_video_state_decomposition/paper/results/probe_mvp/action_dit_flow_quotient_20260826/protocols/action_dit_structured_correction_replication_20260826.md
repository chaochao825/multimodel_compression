# Action-DiT structured correction fixed replication

Date: 2026-08-26

## Decision question

Does the `BOUNDARY` result from the preregistered PushT seed-0 experiment
transfer across independently trained action-DiT checkpoints without changing
the correction family or any hyperparameter?

## Frozen replication

- Checkpoints: PushT Diffusion Transformer train seeds 1 and 2, using their
  existing reported-best checkpoints.
- Reuse the exact seed-0 runner and configuration: per-output-channel W4 on the
  condition MLP and decoder FFNs, 128 calibration samples, 64 validation
  samples, 100 requested DDPM steps, 10 step buckets, ridge alpha `1e-3`,
  temporal radius 2, reduced rank 4, and experiment seed `20260826`.
- Calibration remains restricted to the training-episode split; evaluation
  remains restricted to the validation-episode split.
- No method, rank, radius, bucket count, quantizer, checkpoint, or endpoint may
  be selected from the seed-1/2 results.

## Interpretation

- A replicated mechanism signal requires the same deployable method family to
  improve both teacher-forced and complete-sampling error on both checkpoints.
- A robust `GO` additionally requires the original registered guards on each
  checkpoint, including at least 25% complete-sampling improvement and no P95
  degradation.
- Mixed signs, a different winner on each seed, or persistent P95 degradation
  close the post-hoc structured-correction path as a general action-DiT method.
- This replication does not authorize an environment-success, integer-kernel,
  OpenPI, or VLA claim.
