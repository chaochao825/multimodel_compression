# LLaVA-OneVision Reader-Quotient Capacity Replication

Date: 2026-08-25
Status: frozen before scientific execution

## Decision question

Does the equal-byte native-reader support advantage observed on LLaVA-v1.5
replicate on a stronger and architecturally different video reader, or was it a
model/task-specific diagnostic?

## Frozen model and data

- Model: local `LlavaOnevisionForConditionalGeneration`, Qwen2-7B backbone,
  frozen BF16 weights.
- Calibration tasks: `object_existence`, `state_change`, `scene_transition`,
  `action_sequence`, and `moving_direction`, four samples per task.
- Evaluation tasks: `fine_grained_pose`, `object_interaction`,
  `action_prediction`, `egocentric_navigation`, and `moving_attribute`, four
  samples per task.
- Calibration and evaluation tasks are disjoint. The five evaluation tasks did
  not participate in the preceding LLaVA-v1.5 writer, confidence, Fisher-oracle,
  or static-prior studies.
- Each sample uses 32 uniform source-frame locations, the latest 16 as the
  feature pool, and the latest eight pool frames as exact-recent evidence.

## Frozen codec and variants

- Fit one global PCA-r384 codec on the 20 calibration feature pools only.
- Native state is the projected, spatially pooled OneVision video feature with
  shape `[16, 196, 3584]`; the generated newline token is not stored.
- Each s4 variant stores four FP16 exact residual vectors and four int16 indices
  per pool frame. All s4 variants therefore have identical per-sample bytes.
- Compare `pca_only`, `euclidean_s4`, `fisher_s4`, and fixed equal-weight
  `mixed_s4`.
- Fisher is the diagonal candidate-distribution Fisher with respect to the
  selected native video features. It is a transductive capacity oracle and is
  not a deployable scorer.
- No evaluation sample may alter PCA, rank, residual count, candidate token
  definition, prompt, support mixing weight, or decision threshold.

## Correctness guards

- The standard pixel forward and manual injection of the identical projected
  video features must have first-token logit max-absolute error at most `1e-3`.
- Gradient and no-gradient native-feature forwards must have first-token logit
  max-absolute error at most `1e-3`.
- Euclidean support must exactly match the existing feature codec.
- Candidate labels must each tokenize to one token.
- Report state bytes and dense-state compression ratio from serialized tensor
  dtypes, not parameter counts or FLOP proxies.

## Endpoints and decision

Primary endpoint: aggregate candidate KL reduction relative to `euclidean_s4`.

Secondary endpoints: full-vocabulary first-token KL, candidate top-1 agreement,
P95 candidate KL, per-task reduction, feature relative L2, support overlap,
native answer accuracy, and gradient/inference time.

- `GO`: Fisher or fixed Mixed achieves at least `25%` aggregate candidate-KL
  reduction, P95 ratio at most `1.0`, and positive reduction on at least four of
  five tasks.
- `BOUNDARY`: aggregate reduction is positive but a GO guard fails.
- `NULL`: aggregate reduction is below `5%` without material tail harm.
- `ADVERSE`: aggregate KL or P95 materially worsens.
- `INVALID`: any semantic-equivalence, byte-equality, sample-integrity, or run
  completion guard fails.

Only a GO authorizes a separately frozen low-cost content-conditioned support
scorer experiment. This probe cannot establish deployment speed, end-to-end
accuracy preservation, or superiority to official token-pruning systems.
