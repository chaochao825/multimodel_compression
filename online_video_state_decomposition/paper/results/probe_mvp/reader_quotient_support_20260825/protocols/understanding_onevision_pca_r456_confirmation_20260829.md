# LLaVA-OneVision PCA-r456 Untouched-Task Confirmation

Date: 2026-08-29
Status: frozen before v2 scientific execution; one pre-analysis schema repair

## Decision question

Does the selection-positive, support-free `PCA-r456+s0` codec preserve the
frozen LLaVA-OneVision reader on tasks that were untouched by calibration,
Reader-Quotient replication, and rank/support allocation?

This is a new bulk-allocation confirmation requested after the diagonal-Fisher
line closed. It is not a retroactive `GO` for the preceding rank/support
protocol and does not reopen Fisher support, BCM/BCCB, rotation, or scorer
training.

## Frozen candidate and evidence

- Model, processor, prompts, 32-frame sampling, 16-frame feature pool, and
  eight-frame reader budget remain identical to the 2026-08-25 OneVision run.
- Reuse the calibration-only rank-456 codec at
  `onevision_rank_support_allocation_20_20260825_v1/codec/onevision_feature_pca_rank456.pt`.
  Refitting, basis rotation, support selection, quantizer tuning, and model
  mutation are forbidden.
- Evaluate only full native features and `PCA-r456+s0`.
- Use the five tasks named but not consumed by the preceding protocol:
  `fine_grained_action`, `action_antonym`, `unexpected_action`,
  `counterfactual_inference`, and `action_count`.
- Select 100 samples per task with seed `20260829`, for 500 examples total.
  Task identity is disjoint from every prior OneVision selection task.
- Before seeded selection, exclude records with fewer than two or more than 26
  answer candidates, or whose registered answer is absent from the candidate
  list. This task-format eligibility rule is applied identically to all five
  tasks before any model output is read.

The initial `v1` attempt selected directly from raw records and stopped after
398 checkpoints when two `counterfactual_inference` records contained only one
candidate. No scientific metric was aggregated or inspected. The incomplete
attempt is retained; `v2` restarts from a fresh output directory after applying
the deterministic eligibility rule above.

## Primary endpoints and decision rule

The candidate is `PASS` only if every guard holds:

- all 500 expected samples complete exactly once with finite metrics;
- full-reader candidate accuracy is at least `35%`;
- compressed accuracy is no more than `2` percentage points below full;
- the one-sided 95% Clopper-Pearson upper bound on
  full-correct/compressed-wrong events is at most `2%`;
- candidate prediction agreement is at least `98%`;
- no task loses more than `5` percentage points;
- tensor payload is at most `2,867,328` bytes and compression is at least
  `7.8x`;
- manual feature injection differs from the direct pixel path by at most
  `1e-3` in first-token logits.

Candidate/vocabulary KL, feature relative L2, paired bootstrap accuracy delta,
per-task flips, and stage timings are descriptive secondary endpoints. They
cannot rescue a failed primary guard.

The result is `ADVERSE` if aggregate accuracy loses more than `5` points,
prediction disagreement exceeds `5%`, the empirical harmful-event rate
exceeds `5%`, or any task loses more than `10` points. A valid result that is
neither `PASS` nor `ADVERSE` is `BOUNDARY`. Missing samples, duplicate samples,
non-finite metrics, identity mismatch, or model-injection mismatch is
`INVALID`.

## Resource and consequence

- No training or VAE generation; at most three idle A800 GPUs and two aggregate
  GPU-hours.
- `PASS` authorizes a separate serialization/reconstruction/prefill/TTFT
  profile of this exact codec. It does not establish latency or system speedup.
- `BOUNDARY` or `ADVERSE` parks the strong-reader PCA confirmation and retains
  only the earlier LLaVA-v1.5 bounded-state preservation result.
