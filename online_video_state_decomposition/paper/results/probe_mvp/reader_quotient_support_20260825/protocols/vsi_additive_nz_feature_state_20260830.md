# VSI additive N/Z feature-state development Gate

Date frozen: 2026-08-30

Role: calibration development only

## Decision question

Can a query-independent, additive visual-memory state learn the current
OneVision attention reader better than frozen centroids, moments, and prototype
mixtures without reading exact leaves?

For a positive learned feature map, each visual memory state is

\[
S=\sum_j \phi_k(k_j)v_j^T,
\qquad
z=\sum_j \phi_k(k_j),
\]

and a future query reads

\[
\hat y(q)=\frac{\phi_q(q)^T S}{\phi_q(q)^T z}.
\]

The state is exactly additive across disjoint nodes. This Gate tests that state
family only. It does not yet test sparse exact support, risk routing,
successive-refinement monotonicity, full-reader accuracy, or latency.

## Frozen identities

- Model, frame policy, split protocol, and selected layers are unchanged.
- Capture uses calibration/development positions 1--96 only.
- Training uses positions 1--72.
- Model selection uses exposed positions 73--96.
- Positions 97--120 are a one-shot confirmation reserve and are not captured or
  read unless a later Gate explicitly authorizes confirmation.
- Official selection and formal roles remain unread.
- Layers: Qwen2 0, 13, and 27.
- Feature width: 32.
- Positive map: per-head learned Q/K projections followed by softplus.
- Reader and original QKV remain frozen; only the feature map and one positive
  visual-mass scale per head are trained.
- Optimizer: AdamW, 1,000 steps, batch size 2, seed 20260830.

## Baseline and endpoints

The baseline is the same randomly initialized positive feature map before
optimization. Primary endpoints are visual and full attention-output
relative-L2 over 72 development sample-layer cells. Visual-mass log error is an
auxiliary training term because the visual state must share normalization with
the exact nonvisual context.

The analytic state ratio compares BF16 dense visual K/V with BF16 `(S,z)`.
Writer feature-projection cost, initial prefill, reader task behavior, kernel
fusion, and wall-clock are explicitly outside this Gate.

## Decision rule

`ADDITIVE_NZ_DEV_GO` requires all:

- 72 development sample-layer cells;
- learned visual mean/P95/worst at most 1%/2%/5%;
- learned full mean/P95 at most 0.5%/1%;
- visual mean at least 50% lower than the untrained baseline;
- analytic state ratio at least 32x.

`ADDITIVE_NZ_CAPACITY_SIGNAL` applies when the strict Gate fails but learned
visual mean/P95 are at most 5%/10% and mean improves at least 50%. It permits one
separately frozen width/loss diagnostic, not confirmation.

Otherwise the result is `NO_ADDITIVE_NZ_FEATURE_STATE`. Only a strict development
GO may authorize the untouched positions 97--120 confirmation. No outcome here
authorizes official selection, formal, or a speed claim.
