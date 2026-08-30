# VSI Tiny Group-Risk Controller Prospective Gate

Date frozen: 2026-08-30

Role: low-cost calibration teacher distillation with prospective calibration test

## Decision question

The target-visible reader-gradient selector nearly passed at 43.75% visual-token
retention, while residual energy, query cosine, and scalar margin did not
transfer. Can a single tiny controller recover enough of the task-sensitive
group ordering from deployable quotient/query/residual metadata to support
progressive exact retrieval?

## Frozen data roles

- Controller train: first 24 eligible calibration questions.
- Model selection and fallback calibration: next 24 eligible calibration
  questions. Their scalar-query reader outcomes were exposed previously, so
  they cannot be prospective evidence.
- Prospective test: following 24 eligible calibration questions. Their reader
  outcomes and group-risk labels are unread before this run.
- Selection and formal roles remain unread.

The rank-456 PCA basis remains the calibration-wide unsupervised basis frozen
before these reader outcomes. No OneVision weight is changed.

## Frozen representation

- Uniform 8 frames and 1568 full visual tokens.
- Each frame is partitioned into 49 contiguous groups of four tokens.
- Every group has one quotient-mean token; exactly 98 groups are refined to
  their four exact tokens, retaining 686 tokens or 43.75%.
- A memory writer may compute one residual energy and an 8-dimensional fixed
  Rademacher residual sketch from the exact group before cold storage. These
  nine values are the only innovation metadata available to the controller.
- Quotient and query use separate fixed 8-dimensional Rademacher sketches,
  their elementwise interaction, one exact quotient-query cosine, and four
  normalized temporal/sequence position values.

The resulting controller input has 39 scalars. Sketch seed is `20260830`.

## Frozen teacher and controller

Teacher risk for group \(g\) is

\[
r_g^*=\max_c
\frac{[-\langle\nabla_{X_g}m_c,\delta X_g\rangle]_+}
{\max(m_c,0.05)}.
\]

Within each sample, the 98 largest teacher risks define positive groups.

Controller:

- `Linear(39, 32) -> GELU -> Linear(32, 1)`;
- full-batch AdamW, learning rate `1e-3`, weight decay `1e-4`;
- weighted binary cross entropy with the frozen 98/392 class balance;
- at most 100 epochs;
- choose the epoch with highest validation top-98 recall, breaking ties by the
  earliest epoch;
- normalize inputs using train statistics only.

No architecture, feature, seed, optimizer, or token budget may change after the
prospective role is read.

## Runtime and fallback

The deployed selector uses only controller scores. On the validation role, set
the compressed top-1 margin threshold to the largest margin among controller
reader disagreements. Apply that threshold once to the prospective role.

Controller arithmetic must stay below one million MACs per question, excluding
precomputed memory-write sketches. This is a proxy guard, not a latency claim.

## Decision rule

The prospective result is `GO` only if all conditions hold:

- delivered full-reader agreement at least 98%;
- zero remaining harmful flips;
- exact full fallback rate at most 15%;
- effective visual-token retention at most 53%;
- delivered task accuracy no more than one percentage point below full reader;
- controller arithmetic below one million MACs;
- prospective top-98 teacher-risk recall exceeds both residual-energy and
  query-cosine selectors.

A `GO` permits a separately frozen selection/formal and measured reader-latency
protocol. A `NO_GO` closes this metadata/controller family; further work must
either expose richer current-state information or move to train-time native
memory/tokenization rather than enlarging the same controller.
