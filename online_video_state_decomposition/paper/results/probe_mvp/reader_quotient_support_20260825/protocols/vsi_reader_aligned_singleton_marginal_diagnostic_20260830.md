# VSI reader-aligned singleton marginal diagnostic

Date: 2026-08-30
Status: frozen before execution

## Decision question

Does replacing the dense-gradient support teacher with an actual compact-reader
finite-difference teacher produce a useful, nested refinement path for the
position-preserving quotient representation?

This is an exploratory mechanism diagnostic. It does not change the Wan
`L-026` mainline and does not authorize a deployment, latency, or benchmark
claim.

## Why this diagnostic is needed

The previous target-gradient teacher linearized a fixed-length replacement in
which every four-token group remained four slots. The deployed compact reader
instead replaces an unrefined four-token group with one quotient token. That
operation changes sequence length, token multiplicity, RoPE positions, and the
softmax normalization state. The resulting group utility is therefore
support-dependent and need not equal a dense-path gradient projection.

The target-gradient ordering also changed across two otherwise equivalent
BF16/SDPA runs once nonzero support was selected. It is not sufficiently stable
to remain the main capacity teacher.

## Data and leakage guard

- Use only already exposed calibration positions 73--96 from the frozen VSI
  split.
- Use all 24 positions exactly once.
- Positions 97--120, selection, and formal endpoints remain unread.
- No parameter fitting, threshold search, or method selection uses an untouched
  endpoint.

## Frozen representation and evaluator

- OneVision reader and feature checkpoint are unchanged.
- Eight frames, 196 visual tokens per frame, group size four, and rank 456 are
  unchanged.
- Every unrefined group is represented by one quotient mean at registered
  within-group offset one.
- Every refined group restores its four exact tokens and their original
  positions.
- Candidate-token KL is `KL(p_dense || p_compact)` over the registered answer
  candidates.
- Dense and fully refined positioned paths must match candidate logits within
  `1e-5`.
- The base compact path and singleton group zero are each repeated once; their
  candidate logits must match within `1e-5`.

## Teacher and nested path

For the empty exact support `Omega_0`, evaluate every group `g` once:

```text
benefit_g = KL(p_dense || p_Omega0)
          - KL(p_dense || p_{Omega0 union {g}}).
```

Sort by decreasing `benefit_g`, breaking ties by increasing group index. This
produces one static, reader-aligned nested path per sample. The teacher may use
the reader output and is a capacity diagnostic, not a deployable router.

Evaluate exact group counts:

```text
0, 49, 98, 128, 160, 196, 245, 294, 343, 392.
```

The corresponding reader token retention is `(392 + 3k) / 1568`.

## Frozen outcome mapping

`STRICT_STATIC_READER_PATH` if some `k <= 196` has:

- 24/24 dense-decision agreement;
- zero harmful flips;
- mean candidate KL at most `0.01`;
- P95 candidate KL at most `0.02`;
- no match-to-mismatch transition and no increase in aggregate mean KL larger
  than `1e-6` along registered budgets from zero through `k`.

`READER_PATH_BOUNDARY` if strict does not pass, but some `k <= 245` has:

- 24/24 dense-decision agreement;
- zero harmful flips;
- mean candidate KL at most `0.02`;
- P95 candidate KL at most `0.05`;
- no match-to-mismatch transition through `k`.

Otherwise the outcome is `NO_STATIC_READER_PATH`.

## Interpretation boundary

- A pass supports the existence of a reader-aligned static refinement teacher
  on exposed data; it does not make the teacher deployable.
- A boundary supports useful but distribution-inexact compaction.
- A null means singleton utility is not additive enough to define the required
  path. It motivates at most a current-support-conditioned marginal teacher or
  path-consistency adaptation; it does not refute video redundancy.
- Any equivalence or repeatability guard failure is invalid and stops the run.

## Cost and stop rule

- One isolated GPU on server 210.
- At most 24 x 392 singleton reader evaluations plus the ten registered path
  evaluations per sample.
- One implementation repair is allowed for an engineering failure.
- Stop after one valid outcome or after the repair allowance is exhausted.
