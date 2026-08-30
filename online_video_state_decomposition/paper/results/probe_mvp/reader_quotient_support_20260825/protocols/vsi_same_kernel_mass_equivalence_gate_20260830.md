# VSI same-kernel token-mass equivalence Gate

Date: 2026-08-30
Status: frozen before execution

## Decision question

Can token-mass bias be inserted into the OneVision reader without changing the
attention implementation used by the equal-mass baseline?

The previous proportional-attention attempt compared the model's ordinary 2D
SDPA mask path with an explicit 4D additive-mask path. That changed kernel
dispatch and candidate logits even when every mass was one, so it was invalid.
This Gate isolates mathematical equivalence from production-kernel speed.

## Data boundary

- Use only already exposed calibration positions 73--96 from the frozen VSI
  split, all 24 samples exactly once.
- Positions 97--120, selection, and formal endpoints remain unread.
- No labels, gradients, fitted parameters, or threshold search select a method.

## Frozen implementation

- OneVision checkpoint, eight frames, rank 456 quotient, 196 tokens per frame,
  and four-token groups are unchanged.
- Switch every Qwen2 language attention layer to the eager implementation before
  any evaluated forward. This is an equivalence harness, not a speed candidate.
- Evaluate deterministic support counts `0`, `196`, and `392`. For a count `k`,
  select group indices `floor(i * 392 / k)` for `i=0,...,k-1`; `k=0` is empty.
- Preserve each quotient token's registered original position.
- Construct the explicit causal mask with the installed Transformers
  `create_causal_mask` using the same model config, embeddings, 2D padding mask,
  cache positions, and position IDs as the ordinary eager path.
- Equal mass uses the explicit causal mask unchanged.
- Group mass adds `log(m_j)` as a key bias, where an unrefined quotient token has
  mass four and every exact or non-video token has mass one.

## Guards and outcome

For every sample and support count:

- ordinary eager and explicit equal-mass logits must agree over the full
  vocabulary within `1e-5` maximum absolute error;
- the empty-support explicit path is repeated and must agree within `1e-5`;
- the fully refined positioned path must agree with the dense feature path over
  the full vocabulary within `1e-5`.

`SAME_KERNEL_MASS_VALID` if every guard passes. Otherwise classify
`INVALID_KERNEL_EQUIVALENCE`, preserve the failing evidence, and do not interpret
mass-weighted quality metrics.

Mass-weighted candidate KL and dense-decision agreement may be recorded only as
diagnostics. A valid M0 authorizes a separately frozen current-support marginal
ceiling; it does not establish task quality, deployment, latency, or novelty.

## Cost and stop rule

- One isolated A800 on server 210, at most two GPU-hours.
- One implementation repair is allowed for an engineering failure.
- Stop after one valid classification or after the repair allowance is exhausted.
