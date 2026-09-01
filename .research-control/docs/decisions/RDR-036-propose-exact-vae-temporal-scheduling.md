# RDR-036: Accept exact Wan VAE temporal scheduling above resident rCM

- Status: accepted
- Date: 2026-09-01
- Decider: researcher through the explicit continuing objective to retain the
  quality-passing rCM NFE baseline and then execute exact system optimization
  before FP8 dense and fused attention
- Supersedes: none

## Context

EXP-052 established `9.637995s` as the mandatory exact resident rCM4 F81
baseline. VAE decode is now the largest component at `4.308300s` or 44.7% of
the request; denoising is `3.205365s`, serialization `1.796082s`, and text only
`0.064420s`.

The measured candidate ceilings are no longer comparable to the pre-rCM
profile. Eliminating VAE time has a `1.808x` endpoint ceiling and a `2x` VAE
would yield about `1.288x`. By contrast, the historical `1.51x` self-attention
kernel, even under optimistic full self-attention coverage, yields only about
`1.064x` incremental endpoint speedup and still lacks model-wide exactness or
quality evidence. Serialization has a `1.229x` elimination ceiling but changes
latency/throughput semantics if hidden by asynchronous pipelining.

The official Wan VAE decoder processes all 21 F81 latent frames one at a time
and repeatedly concatenates accumulated output. Its causal convolutions already
carry two-frame feature caches, while spatial attention treats time as a batch
axis. A bounded schedule can therefore preserve the first-frame sentinel and
group only subsequent frames, but mathematical equivalence does not guarantee
bitwise CUDA equality.

## Options

1. **Exact VAE temporal scheduling first (proposed).** Screen chunk sizes
   `1/2/4/8`, require bitwise decoded tensors, then measure complete VAE and
   resident endpoint latency.
2. Serialization first. This has a lower ceiling and risks replacing request
   latency with throughput overlap before the VAE bottleneck is understood.
3. FP8/fused attention first. This has a lower current Amdahl return and adds
   approximate-quality and kernel-integration uncertainty.
4. Stop at the `9.638s` incumbent.

## Decision

Accept option 1 as `C-031 / L-031 / EXP-053 / G-032`. The candidate may change
only temporal grouping in a separate harness; the official rCM repository and
Wan/VAE modules remain unmodified.

- frame 0 must remain one decoder call;
- later frames may use only chunk sizes `1/2/4/8`;
- every candidate must pass `torch.equal` against official decode on the same
  latent before timing can promote it;
- F17 selects the fastest exact chunk by a frozen rule, then F81 confirms it on
  all four EXP-047 prompts at seed `2026082701`;
- complete VAE speedup must be at least `1.26x` and projected resident endpoint
  improvement at least `1.10x` before a full endpoint run;
- the full resident endpoint must then be at least `1.10x` faster than
  `9.637995s` with bitwise output equality.

No compile, CUDA Graph, FP8, quantization, attention change, serialization
overlap, scheduler change, or tolerance relaxation is allowed inside this Gate.

## Consequences

- L-030 remains the immutable incumbent; L-031 is a bounded integration
  candidate, not a replacement until G-032 passes.
- A bitwise failure rejects only the affected chunk. If no exact chunk reaches
  the materiality threshold, close the VAE scheduling line and compare
  serialization against FP8/fused attention using the same resident baseline.
- A pass transfers EXP-047 quality evidence only because decoded tensors are
  bitwise identical; it creates no new model-quality claim.
- One isolated H200, at most two GPU-hours, 5 GiB artifacts, and two bounded
  engineering repairs are allowed.

## Revisit condition

Reopen chunk scheduling only if the official VAE implementation, GPU backend,
or deployment frame geometry changes. A non-bitwise but visually similar chunk
requires a separate approximate-quality decision and cannot be rescued here.
