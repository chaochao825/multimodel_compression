# RDR-038: Propose exact full-F81 Wan VAE CUDA Graph replay

- Status: proposed
- Date: 2026-09-01
- Decider: researcher acceptance pending
- Supersedes: none

## Context

L-030 is the exact incumbent at `9.637995s` per resident rCM4 F81 request.
Its VAE occupies `4.308300s` or 44.7%, denoising `3.205365s`, serialization
`1.796082s`, transfer `0.253741s`, and text `0.064420s`.

EXP-053 closed only temporal regrouping: its selected chunk changed F81 decoded
tensors and missed the complete speed guards. EXP-054 then measured
`1.586377x` Sage self-attention speed but selected `0/120` safe train-free cells.
Neither result tests whether the unchanged 21-frame VAE kernel sequence is
launch-bound.

The official decoder calls `Decoder3d` once per latent frame, maintains causal
feature caches, and repeatedly concatenates output. A full-call CUDA Graph can
capture that fixed F81 sequence without changing frame grouping, operators,
weights, BF16, or cache contents. This is a distinct exact execution question,
not a repair of EXP-053.

PLAN-066 calculates that a `1.119x` VAE is sufficient for a `1.05x` resident
request and `1.255x` VAE for a `1.10x` request. Measured Sage speed under
optimistic full self-attention coverage reaches only a `1.070935x` request
ceiling and currently has zero certified coverage.

## Options

1. **Exact full-F81 VAE CUDA Graph replay.** Preserve the complete official
   temporal schedule and kernel sequence; test only static capture/replay.
2. Optimize transfer/serialization first. This has a lower latency ceiling and
   risks changing the artifact or measuring throughput rather than request
   latency.
3. Train a low-precision rCM checkpoint first. This has higher data, quality,
   and implementation cost with a smaller current Amdahl return.
4. Stop at L-030.

## Proposed decision

Accept option 1 as proposed `C-033 / L-033 / EXP-055 / G-034`. No execution is
authorized until this RDR is explicitly accepted.

The experiment may add a separate harness around the frozen official rCM source
but may not edit the source or checkpoint. Its action is exactly one
`torch.cuda.CUDAGraph` capture/replay of the unmodified complete VAE decode.

### S0: F17 engineering and stale-state screen

- Generate or load three deterministic rCM4 on-policy F17 latents of the same
  shape.
- Warm eager decode, capture one static graph, then copy each different latent
  into the static input and replay in registered order.
- Compare every graph output to an independent eager decode with `torch.equal`.
- Include static input copy, output handoff, and required synchronization in
  latency; do not time a bare replay that omits deployable work.
- Stop as `engineering-null` if capture is unsupported after at most two bounded
  repairs; stop as `exactness-null` on any non-bitwise output or stale-state
  failure.

### S1: F81 exactness and local materiality

- Use the four EXP-052 prompts and its fixed rCM4 seed to obtain real F81
  on-policy latents. No identity may select graph configuration.
- Compare eager and graph on all four inputs, with at least two replay rounds
  after warmup and identical synchronization.
- Require bitwise GPU and decoded CPU tensor equality, finite timings, unchanged
  network-call count, and peak reserved memory at most `59,948 MiB`, twice the
  EXP-052 request peak.
- Require median complete-VAE speedup at least `1.12x` and projected resident
  request speedup at least `1.05x` before S2.

### S2: complete resident request

- Run the exact EXP-052 resident rCM4 endpoint with graph VAE and the same four
  prompts, seed, output path, transfer, and serialization contract.
- Promotion requires decoded CPU tensors bitwise equal, all request guards
  valid, and median request at least `1.05x` faster than `9.637995s`.
- Record `1.25x` VAE and `1.10x` request as stretch thresholds, not alternate
  pass rules.

## Prohibited changes

No temporal regrouping, `torch.compile`, kernel fusion, new convolution or
attention kernel, FP8/quantization, cache approximation, scheduler/model
change, codec/preset change, asynchronous response contract, or tolerance
relaxation is allowed.

## Outcome mapping

- `pass`: every exactness, memory, local-speed, and endpoint-speed guard passes;
- `speed-boundary`: exact S1 passes but S2 misses `1.05x`;
- `exactness-null`: capture runs but any graph output differs bitwise;
- `performance-null`: exact graph remains below `1.12x` complete-VAE speed;
- `engineering-null`: capture cannot be made runnable within two repairs;
- `invalid`: identity, baseline, isolation, timing, or source guards fail.

## Consequences if accepted

- L-030 remains the immutable incumbent until a valid G-034 pass.
- One isolated H200, at most two GPU-hours, 5 GiB artifacts, and two bounded
  repairs are allowed.
- A null closes only CUDA Graph replay. It does not reopen EXP-053 and does not
  refute a later custom exact VAE kernel/backend Gate.
- Trainable low-precision attention remains a separately protected decision.

## Revisit condition

Reopen this exact graph candidate only if the VAE graph, PyTorch/CUDA backend,
GPU architecture, or deployment shape changes.
