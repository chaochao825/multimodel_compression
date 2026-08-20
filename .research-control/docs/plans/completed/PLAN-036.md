# PLAN-036: Joint QK/V scale-granularity capacity for one-pass Sage

- Status: completed
- Owner: researcher and Agent
- Gate: exposed low-precision scale-capacity diagnostic after DIAG-035
- Claims: none; all DIAG-030 identities are globally exposed
- Candidate line: per-token Q/K INT8 scale plus block-group V E4M3 scale
- Lane: explore
- Resource cap: immutable DIAG-030 QKV, one implementation repair, one
  exclusive H200 pass under 90 minutes, no custom kernel or rollout

## Decision to unlock

Determine whether changing only quantization **scale granularity** can turn the
faster thread-no-smooth Sage path into a no-fallback low-precision operator
that passes all Layer-0/29 cells.

The primary candidate uses:

- one INT8 scale per Q token/head and K token/head across 128 channels;
- one E4M3 scale per head × 128-key block × 16-channel group for V;
- the same dense attention support, online-softmax order, and FP32+FP32 PV
  accumulation model as the registered Sage path;
- no sparse blocks, low-rank correction, second pass, runtime router, or BF16
  precision island.

This is a capacity and numerical-model gate. It does not implement or time the
new scale layout in the CUDA kernel.

## Scope and evidence boundary

- Reuse 96 hashed DIAG-030 captures for layers `[0,29]`, steps
  `[2,7,12,17]`, both CFG branches, 16 fixed 64-query tiles, 12 heads, 128
  channels, and 32,760 tokens.
- Preserve calibration/evaluation labels only for tables; all identities are
  globally exposed and cannot support transfer claims.
- Actual baseline is installed Sage SM90 with per-thread INT8 Q/K,
  `smooth_k=false`, global per-(batch,head,channel) E4M3 V, and `fp32+fp32` PV.
- Use the installed Sage per-thread quantizer to reconstruct the current
  dequantized Q/K and scale map. Do not approximate the incumbent scale layout
  with a separately written quantizer.

## Counterfactual delta evaluator

For current and candidate quantized operands, construct modeled online
probabilities with the Sage SM90 source-faithful running-max E4M3 numerator and
FP32 denominator. The model multiplies natural-scale logits by `log2(e)`, uses
base-2 exponentiation, and applies source constant `S_FP8_OFFSET=8.807`, which
places the local maximum near E4M3 value `448`.
For each query tile evaluate:

`Y_candidate = Y_actual_sage + Y_modeled_candidate - Y_modeled_current`.

This preserves actual Sage residuals shared by both paths and changes only the
registered operand scales. Also report direct modeled error and an exact-softmax
delta variant. A pass requires the modeled-online delta; exact softmax is a
diagnostic only.

## Frozen scale families

1. `current`: installed per-thread Q/K scales plus global-channel V scale.
2. `token_qk_global_v`: one Q/K scale per token/head; current V scale.
3. `token_qk_blockhead_v`: one V scale per 128-token block/head.
4. `token_qk_blockgroup16_v`: **primary**, one V scale per block/head/16
   channels.
5. `token_qk_blockchannel_v`: one V scale per block/head/channel; metadata
   upper boundary.
6. `token_group32_qk_blockgroup16_v`: four Q/K scales per token/head over 32
   channels and block-group16 V; arithmetic/scale-composition upper boundary.
7. `bf16_qk_blockchannel_v`: perfect-QK plus finest registered FP8 V scale;
   low-precision V capacity upper boundary.
8. `bf16_qk_bf16_v`: dense operand upper control; no deployment eligibility.

All INT8 quantizers use symmetric max-abs `/127` rounding. All V methods use
E4M3 max finite `448`. Padded tokens are masked from QK and excluded from scale
statistics.

## Hardware-bounded metadata

At 32,760 tokens, 12 heads, and FP16-stored scales:

- tokenwise Q+K scale metadata is approximately `1.50 MiB`;
- block-group16 V scale metadata is `48 KiB`;
- block-channel V scale metadata is `768 KiB`;
- token-group32 Q+K scales are approximately `6.00 MiB` and are an upper
  boundary, not the primary deployment layout.

Primary scale metadata must remain below `2%` of Q/K/V low-precision payload
bytes, and modeled extra scale arithmetic must remain below `5%` of dense
attention arithmetic. These are cost guards, not measured latency.

## Metrics and outcomes

Quality guards remain aggregate `<=1%`, worst head `<=2%`, and worst tile
`<=2%` for every one of the eight evaluation-labeled cells.

Report cell/identity/head/tile errors, exact versus online delta gaps,
Q/K/V operand relative errors, saturation, scale dynamic range, metadata,
current-model fidelity to actual Sage, and error-energy reduction versus stock.

- `joint-scale-pass`: primary modeled-online counterfactual passes all 8 cells,
  current-model fidelity is `<=0.5%` aggregate in every cell, and cost guards
  pass. Authorize a custom quantizer/attention-kernel timing gate.
- `fine-scale-boundary`: block-channel V or group32-QK passes all cells while
  primary fails. Retain only if a lower-cost approximation can be derived; do
  not write CUDA yet.
- `softmax-model-boundary`: primary exact-softmax delta passes all cells but
  modeled-online delta fails. Audit actual online-exp state before changing
  scale granularity.
- `qk-boundary`: token-QK materially improves the perfect-V residual but
  primary still fails one or two cells; retain one bounded scale-layout
  refinement only if the remaining worst error is `<=2.2%`.
- `scale-null`: even `bf16_qk_blockchannel_v` fails, or no deployable family
  improves all three error metrics by at least 10% in both layers. Close custom
  low-precision operand work.
- `invalid`: identity/provenance mismatch, leakage, changed families/gates,
  nonfinite output, or current-scale reconstruction mismatch.
- `engineering-failure`: cannot complete within one repair and the resource
  cap; preserve evidence and stop.

## Stop rules

Do not alter scale families, block/group sizes, rounding, probability model,
delta evaluator, quality/cost gates, or method order after the first formal
capture. Do not train, tune clipping, add sparse/low-rank/residual branches,
write CUDA/Triton, capture fresh identities, or run rollout in this plan.

Before the first formal capture, the smoke test found that the earlier
DIAG-032 helper quantized natural-exponent weights in `[0,1]`; this did not
match the registered Sage SM90 source. The single allowed implementation
repair replaced it with the base-2, offset-8.807 model above. Historical
DIAG-032 code and results remain unchanged.

## Frozen implementation hashes

- Config: `a0600dc680c672496d66d0dc48f66c232f0f52f28e39f48ae06b039ec6a9f15d`
- Core: `5f51ba5711248d8e1cca0cf846e2ae23a43cca7f6aebde1728bfcf19724b0b1d`
- Test: `481734d99d034d5c8716bf7fcf08f5057a896e6009780024b50a28c38549cc17`
- Probe: `9d0757befa428d7609ac241280de1991b8abc5e82fa2474cc780291cb2343c4b`

## Required closure

Archive frozen hashes, reconstruction tests, records, cell and identity
summaries, operand/metadata diagnostics, decision, manifest, one visualization,
and one Chinese report. Classify before opening a kernel gate.

## Closure

- Closed date: `2026-08-12`.
- Validity class: valid, fully exposed scale-capacity diagnostic.
- Outcome: `scale-null`.
- Required action: `CLOSE_CUSTOM_LOW_PRECISION_OPERAND_WORK`.
- Formal run: `96/96` immutable DIAG-030 captures, `9/9` tests, `106.56 s`
  probe runtime, exclusive H200 GPU3, and zero foreign GPU processes.
- Source-model repair: the single preregistered implementation repair replaced
  the old natural-exponent `[0,1]` probability proxy with the installed Sage
  SM90 base-2 online-softmax model using `S_FP8_OFFSET=8.807`. Current-model
  fidelity was at most `0.313%`, and exact-versus-modeled candidate aggregate
  gaps were at most `0.0028` percentage points.
- Primary token-QK plus block-group16-V candidate passed `0/8` cells. It
  reduced stock error energy by only `1.1--4.9%` per cell, below the frozen
  `10%` layerwise boundary.
- The strongest deployable family, token-QK plus block-channel E4M3 V, also
  passed `0/8`, with mean aggregate error `1.833%` and maxima
  `2.287% / 2.887% / 2.458%` for aggregate/head/tile error.
- Perfect BF16 Q/K plus block-channel E4M3 V still passed `0/8`. The BF16-QKV
  control retaining modeled online P/PV passed `7/8`; Layer-0 step-2 remained
  at `1.073%` aggregate error.
- Primary scale metadata was `1.074%` of low-precision payload and the modeled
  extra arithmetic was `2.344%`; both cost guards passed. The null is numerical,
  not a metadata-budget rejection.
- Supports: finer Q/K scale granularity reduces operand error, especially on
  Layer 29, and the source-faithful counterfactual evaluator isolates this
  improvement reliably.
- Does not support: scale granularity alone as a no-fallback strict-quality
  replacement for the faster Sage path, or further custom scale-layout kernel
  work under this function class.
- Unknown: trained QAT, other number formats, a prospective full-model
  precision policy, rollout quality, and end-to-end speed; none was tested.
- Action: reject the scoped custom operand-scale candidate. Preserve the
  DIAG-029 grouped precision-island implementation witness and the DIAG-030
  coverage boundary, but do not infer a new rollout or kernel authorization.
- Runner SHA-256:
  `d3630fa3e0bded92b388db35cbbbf8e38188b253ab8a93a5ec296a56547e2bea`.
- Raw records SHA-256:
  `a68778c2754f348fefb75768fa8e3826dc9d5b805dd2d9bd6b991ecb5e2363a3`.
- Cell / identity / operand summaries SHA-256:
  `4d93ac82cddef4a067d06ba7f69299db4a4c62c5dc65da9a6ee086cc8d6277ac` /
  `426ce4fc4c645cd6c405debbee59dca3415e8b521d77f47aa6e6e742ee389869` /
  `2f96bae9f9287d54f03da1c5bda2c97800c834d62a02b6c52211c2c39405f007`.
- Cost / decision / manifest SHA-256:
  `ca499eba21ae44fb2f9fad9d3e56fab9d0cd6da5776336fe1edca7f18a5f8a19a` /
  `3363286030d963b602d0c867476a8f8f5785eec9f6b7737156c0bcb705548f42` /
  `bce29a14cd399618c4dfbf9d0977509f89bcfaf252871b57aa2cfaa22ad467fa`.
- Report SHA-256:
  `e7e19b64a36c592b1480bb20202642e7d49e7ce0c5a76c51cc55048996d1a1d6`.
- Figure PNG / PDF / CSV SHA-256:
  `6c7ba607ede5c0ea2f26c3227ec6ed0cffe36588be3ec90f4bee17a14e1e04fa` /
  `74c6e7d78f7e8dc6288474f9f4c8d6ad227663674a9cde0b54c5999251eade94` /
  `03b8e93048081f96f700d53c0839b3297c4896b547405472297c1266db84934c`.
