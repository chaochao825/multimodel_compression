# VSI exact-boundary plus additive-tail oracle Gate

Date frozen: 2026-08-30

Role: calibration-development capacity oracle only

## Decision question

The width-32 additive `N/Z` state failed when it had to approximate the whole
visual attention measure. This Gate changes the function class once: retain an
exact 25% boundary and train the same additive state only on the residual bulk.

For exact support \(\Omega(q)\), the shared-normalization reader is

\[
\hat y(q)=
\frac{N_{\Omega}^{\mathrm{exact}}(q)+\hat N_{\bar\Omega}(q)}
     {Z_{\Omega}^{\mathrm{exact}}(q)+\hat Z_{\bar\Omega}(q)}.
\]

The experiment asks whether removing high-leverage atoms changes the residual
bulk into a width-32 additive state. It is not a deployable support selector.

## Frozen identities

- Frozen LLaVA-OneVision Qwen2-7B reader and existing exact post-RoPE capture.
- Training positions 1--72; exposed development positions 73--96.
- Confirmation positions 97--120 are not captured or read.
- Official selection and formal roles remain unread.
- Language layers 0, 13, and 27; eight frames; 1,568 visual tokens.
- Exact visual-token fraction: 25% per sample, layer, and head.
- Additive feature width: 32; AdamW; 1,000 steps; batch size 2; seed 20260830.
- Shared visual/nonvisual numerator and denominator; no branch-wise output mix.

## Capacity selectors

Both selectors use the exact current query and are oracle diagnostics:

1. `mass_topk`: highest exact visual attention logits.
2. `effect_topk`: highest first-order output effect
   \(e^{s_j}\lVert v_j-y_{\mathrm{visual}}\rVert_2\).

Neither selector supports a runtime or speed claim. Their only purpose is to
determine whether the remaining bulk is representable before developing a
cheap boundary router.

## Cost proxy

The analytic active-state ratio charges exact K/V for 25% of tokens plus the
BF16 additive `(S,z)` state. It excludes dense oracle scoring, writer feature
projection, kernel fusion, and cold exact storage.

## Decision rule

`BOUNDARY_ADDITIVE_TAIL_ORACLE_GO` requires one selector to satisfy all:

- 72 development sample-layer cells;
- visual mean/P95/worst at most 0.5%/1%/2%;
- full mean/P95 at most 0.25%/0.5%;
- visual mean at least 25% below the same-support exact-only baseline;
- analytic active-state ratio at least 2x.

`BOUNDARY_ADDITIVE_TAIL_CAPACITY_SIGNAL` uses visual 1%/2%/5%, full
0.5%/1%, the same 25% improvement, and ratio threshold. It permits a later
separately frozen support-generation Gate, not confirmation.

Otherwise the result is `NO_BOUNDARY_ADDITIVE_TAIL_PATH`. A no-go stops wider
feature maps, more training steps, confirmation, and official selection for
this decomposition. No outcome authorizes a latency or reader-task claim.
