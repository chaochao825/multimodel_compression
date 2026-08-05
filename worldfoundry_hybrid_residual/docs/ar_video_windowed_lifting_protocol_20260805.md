# Windowed Butterfly-Lifting v2 Gate

The v1 global-shift transform is a valid null: at 20% details it obtained
`14.13%/25.45%` direct and `8.00%/15.30%` adaptive-rank-16 held-out error.
Its cyclic predictor improved the identity transform by only `0.82%` relative.
Shared shifts were zero in 27/40 merge nodes, so the remaining uncertainty is
not global motion but piecewise spatial motion.

v2 changes exactly one logical unit: the prediction operator. Each dyadic merge
uses a block-diagonal permutation consisting of cyclic shifts inside `8x8`
windows. The partition alternates `(0,0)`, `(4,4)`, and `(0,4)` offsets across
tree levels, analogous to shifted-window/Butterfly communication. The transform
remains exactly invertible when all details are retained. Shifts are shared
across heads inside a window and selected only from current K/V reconstruction
energy, giving 28 small regular windows rather than token-level flow.

The primary retains 19% of 64-token detail blocks. This fraction was fixed from
byte accounting, not quality: 20% in v1 produced `1.495x` after metadata, while
19% leaves one fewer block and restores the registered `>=1.5x` systems guard.

The same eight hard captures, exact sink/recent frames, K/V weights, adaptive
rank, and quality evaluator remain unchanged. The primary must both pass
`0.5%/1%` adaptive capacity and improve at least 20% over a same-budget global
shift. Otherwise piecewise cyclic prediction is stopped; no learned flow,
kernel, or rollout is permitted.
