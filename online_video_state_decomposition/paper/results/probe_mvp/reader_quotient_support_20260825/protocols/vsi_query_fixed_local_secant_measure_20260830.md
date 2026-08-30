# VSI query-fixed local secant measure Gate

Date frozen: 2026-08-30

Role: exposed calibration capacity diagnostic

## Decision question

Can a non-periodic local rank-1 secant preserve the discrete K/V coupling inside
each four-token visual block better than a centroid or Gaussian moment, while
retaining a plausible two-times attention arithmetic reduction and four-times
BF16-to-INT8 state-byte reduction?

This Gate tests one fixed final-token query at Qwen2 layers 0, 13, and 27. It does
not establish full-reader accuracy, a reusable multi-query KV cache, TTFT, kernel
latency, official selection, or formal generalization.

## Frozen identities

- Model and split protocol: the existing frozen OneVision/VSI setup.
- Data: calibration positions 73--96 only; positions 97--120, official selection,
  and official formal remain unread.
- Geometry: eight frames with a 14x14 token grid per frame.
- Layers: 0, 13, and 27.
- Group budget: 392 groups of four tokens, compared under both flat-contiguous-4
  and true spatial-2x2 partitions.
- Numerical reference: captured post-RoPE Q/K/V and exact shared softmax
  numerator/denominator in FP32 after a valid native BF16 replay check.

## Frozen candidates

1. centroid: one mean K and one mean V per group;
2. key secant: rank-1 K SVD with V regressed on the same member coordinate;
3. joint secant: rank-1 SVD of scale-balanced centered [K,V];
4. independent secant: separate rank-1 K and V member coordinates, with member
   pairing retained in the exact four-point exponential sum.

Every candidate is evaluated without vector quantization and with deterministic
per-vector symmetric INT8 simulation. No query, attention weight, output defect,
answer, or reader gradient is used to choose a secant direction.

## Cost model

Dense four-member attention charges four QK dot products and four weighted-V
accumulations. A rank-1 secant charges two direction dot products and two vector
accumulations plus four scalar exponentials. The registered arithmetic proxy is
therefore 2x for secants. INT8 secant state stores mean/direction K and V plus
FP16 member coordinates and scales; its byte ratio is computed against BF16
dense K/V and must be at least 3.5x.

## Decision rule

`LOCAL_SECANT_DEPLOYABLE_CAPACITY` requires one target-free INT8 secant candidate
to satisfy all:

- 72 sample-layer cells;
- visual output relative-L2 mean/P95/worst at most 1%/2%/5%;
- full output relative-L2 mean/P95 at most 0.5%/1%;
- arithmetic proxy at least 1.8x;
- state-byte ratio at least 3.5x.

If no INT8 candidate passes, `LOCAL_SECANT_CAPACITY_ONLY` requires an unquantized
secant to satisfy the stricter visual 0.5%/1%/2% and full 0.25%/0.5% thresholds.
Otherwise the decision is `NO_LOCAL_SECANT_PATH`.

A deployable-capacity result permits only a separately frozen multi-query and
reader-level Gate. A capacity-only result permits one quantization-repair
diagnostic. A null closes train-free local rank-1 secants; it does not refute
trained native memory tokenizers or higher-cost local decompositions.
