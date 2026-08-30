# Measure-preserving compaction attempts

Both registered attempts are classified as **invalid engineering runs**. They
must not be used as positive or negative evidence about proportional attention.

- Attempt 1 failed because the equivalence guard checked full-vocabulary logits
  instead of the registered candidate logits.
- The single allowed repair narrowed only that guard.
- Attempt 2 still changed candidate logits by `0.25` when all token masses were
  one. The explicit four-dimensional additive mask therefore did not reproduce
  the ordinary two-dimensional model path.
- The repair allowance was exhausted, so the diagnostic stopped without a
  method verdict.

The remote attempt directories remain preserved. A future proportional-attention
test requires an implementation that is dense-equivalent in the same attention
kernel path before any mass-weighted result can be interpreted.
