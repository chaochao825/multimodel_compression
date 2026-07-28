# Train-Free Residual-Tail Oracle Protocol

## Purpose

This is the final bounded screen for stronger train-free Attention tails. It
does not revisit ordinary Nystrom rank, BCM capacity, or mass-only routing.
Every method keeps an exact dense-mass-oracle critical block set and evaluates
the remaining tail directly against `A V`.

The captures were used in earlier exploratory work. Dense probabilities may
select critical blocks, value-aware sampling weights, and a per-record best
candidate. Results are therefore post-hoc function-class capacity diagnostics,
not untouched tests or deployable routing results.

## Shared Normalization

All sparse and tail terms are combined as unnormalized numerator and
denominator contributions:

```text
Y_hat = (N_exact + N_tail_hat) / (Z_exact + Z_tail_hat)
```

No method separately normalizes sparse and tail outputs before mixing them.

## Families

### Value-aware coreset

Tail tokens are clustered per query tile using current K/V and optional THW
coordinates. The value-aware variant uses the dense diagnostic leverage

```text
w_j = sqrt(mean_i p_ij^2 ||v_j - y_i||_2^2)
```

for fitting. Cluster counts and unweighted K/V centroids define the tail
numerator and denominator. This is an oracle because the fitting weight reads
the dense reference.

### Residual-tail polynomial

After removing critical keys, orders 1 through 4 approximate `exp(s)` around
the per-query tail mean or midrange. Exact and approximate weights share one
numerical shift and one final denominator. Negative-weight and tail score-span
statistics are mandatory diagnostics. No arithmetic speed claim is made until
a TensorSketch or random-Maclaurin realization is implemented.

### Low-rank covariance moments

For each active tail group,

```text
Sigma_kk ~= D_k D_k^T
Sigma_vk ~= D_v D_k^T
```

is built from the current K/V tensors. Rank 4/8/16 and one/two components per
64-key block are compared with centroid and diagonal-Gaussian moments. Query
work is counted, but online SVD/moment formation remains an explicitly omitted
preprocessing cost during this capacity screen.

## Stop/Go Rules

- Oracle continuation: aggregate output relative L2 at most `0.5%` and every
  sample/head record at most `1%`.
- Deployable generator: aggregate at most `1%`, worst record at most `2%`, and
  no dense information at inference.
- Kernel: only after the deployable gate, with measured whole-Attention H200
  speedup at least `1.5x`.

If all three oracle envelopes fail, the train-free tail path stops. The next
experiment is a small learned Q/K-conditioned sparse-linear tail with frozen
base QKV, not a larger fixed landmark or BCM model.
