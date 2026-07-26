# Nystrom / Landmark Sparse-Tail Protocol

## Scope

This protocol tests whether current `Q/K/V` can generate a transferable
marginal Attention tail without a frozen output basis. It is a numerical
capacity and transfer probe. It is not an H200 latency benchmark.

The registered pilot uses Wan F81 captures at layer 14, sampling step 9,
conditional CFG branch, all heads, four existing samples, one query tile per
head, landmark counts `{32, 64, 128}`, and sparse densities `{0.125, 0.25}`.

## Associatively Executable Paths

Let

```text
F = softmax(Q K_L^T),
A = softmax(Q_L K_L^T),
B = softmax(Q_L K^T).
```

The signed Nystrom output is evaluated as

```text
Y_nys = F (A^+ (B V)),
```

not by materializing `F A^+ B`. The positive landmark output is

```text
Y_landmark = F (B V).
```

Both preserve the low-rank association. In contrast, clamping and
renormalizing the full approximate probability matrix requires an `N x N`
materialization and is diagnostic-only.

For a tile-shared critical-key mask `S`, the positive landmark partition uses

```text
m_S = F (B[:, S] 1),
Y_tail = (F (B V) - F (B[:, S] V[S])) / max(1 - m_S, eps),
Y = p_S Y_exact,S + (1 - p_S) Y_tail.
```

`p_S` comes from the deployable moment router. Dense-reference mass is logged
only as a diagnostic and is not a quality oracle.

## Arithmetic Proxy

All speed values are full-Attention arithmetic upper bounds. They are not
measured latency. The proxy includes:

- Nystrom/landmark core: approximately `2m/N` of dense Attention work.
- Exact critical branch: actual selected-key fraction.
- Moment router: `1/(2b)` for block size `b`.
- Landmark partition exclusion: `m*density/(2q) + m/(2N)` for query tile `q`.
- Full-matrix clamp materialization for the diagnostic path.

The proxy omits small `m x m` inversion, mask construction, launch, gather,
memory-layout, and fusion overhead. A fused H200 benchmark is mandatory before
an acceleration claim.

## Leakage And Selection

The sweep must be rectangular: every evaluated record has every candidate
configuration. A configuration is frozen from validation only. Test metrics
are read after the tuple `(method, landmark_mode, landmarks, pinv_rtol,
density)` is frozen. Dense-derived head roles never route a deployable path.

The current four captures were already used in exploratory work. Therefore,
the registered splits are within-run holdouts, not untouched external tests.
New prompts and seeds must be registered before inspection for confirmatory
claims.

## Gates

- Numerical gate: aggregate output error at most `1%`, worst record at most
  `2%`, arithmetic upper bound at least `1.5x`, and work ratio at most `0.5`.
- Deployment gate: measured fused H200 Attention speedup; initially
  `UNMEASURED`.
- Scientific gate: cannot pass while deployment is unmeasured.

A smoke run can never trigger stop/go. If the registered pilot fails, stop
expanding this train-free Nystrom/landmark family. That result does not reject
a learned content-conditioned tail, which is a separate hypothesis.

## Artifacts

Each probe writes `run_state.json` first and `SUCCESS.json` last. Input capture
payloads are SHA256-hashed by default, and size/mtime are checked again after
the run. Existing output directories are rejected; runners move prior outputs
to ignored `trash/<timestamp>-task/...` before rerun.
