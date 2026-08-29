# RESULT-EXP-049: Target-separated conditional rate-distortion screen

- Experiment: `EXP-049`
- Gate: `G-028`
- Outcome: `boundary/null at Stage 0`
- Date: 2026-08-29

## Scope

The valid endpoint is the preregistered exposed-data local screen. It uses one
EXP-003 identity for calibration, one for policy selection, and two for held-out
evaluation over all 30 blocks, steps 2--19, both CFG branches, and the saved
eight token rows. It compares the same reuse, scalar AR(2), calibration-only
diagonal current-input field, and target-visible diagonal field for
self-attention, FFN, and whole-block targets.

This endpoint is not suffix risk and contains no H200 candidate timing. Those
stages were conditionally authorized only for a promoted target; none passed the
Stage-0 promotion rule.

## Validity

- Exact dense replay error in all loaded payloads was zero.
- The inherited sampled additive reconstruction floor was 0.156%--0.592% and
  remained below the accepted 1% artifact bound.
- The frozen split produced 12,960 selection and 25,920 held-out metric rows;
  all 38,880 rows were finite. The frontier contains all 18 registered
  target-threshold points.
- Attempt v1 stopped before evaluation on an old integer schema identifier.
  Attempt v2 stopped before evaluation on the inherited sampled-floor bound.
  Both logs remain preserved. Attempt v3 is the only scientific endpoint.
- Three synthetic recovery tests passed in the remote PyTorch environment.

## Frozen-policy result

At the registered 1% selection threshold:

| Target | Selected calls | Deployable agg./worst | Target-visible agg./worst | Gap recovery | Zero-renderer end-to-end ceiling |
|---|---:|---:|---:|---:|---:|
| Self-attention | 5.926% | 1.127% / 1.314% | 0.578% / 0.721% | 18.19% | 1.033x |
| FFN | 1.759% | 1.901% / 2.719% | 0.893% / 1.250% | -2.02% | 1.002x |
| Whole block | 0% | unavailable | unavailable | unavailable | 1.000x |

No target passes the 0.5%/1% target-visible, 1%/2% deployable, 70% recovery,
and 1.2x speed-ceiling conjunction.

The closest local-quality point is self-attention at a 0.75% selection
threshold: 0.741% of calls, 0.902%/1.143% deployable error,
0.541%/0.695% target-visible error, 61.98% gap recovery, and only a 1.004x
zero-cost end-to-end ceiling. Its eight selected cells occur only in blocks 0/1
and steps 6/8/9/10, so the signal is neither broad nor runtime-material.

At the relaxed 2% selection threshold, self-attention covers 26.111% of calls
but reaches 3.264%/4.390% deployable error and a 1.164x zero-cost ceiling. Whole
block covers 15.0% with 2.680%/3.452% error and a 1.176x zero-cost ceiling.
Both remain below 1.2x even before charging a renderer.

Across every held-out call rather than selected cells, the calibration-only
field is worse than AR(2):

| Target | AR(2) | Deployable field | Target-visible field |
|---|---:|---:|---:|
| Self-attention | 5.504% | 7.155% | 2.728% |
| FFN | 6.719% | 8.011% | 3.542% |
| Whole block | 6.118% | 7.758% | 3.507% |

## Why H200 timing was correctly skipped

For target runtime share `f`, selected fraction `q`, and nonnegative
candidate/exact cost ratio `rho`, the speedup obeys

`S = 1 / (1 - f*q + f*q*rho) <= 1 / (1 - f*q)`.

Even a free renderer therefore requires 30.93% self-attention coverage, 134.52%
FFN coverage, or 16.67% whole-block coverage to reach 1.2x. FFN cannot reach the
goal by itself. Every quality-plausible self-attention policy is orders of
magnitude below the required coverage, while relaxed attention and whole-block
policies miss both quality and the zero-cost speed bound. Positive measured
H200 cost can only worsen these points, so timing them cannot alter G-028.

## Interpretation

The result is consistent with the conditional-innovation theory. Current state
does contain information beyond history for a small early-layer subset, but a
calibration-shared diagonal field does not encode the content-dependent mode
over enough layer-step cells. Target visibility roughly halves all-call error,
showing conditional redundancy rather than absence of redundancy; the large
capacity-to-transfer and quality-to-coverage gaps show that this redundancy is
not exposed as a low-bandwidth deployable interface.

This refutes only the registered post-hoc diagonal conditional-interface
family. It does not refute same-step sparse attention, a learned content router,
released few-step students, training-native state/render separation, or
physical-time long-video memory.

## Decision

Close G-028 and park L-028. Do not spend fresh identities, suffix rollouts,
router training, or H200 kernel work on this function class. Revival requires a
different observable/function class with a development-only capacity result
that increases quality-valid coverage beyond the zero-cost Amdahl requirement,
not another coefficient, rank, BCM, or whole-block state sweep.

Artifacts:

- `worldfoundry_hybrid_residual/results/conditional_rate_distortion_exp049_20260829/local_rows.csv`
- `worldfoundry_hybrid_residual/results/conditional_rate_distortion_exp049_20260829/local_frontier.csv`
- `worldfoundry_hybrid_residual/results/conditional_rate_distortion_exp049_20260829/conditional_rate_distortion_screen.png`
- `worldfoundry_hybrid_residual/results/conditional_rate_distortion_exp049_20260829/conditional_rate_distortion_screen.pdf`
