# RESULT-EXP-051: True joint support-state capacity

- Experiment: EXP-051
- Gate: G-030
- Date: 2026-09-01
- Validity: valid exposed-development function-class ceiling
- Outcome: adverse / no-go

## Frozen comparison

The run used calibration positions 1--72 and exposed development positions
73--96 from the repaired exact-page capture. Positions 97--120, reader task
labels, deployable routing, confirmation, formal data, and H200 timing remained
unread or unrun. All arms used layers 0/13/27, four-token pages, 25% exact
tokens, width-32 positive-feature N/Z state, the same initialization, 1,000
steps, learning rate `3e-4`, seed 20260901, and one shared batch schedule.

The inherited whole-state checkpoint had been selected on this same exposed
development cohort by the earlier additive-state capacity screen. Therefore
EXP-051 is deliberately a favorable capacity screen, not held-out evidence.
EXP-051 training itself used calibration positions only and never updated a
parameter from a development target.

## Engineering validity

The run completed 432 finite development rows and 66 finite training-history
rows in 153.36 seconds. Both trainable arms reduced their calibration losses;
no gradient was nonfinite. A separate read-only replay of both saved states over
all 144 state-layer-development cells passed:

- exact page budget: 98 of 392 pages for every head and selector;
- maximum state-replay absolute error: `4.7684e-6`;
- maximum all-page dense-recovery absolute error: `3.5763e-6`;
- maximum all-page dense-recovery relative error: `2.3565e-7`;
- all checkpoint tensors finite.

This validation did not train, select, or read sealed identities.

## Aggregate results

| Method | Visual mean / P95 / worst | Full mean / P95 | Visual risk |
|---|---:|---:|---:|
| exact-only mass | 13.728% / 21.363% / 23.325% | 2.948% / 6.632% | 0.020674 |
| state-only | 31.509% / 52.518% / 65.178% | 2.863% / 7.453% | 0.110991 |
| mass-trained state + mass support | **4.019% / 7.651% / 10.536%** | **0.535% / 1.094%** | **0.002095** |
| mass-trained state + residual support | 6.591% / 18.060% / 23.396% | 0.978% / 1.759% | 0.007298 |
| residual-trained state + mass support | 6.322% / 12.553% / 17.077% | 0.685% / 1.091% | 0.005088 |
| residual-trained state + residual support | 4.281% / 13.642% / 21.254% | 0.548% / 1.220% | 0.003489 |

The analytic active-state ratio was `3.842x`, but no learned arm approached the
registered 0.5%/1%/2% visual capacity tier.

## Factorial interaction

- Joint residual versus independent mass: `-66.536%` risk improvement, 95%
  bootstrap interval `[-142.261%, -0.227%]`.
- Joint residual versus mass-trained state under residual support: `+52.198%`,
  interval `[32.009%, 70.852%]`. Adapting the state helps if residual support is
  imposed.
- Joint residual versus residual-trained state under mass support: `+31.437%`,
  interval `[-5.545%, 60.319%]`. The support contribution is not established.

The aggregate miss is localized but decisive. Joint residual improves visual
risk by `7.6%` at layer 0 and `32.1%` at layer 13, while layer 27 worsens by
`95.3%`; its visual mean/P95/worst becomes 8.251%/19.604%/21.254%.

## Interpretation

EXP-050 correctly found that residual-aware exact pages help a frozen
whole-measure state. EXP-051 shows that this fixed-state interaction does not
survive as a useful jointly adapted decomposition at the registered payload.
The best arm instead trains the state around stable mass pages. The residual
selector is a moving, target-visible discrete objective whose selected pages
and residual geometry change with the state; the deep-layer tail is not stable
enough for width 32 plus 25% regular pages to meet the local fidelity tier.

This does not refute conditional redundancy, train-native streaming memory,
irregular support, wider states, or other models. It refutes C-029's registered
post-hoc regular-page/width-32 mechanism before observability, task transfer,
or system cost become relevant.

## Decision

Close G-030 and park L-029. Per the accepted stop rule, do not train a router,
read positions 97--120, run reader-task intervention, or measure H200 latency.
Do not rescue the line by changing width, density, page regularity, steps, or
fallback after observing the result. The released rCM Wan baseline L-026
remains the independent acceleration mainline.
