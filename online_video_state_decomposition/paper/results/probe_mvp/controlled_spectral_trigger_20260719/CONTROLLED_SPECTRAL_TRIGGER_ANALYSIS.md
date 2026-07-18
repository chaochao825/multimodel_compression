# Controlled Spectral Event Trigger Analysis

## Evidence Boundary

This is a deterministic synthetic feature-stream experiment. It tests
event-trigger behavior and CPU proxy cost, not Video-LLM answer quality.
`causalmem_residual_proxy` is an independent mechanism proxy and not the
official CausalMem implementation.

## Primary Matched-Rank Result

Total basis-rank budget: `32`. Detection tolerance: `2` frames.

| Method | State | Event recall | Rare recall | Camera FTR | Scene delay | Frame AUC | Update P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CausalMem residual proxy | 8,320 B | 80.0% | 100.0% | 0.0% | 0.00 | 0.861 | 256.9 us |
| Dual-state residual only | 8,320 B | 80.0% | 100.0% | 0.0% | 0.00 | 0.910 | 401.1 us |
| Dual-timescale spectral | 8,320 B | 80.0% | 100.0% | 0.0% | 0.00 | 0.880 | 401.1 us |
| Frame delta | 16,384 B | 20.0% | 0.0% | 4.9% | 0.00 | 0.484 | 39.6 us |
| Single Oja residual | 8,320 B | 80.0% | 100.0% | 0.0% | 0.00 | 0.909 | 293.3 us |

## Scenario Localization

| Event scenario | CausalMem proxy | Dual residual | Dual spectral |
|---|---:|---:|---:|
| object enter | 100.0% | 100.0% | 100.0% |
| object disappear | 0.0% | 0.0% | 0.0% |
| scene cut | 100.0% | 100.0% | 100.0% |
| one frame ocr | 100.0% | 100.0% | 100.0% |
| brief action | 100.0% | 100.0% | 100.0% |

| Negative scenario | CausalMem proxy FTR | Dual residual FTR | Dual spectral FTR |
|---|---:|---:|---:|
| static | 0.0% | 0.0% | 0.0% |
| camera slow | 0.0% | 0.0% | 0.0% |
| camera fast | 0.0% | 0.0% | 0.0% |
| lighting drift | 4.5% | 3.6% | 1.8% |
| periodic motion | 0.0% | 0.0% | 0.0% |

## Paired Selectivity Analysis

Paired bootstrap intervals resample the eight evaluation seeds. Differences are dual spectral minus baseline false-trigger rate; negative values favor dual spectral. This is secondary evidence, not a replacement for the preregistered recall gate.

| Baseline | Baseline FTR | Dual FTR | Relative reduction | Difference (pp) | Paired 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| CausalMem residual proxy | 0.9% | 0.4% | 60.0% | -0.536 | [-0.893, -0.045] |
| Single Oja residual | 0.9% | 0.4% | 61.9% | -0.580 | [-0.893, -0.179] |
| Dual-state residual only | 0.7% | 0.4% | 50.0% | -0.357 | [-0.670, 0.089] |

## Gates

- `camera_false_trigger_rate`: value `0.0`, threshold `0.05`, passed `True`.
- `scene_cut_delay`: value `0.0`, threshold `2.0`, passed `True`.
- `event_recall_vs_causalmem_proxy`: value `0.0`, threshold `0.0`, passed `True`.
- `rare_recall_vs_residual_only`: value `0.0`, threshold `0.1`, passed `False`.
- `writer_p95_us`: value `401.11265000000003`, threshold `100000.0`, passed `True`.
- `matched_basis_state_bytes`: value `1.0`, threshold `1.05`, passed `True`.

## Failure Localization

- Rare-event recall gain was `0.0` percentage points versus the required `10.0`.
- Dual spectral object-disappearance recall was `0.0%`; the spectral terms did not recover this missing event class.
- The observed advantage is lower false-trigger rate at unchanged event recall, not improved rare-event recall.

## Verdict

The dual-timescale trigger **does not pass** the preregistered synthetic gate.
A pass only promotes the mechanism to native-feature validation. A
failure keeps spectral state as a diagnostic rather than a memory writer.
Single-Oja task-memory results remain rejected regardless of this trigger
outcome.

## Files

- `summary.csv`: method/rank metrics and latency percentiles.
- `per_scenario.csv`: event recall and negative-control FTR by scenario.
- `paired_false_trigger_bootstrap.csv`: paired seed-bootstrap comparisons.
- `paired_false_trigger_by_seed.csv`: raw seed-level paired rates.
- `per_frame.csv`: complete score and trigger traces.
- `event_outcomes.csv`: event-level recall and delay.
- `calibration.json`: negative-only scales and thresholds.
- `event_recall_vs_false_trigger_rate.*`: quality trade-off.
- `spectral_trigger_traces.*`: representative normalized traces.
- `spectral_trigger_update_latency.*`: CPU P95 update cost.
- `scenario_event_recall.*`: event recall localized by scenario.
- `scenario_false_trigger_rate.*`: false triggers by negative scenario.
