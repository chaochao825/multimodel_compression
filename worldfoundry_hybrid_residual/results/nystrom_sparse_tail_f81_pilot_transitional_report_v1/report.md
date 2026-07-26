# F81 Calibration-Frozen Transitional-Head Tail Pilot

## Gate Status

| Gate | Status |
|---|---|
| Artifact completeness | `SUCCESS` |
| Numerical quality + arithmetic upper bound | `FAIL` |
| Measured H200 deployment | `UNMEASURED` |
| Scientific claim | `FAIL` |

**Decision:** STOP_THIS_TRAIN_FREE_FAMILY: do not expand this Nystrom/landmark sweep. A learned low-cost tail remains a separate, untested hypothesis.

The validation-frozen train-free candidates failed the registered pilot gate. This does not falsify content-conditioned learned tails in general.

## Frozen Results

Configurations were selected on validation only. Test metrics were evaluated after
the configuration tuple was frozen. Dense roles were used only on calibration to freeze a static head map; validation/test roles did not alter routing or selection.

| Protocol | Split | Head scope | Frozen method | m | Density | Aggregate | Worst record | Arithmetic upper bound |
|---|---|---|---|---:|---:|---:|---:|---:|
| seed_holdout | validation | transitional (2) | `proxy_mass_nystrom_mixture` | 128 | 25.0% | 19.025% | 21.432% | 3.76x |
| seed_holdout | test | transitional (2) | `proxy_mass_nystrom_mixture` | 128 | 25.0% | 20.169% | 21.903% | 3.76x |
| prompt_holdout | validation | transitional (2) | `proxy_mass_nystrom_mixture` | 128 | 25.0% | 19.048% | 19.444% | 3.76x |
| prompt_holdout | test | transitional (2) | `proxy_mass_nystrom_mixture` | 128 | 25.0% | 19.986% | 21.903% | 3.76x |
| combination_holdout | validation | transitional (2) | `proxy_mass_nystrom_mixture` | 128 | 25.0% | 19.034% | 21.432% | 3.76x |
| combination_holdout | test | transitional (2) | `proxy_mass_nystrom_mixture` | 128 | 25.0% | 20.989% | 21.903% | 3.76x |

## Claim Boundaries

- `seed_holdout`: Within-run holdout only: both prompts are observed before test and seed 20260741 is held out; these captures were used in earlier exploratory probes.
- `prompt_holdout`: Within-run holdout only: prompt 1 is absent from calibration and validation, while seeds are not held out; these captures were used in earlier exploratory probes.
- `combination_holdout`: Within-run holdout only: the prompt-1/seed-20260741 pair is held out while each factor is observed separately; these captures were used in earlier exploratory probes.

- The dataset contains four captures and is pilot evidence, not a population estimate.
- These captures were reused from exploratory work; this is not an untouched external test.
- A confirmatory claim requires new prompts and seeds registered before inspection.
- Run kind is `pilot`; smoke runs never trigger a scientific stop/go decision.
- Reported speed is an arithmetic upper bound, not wall-clock H200 speed.
- `dense_mass_*_diagnostic` isolates selected-mass error but is not a quality oracle.
- Full-matrix nonnegative clamping is diagnostic-only because it cannot preserve the low-rank association.
- Passing this report cannot produce a final acceleration claim while deployment is `UNMEASURED`.

## Reproducibility

- Selection run: `nystrom-select-c1e8730ad943-1785086890`
- Probe run: `nystrom-831276b91f3b-1785086663`
- Probe config SHA256: `831276b91f3b27ff86edba4f395c54af8a4ff79861a7b519aeec292e84037c8a`
- Probe resource mode: `shared_capacity_numerical_only`
- Probe detail SHA256: `580ba8a49de85ec11a82eb9c06013053cf52d70b6e90303ce960ca20dd5b1005`
- Split config SHA256: `17b0e8d0bd602e92488c0c12b512c5cdab021a6d22a9dd53e28737adf4a1ac0f`
