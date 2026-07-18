# Controlled Spectral Trigger Result Bundle

This bundle records the final `2026-07-19` controlled synthetic trigger run on
server 210. It evaluates frame delta, an independent CausalMem residual proxy,
a single Oja state, a dual-state residual control, and the proposed
dual-timescale spectral trigger under matched basis-rank budgets.

## Reproduction

Runtime environment:

```text
Python 3.10.0
NumPy 2.2.6
Matplotlib 3.10.9
PyYAML 6.0.3
```

Command:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHON_BIN=/home/wangmeiqi/anaconda3/envs/Qwen3/bin/python
bash experiments/scripts/run_spectral_event_trigger.sh \
  remote_results/controlled_spectral_trigger_20260719_v6
```

The command returned exit code `2` because one preregistered gate failed. This
is an experimental verdict, not a runtime failure.

## Primary Finding

At rank budget 32, all learned-state methods used 8,320 bytes. Dual spectral
matched the CausalMem proxy at 80% event recall and reduced aggregate
negative-control false-trigger rate from 0.893% to 0.357%, a 60.0% relative
reduction. The paired seed-bootstrap difference was -0.536 percentage points
with a 95% interval of [-0.893, -0.045].

The preregistered rare-event gain was 0 rather than the required 10 percentage
points, and object-disappearance recall was 0%. The mechanism therefore stays
a diagnostic/selectivity result and is not promoted as the memory writer.

## Evidence Boundary

These are deterministic synthetic feature streams and CPU update timings.
They are not Video-LLM answer-quality results, an official CausalMem
reproduction, or end-to-end StreamingTOM/STC GPU latency. The independent
proxy label must remain attached to CausalMem comparisons.

The 15.9 MB `per_frame.csv` remains in the remote run directory and is omitted
from Git. Included CSV files retain aggregate, scenario, event, calibration,
and paired-seed evidence; PNG and PDF figures are generated from those data.
