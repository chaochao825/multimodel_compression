# VSI true-2x2 quotient geometry control

Date: 2026-08-30
Status: frozen before execution

## Decision question

Does the M1 null primarily come from grouping a flattened `14x14` frame into
contiguous four-token groups, some of which cross row boundaries, rather than
from the frozen reader quotient interface itself?

This control isolates group topology at the empty exact support. It does not
test current-support routing, exact sequential refinement, or a PPE
implementation.

## Data and identity

- Require the valid M0 summary with decision `SAME_KERNEL_MASS_VALID`.
- Require the valid M1 summary with decision
  `NO_BATCHED_CURRENT_SUPPORT_PATH`.
- Use only already exposed VSI calibration positions 73--96, all 24 samples.
- Keep positions 97--120, selection, and formal endpoints unread.
- Freeze OneVision, eight frames, rank 456, group size four, original
  positions, eager attention, candidate tokens, prompts, and dense endpoint.
- Use the same representative constituent offset, `1`, in both geometries.

## Equal-cost geometries

Evaluate both mass modes for each geometry:

1. `flat_contiguous_4`: the existing per-frame flattened groups
   `[0:4], [4:8], ...`.
2. `spatial_2x2`: reshape each frame to `14x14` and group true non-overlapping
   `2x2` patches in row-major block order.

Both geometries produce exactly 49 quotient tokens per frame and 392 over
eight frames, so reader token retention remains `25%`. For each geometry test:

- `positioned_equal_mass`: every compact token has mass one;
- `positioned_group_mass`: every quotient token has mass four via `log(m_j)`.

The flat rows must reproduce the M1 `k=0` prediction and candidate KL within
`1e-6`; otherwise the run is invalid.

## Frozen outcome mapping

For each mass mode compare `spatial_2x2` with `flat_contiguous_4`.

`TRUE_2X2_GEOMETRY_HEADROOM` requires at least one mode to satisfy all four:

- at least two fewer dense-decision mismatches;
- no increase in harmful flips;
- mean candidate KL at most `0.8x` the flat value;
- P95 candidate KL at most `0.8x` the flat value.

`TRUE_2X2_DECISION_HEADROOM` applies when strict headroom fails but at least one
mode has at least two fewer mismatches, no harmful increase, and mean KL no
more than `1.05x` the flat value.

`NO_TRUE_2X2_GEOMETRY_HEADROOM` applies otherwise.

Strict headroom authorizes rerunning the M1 current-support path with true
`2x2` groups. Decision-only headroom authorizes only a paper-faithful
multi-position/PPE control. A null parks further train-free topology tuning;
it does not refute PPE because a single representative position is still used.

## Cost and stop rule

- One isolated A800 on server 210, at most 30 GPU-minutes.
- One implementation repair is allowed.
- Stop after one valid outcome or after the repair allowance is exhausted.
