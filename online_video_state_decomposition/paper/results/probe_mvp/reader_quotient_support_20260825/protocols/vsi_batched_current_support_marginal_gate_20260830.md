# VSI batched current-support marginal Gate

Date: 2026-08-30
Status: frozen before execution

## Decision question

After position geometry and token measure are represented correctly, does
recomputing reader-aligned group utility from the current support produce a
high-fidelity nested quotient refinement path?

M0 established exact all-mass-one equivalence in one eager attention kernel.
The static empty-support singleton path still failed because group utility was
strongly support-dependent. This Gate measures whether receding-horizon utility
updates recover the required path.

## Data and identity

- Require the valid M0 summary with decision `SAME_KERNEL_MASS_VALID` and all
  three full-vocabulary errors at most `1e-5`.
- Use only already exposed VSI calibration positions 73--96, all 24 samples.
- Keep positions 97--120, selection, and formal endpoints unread.
- OneVision, eight frames, rank 456, four-token groups, original positions, and
  the eager attention equivalence harness remain frozen.

## Two independent paths

Evaluate both modes independently:

1. `positioned_equal_mass`: every compact token has mass one.
2. `positioned_group_mass`: every unrefined quotient token has mass four;
   exact and non-video tokens have mass one, implemented as `log(m_j)` key bias.

For each mode and sample, start with empty exact support. At current support
`Omega`, evaluate the actual compact reader once for every remaining group
`g` and compute

```text
benefit(g | Omega) = KL(p_dense || p_Omega)
                   - KL(p_dense || p_{Omega union {g}}).
```

Select the top 49 groups by benefit, breaking ties by group index, then
recompute every remaining marginal from the new support. Evaluate path counts
`0`, `49`, `98`, `147`, and `196`. Candidate forwards are batched by eight only
for execution efficiency; this does not change the path definition.

This is a batched receding-horizon teacher, not exact one-group-at-a-time greedy.
It reads dense candidate logits and is not deployable.

## Frozen outcome mapping

For each mode, a strict path exists if some registered `k <= 196` has:

- 24/24 dense-decision agreement;
- zero harmful flips;
- mean candidate KL at most `0.01`;
- P95 candidate KL at most `0.02`;
- no per-sample match-to-mismatch transition and no aggregate mean-KL increase
  larger than `1e-6` from zero through `k`.

Outcomes:

- `MASS_CURRENT_SUPPORT_HEADROOM` if group mass has a strict path and at that
  budget either equal mass does not, or group-mass mean KL is at least 20% lower.
- `CURRENT_SUPPORT_HEADROOM` if either mode has a strict path without the
  independent mass gain above.
- `DECISION_ONLY_BOUNDARY` if neither is strict but either mode reaches 24/24
  agreement with zero harmful at `k=196`.
- `NO_BATCHED_CURRENT_SUPPORT_PATH` otherwise.

A positive result authorizes a low-cost path generator study. A null closes this
49-group receding-horizon family, but does not refute exact sequential greedy or
train-time path-consistency adaptation.

## Cost and stop rule

- One isolated A800 on server 210, at most four GPU-hours.
- Candidate batch size eight.
- One implementation repair is allowed.
- Stop after one valid outcome or after the repair allowance is exhausted.
