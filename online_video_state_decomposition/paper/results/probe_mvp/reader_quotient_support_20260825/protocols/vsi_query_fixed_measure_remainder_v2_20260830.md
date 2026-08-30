# VSI query-fixed visual-measure remainder Gate v2

Date: 2026-08-30
Status: frozen after engineering smoke and before the decision run
Supersedes: `vsi_query_fixed_measure_remainder_20260830.md`

## Reason for v2

The one-sample engineering smoke validated Q/K/V capture, attention replay, the
local analytic inequalities, and the result schema. It also showed that the v1
`exact_local_oracle` was only a one-shot first-order score and could be worse
than attention-mass ordering at some budgets. Calling it a capacity ceiling
would make a later null scientifically invalid.

The smoke output is preserved as engineering evidence only. It is excluded
from every v2 aggregate and decision. V2 keeps all v1 identities, endpoints,
budgets, guards, and costs, but makes the following bounded correction.

## Corrected selector family

Use five equal-budget selectors:

1. `analytic_remainder`: greedy reduction of the aggregate analytic
   certificate;
2. `attention_mass`: descending exact visual attention mass;
3. `exact_local_score`: the former one-shot first-order local-defect score,
   retained only as a diagnostic and never called a ceiling;
4. `exact_greedy_oracle`: after every selected node, evaluate the exact
   visual-output error for replacing each remaining coarse node by its four
   leaves, select the best current candidate, and repeat;
5. `fixed_random`: registered seed `20260830`.

The exact-greedy path is target-visible and non-deployable. It is a strong
registered sequential oracle, not a proof of the globally optimal cardinality
subset.

## Corrected outcome language

All v1 numerical thresholds remain unchanged. Classification uses
`exact_greedy_oracle` at 196 groups for the capacity condition.

- `QUERY_FIXED_CERTIFIED_HEADROOM`: unchanged v1 certified condition.
- `QUERY_FIXED_CAPACITY_BOUND_LOOSE`: exact-greedy satisfies the actual-error
  thresholds but the analytic candidate fails accuracy or certificate guards.
- `NO_REGISTERED_QUERY_FIXED_MEASURE_PATH`: exact-greedy fails the registered
  thresholds. This closes only the tested true-2x2 centroid and nested split
  family; it does not prove that every query-fixed support or hierarchy is
  impossible.
- `INVALID_QUERY_FIXED_MEASURE_GATE`: unchanged v1 implementation/identity
  failure class.

## Unchanged scope and stop rule

Use exposed positions 73--96, layers 0/13/27, the final query, all heads,
budgets 0/49/98/147/196/392, one isolated A800 for at most 30 GPU-minutes, and
one implementation repair. Positions 97--120, selection, formal, reader
logits, training, and latency remain out of scope.
