# VSI query-fixed Taylor cross-moment capacity Gate v2

Date: 2026-08-30
Status: frozen after engineering smoke and before the decision run
Supersedes: `vsi_query_fixed_taylor_cross_moment_20260830.md`

## Reason for v2

The position-73 engineering smoke stopped before writing scientific rows because
an odd-order truncated exponential produced non-positive coarse group masses.
This is a mathematical property of the registered polynomial, not a CUDA or
data error. Clipping or renormalizing it would silently change the method and
destroy the positive-measure interpretation.

## Corrected validity semantics

- Keep the registered order-0/1/2/3 polynomials, budgets, identities, metrics,
  and outcome thresholds unchanged.
- If any head-group has `Z_g^P <= 0`, mark that entire sample-layer-order cell
  invalid for every budget. Preserve the count and do not clamp, normalize, or
  substitute another approximation.
- Capacity pass requires every one of the 72 cells for that order to be valid.
- An invalid order does not invalidate other orders. In particular, an invalid
  order 3 cannot prevent a valid order 1 or 2 decision.
- Order 0 must still reproduce the previous headwise control exactly.

All remaining scope, cost, stop rules, and claim boundaries are unchanged. The
first smoke directory remains preserved as engineering evidence and is excluded
from the v2 decision aggregate.
