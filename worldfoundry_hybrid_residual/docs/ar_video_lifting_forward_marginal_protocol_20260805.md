# Forward-Marginal Lifting Detail Gate v4

The v3 outcome-aware selector scored each detail block in isolation and then
retained the best 24 singleton scores. Independent review correctly identified
that this is not a capacity upper bound: detail blocks can be complementary or
redundant, so the best singleton ordering need not minimize the final joint
error.

v4 changes only the selector and storage validation. Starting from an empty
support, each greedy step evaluates every remaining candidate together with all
already selected blocks, including the same adaptive rank-16 capacity
diagnostic. The lowest-error candidate is committed and the process repeats
until 24 regular blocks are selected. This is a forward-marginal dense-label
oracle, not a deployable router and not a globally optimal combinatorial oracle.

The screen is restricted to the two development-exposed held-out Layer-14 hard
captures because Layer 14 is the current blocker. It uses exactly 24 padded
64-token detail tiles for every selector. After support search, the stored root
coarse map and every retained detail coefficient are round-tripped through
BF16 before replay. This separates the optimistic FP32-coefficient capacity
diagnostic from the logical BF16 payload assumed by the cache cost model.

The primary must satisfy aggregate `<=0.5%`, worst head `<=1%`, logical padded
BF16 cache compression `>=1.5x`, and at least 20% relative improvement over the
singleton selector. Failure blocks router training and kernel work. Because
forward greedy is still not global search, failure is strong evidence against
the current representation at this budget but is not an impossibility theorem.
