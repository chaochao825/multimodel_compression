# Residual-Informed Lifting Detail Oracle v3

Global and 8x8-window cyclic predictors both failed at the registered systems
budget. One uncertainty remains: the runtime K/V-energy selector may preserve
large transform coefficients that have little effect on `AV`, producing the
observed 10%--20% plateau.

For each of the 125 regular detail blocks, v3 reconstructs that singleton on
top of the zero-detail coarse tree, evaluates dense `AV`, applies the same
adaptive rank-16 capacity diagnostic, and scores reduction in residual energy.
The top 19% blocks are replayed together. This uses forbidden dense labels and
is not a deployable router; it is an upper-bound diagnostic analogous to the
earlier residual-width write screen.

The transform, captures, exact frames, shift bank, K/V weights, rank, byte
accounting, and evaluator are unchanged. The oracle must reach `0.5%/1%` at
`>=1.5x` cache compression and improve at least 20% over K/V-energy selection.
Failure closes the whole lifting/local-cyclic line rather than motivating more
windows, shifts, ranks, or learned routing.
