# Nystrom / Landmark Sparse-Tail Failure Analysis

The registered pilot fails. A post-hoc search over all four captures
(diagnostic only, not a validation-frozen test estimate) finds
`proxy_mass_nystrom_mixture` with `m=64` and density
`25.0%`. Its aggregate error is
`22.684%`, worst-record error is
`57.062%`, and arithmetic speedup upper bound
is `3.82x`.

The calibration-frozen transitional-head selection also fails at
`19.025-20.989%`
aggregate error. Therefore, the all-head failure is not explained
only by applying the tail to diffuse heads.

The selected-mass/output-error Pearson correlation is `0.380`.
Increasing landmark count raises middle-matrix condition number and signed
negative mass while delivering only modest quality improvement. These results
support stopping this train-free Nystrom/landmark family at the registered
capacity. They do not test a learned content-conditioned tail.

All speed values are arithmetic upper bounds. No H200 latency claim is made.
