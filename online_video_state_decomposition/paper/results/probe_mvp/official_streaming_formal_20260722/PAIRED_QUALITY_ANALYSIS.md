# Paired Official Quality Analysis

OASIS scored 209/250 (83.6%) and CausalMem scored 206/250 (82.4%). The difference is +1.2 percentage points, or +3 questions.

The paired outcomes are 187 both correct, 22 OASIS-only correct, 19 CausalMem-only correct, and 22 both wrong. The exact McNemar p-value is 0.755, so the overall difference is not statistically distinguishable in this 250-question run.

Among task groups with at least 10 questions, the largest positive difference is Action Perception (+15.7 points; uncorrected p=0.021, Bonferroni p=0.215). The largest negative difference is Object Perception (-5.9 points; uncorrected p=0.263).

This is a benchmark-system comparison, not a controlled memory-module ablation: OASIS uses Qwen3-VL-8B-Instruct while CausalMem uses LLaVA-OneVision-Qwen2-7B. The result therefore does not establish that either memory mechanism is superior. A defensible method claim requires a shared backbone, identical frame sampling, and matched memory/token budgets.
