# Action-DiT FlowQuotient probe

This package contains the first action-generating Diffusion Transformer probe
in the structured-state research program. It evaluates selective W4 FFN fake
quantization and action-horizon mean, affine, circular, non-periodic Toeplitz,
low-rank, and hybrid corrections on three independently trained PushT
checkpoints.

The fixed replication result is mixed: train-0 and train-2 are `BOUNDARY`,
while train-1 is `GO`. Local action-axis structure transfers, but the post-hoc
low-rank coefficient predictor and complete-sampling stability do not. This is
not an environment-success or integer-kernel speed result.

Start with `reports/ACTION_DIT_PI_WAM_STRUCTURED_ANALYSIS_20260826.zh-CN.md`.
The report links the experiment to QuantVLA, QVLA, LightDP, pi0-class action
flow models, and world action models, and develops the proposed Action-Flow
Quotient Error Shaping objective.

## Reproduction

Run `scripts/probe_action_dit_structured_correction.py` with a compatible
Diffusion Policy checkpoint and dataset. Scientific settings and stop rules are
frozen in the two files under `protocols/`. Unit tests cover schedule bucketing,
non-periodic boundaries, local-plus-global defect recovery, and bounded
low-rank payload.

Raw CSV/JSON outputs for all three checkpoints are under `results/`. The
publication plots and their bound CSV data are under `figures/`.
