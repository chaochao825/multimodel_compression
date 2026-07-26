# Function-Aware Entropy Audit v1

This directory contains the formal Wan2.1-T2V-1.3B probes used by
`ENTROPY_STRUCTURE_AUDIT_20260726.zh-CN.md`.

Regenerate the dashboard from the repository root with:

```bash
python worldfoundry_hybrid_residual/scripts/plot_entropy_structure_audit.py \
  --raw-dir worldfoundry_hybrid_residual/results/entropy_structure_audit_v1/raw \
  --output-dir worldfoundry_hybrid_residual/results/entropy_structure_audit_v1/figures
```

The large sampled activation tensor is intentionally excluded. Published
artifacts are limited to source code, manifests, CSVs, logs, PNG, PDF, and the
analysis report.
