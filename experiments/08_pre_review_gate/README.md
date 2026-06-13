# 08 Pre-review Gate

Purpose: run automatic QC and retrieval benchmarks on the sampled exval
`z_hcc` IAC cache before freezing a blinded expert-review package.

Primary output:

- `results/embedding_qc.csv`
- `results/retrieval_<model>_exval.csv`
- `results/retrieval_metrics.csv`
- `results/query_set.csv`
- `results/gallery_set.csv`
- `reports/pre_review_gate.md`
