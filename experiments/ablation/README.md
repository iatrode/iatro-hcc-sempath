# Reduced-Scale Ablation

This experiment contains the public A0-A4 configuration matrix and the compact
evidence derived from the completed reduced-scale runs.

Conditions:

- A0: full multi-teacher PAMT-D
- A1: multi-teacher without prototype supervision
- A2: multi-teacher without prototype-adjudicated filtering
- A3: single teacher without prototype supervision
- A4: single teacher with prototype supervision and adjudication

Primary comparisons:

- A0 vs A4: multi-teacher contribution
- A0 vs A2: adjudicated-filter contribution
- A0 vs A1: complete prototype-system contribution
- A3 vs A4: prototype contribution under one teacher

The tracked `configs/`, `tables/`, and `reports/` are publication-facing.
Completed run directories and checkpoints are stored locally under
`artifacts/experiments/ablation/`.

The example base configuration contains placeholder data and prototype paths.
Local or server overrides must provide those paths before execution.

```bash
bash experiments/ablation/scripts/run_ablations.sh
bash experiments/ablation/scripts/run_ablations.sh a0 a2 a4
```
