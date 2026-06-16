# Matched Reduced-Scale Ablation

This experiment contains the completed mechanism study. All conditions use the
same fixed 10%
training/validation subset, 20-epoch schedule,
optimizer, seed, and evaluation protocol. Only the stated method component
changes between conditions.

Conditions:

- A0: dynamic student prototypes with half-strength filtering
- A1: multi-teacher without prototype supervision
- A2: multi-teacher without prototype-adjudicated filtering
- A3: single teacher without prototype supervision
- A4: single teacher with prototype supervision and adjudication
- A5: A0 with static student prototypes
- A0': A0 with complete filter strength
  (`prototype_filter_weight=1.0`)

Primary comparisons:

- A1 vs A3: multi-teacher contribution without the prototype system
- A2 vs A1: prototype-supervision contribution without filtering
- A4 vs A3: independent prototype replication under one teacher
- A0' vs A2: complete-filter contribution
- A0 vs A5: dynamic student-prototype refresh under matched filter strength

The orthogonal checks are positive on the 1,000 expert-reviewed tiles:

- A1 minus A3: `+0.144` L1 accuracy (95% CI `0.116` to `0.174`)
- A2 minus A1: `+0.038` L1 accuracy (95% CI `0.019` to `0.057`)
- A4 minus A3: `+0.043` L1 accuracy (95% CI `0.025` to `0.061`)
- A0 minus A5: `+0.090` L1 accuracy (95% CI `0.064` to `0.116`)

These comparisons isolate multi-teacher learning, prototype supervision, and
dynamic prototype refresh without requiring a single condition to serve as the
baseline for every mechanism. A0 is the matched reference only for A0 vs A5;
A0' is the matched full-filter variant for the direct A0' vs A2 filtering
contrast.

On top500, A0' minus A2 increases the mean expert-class margin by `0.00093`
(95% CI `0.00025` to `0.00158`) and improves Brier score by `0.00299`
(95% CI `0.00077` to `0.00508`). On random500, the margin decreases by
`0.00172` (95% CI `0.00120` to `0.00223`). Filtering is therefore interpreted
as conflict-directed reweighting.

This matched reduced-scale design is the manuscript ablation. Replicating A1-A4
at the full training scale would require roughly a month of additional training
and is not required to isolate component effects when all ablation conditions
are compared at the same scale.

The tracked `configs/`, `tables/`, and `reports/` describe the completed runs.
Run directories and checkpoints are stored locally under
`artifacts/experiments/ablation/`.

Expert-reviewed paired comparisons and manuscript figures are stored under
`../10_teacher_disagreement_review/tables/paired_ablation_l1_comparisons.csv`
and `../10_teacher_disagreement_review/reports/figures/`.

The example base configuration contains placeholder data and prototype paths.
Local or server overrides must provide those paths before execution.

```bash
bash experiments/ablation/scripts/run_ablations.sh
bash experiments/ablation/scripts/run_ablations.sh a0 a2 a4
bash experiments/ablation/scripts/run_ablations.sh a6
```
