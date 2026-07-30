# Formal 1/10 Ablation Matrix

The selected Optuna trial supplies the formal A0 hyperparameters.
A1-A12 reuse its exact 10% population subset, fixed classification/spatial
expert banks, seed, optimizer, learning-rate schedule, intervention schedule,
loss weights, maximum epoch budget, and normalized T/C/S checkpoint-selection
rule. Every intervention computes fresh epoch-0 denominators because its
computation path may change, but preserves the A0 selection weights, ramp
boundary, patience, and relative-delta rule. Population development loss and
teacher-only alignment never select an ablation checkpoint. Each condition
runs once and changes only the mechanism stated below; hyperparameters are not
retuned per condition.

The resolver derives a condition-specific formal contract from A0: it retains
the exact source, student initialization, tile population, validation assets,
and only the teacher/prototype IAC subset active in that condition. The runner
resumes only a checkpoint created with the same planned maximum epoch budget;
an obsolete three-/six-epoch run is rejected rather than extended under a
different cosine schedule. The base must be the final exported
`best_config.yaml`; its selected-trial config and checkpoint identities are
verified before a condition is resolved, and the complete resolved condition
config is rehashed again inside the training process.

| ID | Change from A0 | Primary contrast |
|---|---|---|
| A0 | Full PAMT-D; supplied by the selected Optuna trial | reference |
| A1 | remove the global prototype coordinate and classification supervision; retain matched replay images and spatial | A2–A1: global prototype/classification intervention |
| A2 | disable prototype-adjudicated teacher reliability | A0–A2: adjudication |
| A3 | single Virchow2 teacher, without global prototype/classification supervision | A1–A3: multi-teacher contribution without global prototypes |
| A4 | single Virchow2 teacher with prototype supervision | A0–A4: multi-teacher contribution with prototypes; A4–A3: prototypes in a single-teacher background |
| A5 | compute global student prototypes once and hold them fixed | A0–A5: dynamic global prototypes |
| A6 | compute local spatial prototypes once and hold them fixed | A0–A6: dynamic spatial prototypes |
| A7 | apply full rather than half-strength reliability filtering | filter-strength sensitivity |
| A8 | detach dense spatial-objective gradients from the shared encoder while retaining spatial-head training and spatial reliability evidence | A0–A8: local spatial feedback into `z_HCC` |
| A9 | remove the stride-7 local branch | cell-scale local observation |
| A10 | remove the final-Transformer semantic branch | teacher-shaped semantic context |
| A11 | bypass the dilation 1/2/4 spatial context stack | multi-grid structural context |
| A12 | legacy top-quarter MIL pooling instead of full cell range support | contour-faithful cell brush/circle supervision |

A1 and A3 retain the same classification tile images in the replay population
and the same complete L1 validation bank while zeroing only classification
training loss. This keeps replay composition and joint checkpoint selection
matched. A5 and A6
retain identical module topology and differ only in whether the corresponding
complete-bank student centroids refresh. A9-A11 also retain identical
parameters and output geometry; their named computation path is bypassed.
A12 affects cell range-support bags from brushes and circles; point centres,
structure/continuous area support, and negative targets are unchanged.

Generated configs and outputs remain external. Set the selected A0 trial config
and optionally an output root, then run all remaining conditions:

```bash
HCC_SEMPATH_ABLATION_BASE_CONFIG=/path/to/best_trial/config.yaml \
HCC_SEMPATH_ABLATION_OUTPUT_ROOT=/path/to/formal_ablations \
  bash experiments/ablation/scripts/run_ablations.sh
```

Specific conditions can be named:

```bash
HCC_SEMPATH_ABLATION_BASE_CONFIG=/path/to/best_trial/config.yaml \
  bash experiments/ablation/scripts/run_ablations.sh a2 a8 a9 a10 a11 a12
```
