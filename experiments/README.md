# Experiments

This is the single repository-local home for HCC-SemPath manuscript study
protocols and derived experiment utilities. It is not part of the installed
`hcc_sempath` package. `configs/` and `scripts/` contain the A0 search,
calibration, release, and matched one-tenth ablation workflows. Generated
checkpoints, logs, tables, figures, review packages, and cohort records are
written to external experiment storage.

The Results evidence is organized in this fixed order:

1. freeze the population, classification prototype bank, and component-wise spatial annotation
   asset after information-curve and geometry QC;
2. train one terminal full-population model under a prespecified schedule and
   verify that global and spatial gradients reach the shared representation;
3. evaluate classification on the frozen random and teacher-conflict external queues;
4. calibrate the spatial decoder on a slide-separated calibration set and
   evaluate spatial on a distinct locked slide-separated test set;
5. run the matched 10%-population mechanism study defined below.

The first Results subsection reports corpus flow, the fixed 2,800-tile classification
prototype bank, final spatial tile/slide/geometry counts, and the prespecified
information-saturation decision in all four teacher spaces. Model performance
is not used to decide whether annotation is sufficient.

The A0 model is selected only by the frozen teacher-retention plus complete
classification and spatial validation score. Classification and spatial comparisons use paired
slide-level bootstrap intervals. The selected Optuna trial is the A0 reference
for one fixed-seed 10% mechanism matrix; every other condition inherits its
subset and hyperparameters without retuning. Decoder calibration and locked
spatial testing use different slides, and neither cohort may be
optimizer-visible.

## Formal 1/10 ablation matrix

The selected Optuna trial supplies the formal A0 hyperparameters. A1-A11 reuse
its exact 10% population subset, fixed classification/spatial expert banks,
seed, optimizer, learning-rate schedule, intervention schedule, loss weights,
maximum epoch budget, and normalized T/C/S checkpoint-selection rule. Every
intervention computes fresh epoch-0 denominators because its computation path
may change, but preserves the A0 selection weights, ramp boundary, patience,
and relative-delta rule. Each condition changes only the named mechanism;
hyperparameters are not retuned per condition.

| ID | Change from A0 | Primary contrast |
|---|---|---|
| A0 | Full PAMT-D; supplied by the selected Optuna trial | reference |
| A1 | remove the complete global expert intervention; retain matched replay images and spatial supervision | A2–A1: global intervention without reliability adjudication |
| A2 | disable prototype-adjudicated teacher reliability | A0–A2: adjudication |
| A3 | single Virchow2 teacher without the global expert intervention | A1–A3: multi-teacher contribution without global intervention |
| A4 | single Virchow2 teacher with the global expert intervention | A0–A4: multi-teacher contribution with intervention; A4–A3: single-teacher intervention gain |
| A5 | compute global student prototypes once and hold them fixed | A0–A5: dynamic global prototypes |
| A6 | compute local spatial prototypes once and hold them fixed | A0–A6: dynamic spatial prototypes |
| A7 | apply full rather than half-strength reliability filtering | filter-strength sensitivity |
| A8 | remove the stride-7 local branch | cell-scale local observation |
| A9 | remove the final-Transformer semantic branch | teacher-shaped semantic context |
| A10 | bypass the dilation 1/2/4 spatial context stack | multi-grid structural context |
| A11 | remove student prototype-response matching while retaining adjudicated teacher weighting | shared semantic response target |

The normalized teacher/classification/spatial selection score chooses a
checkpoint only within each condition. Cross-condition comparison uses the
directly comparable classification and spatial losses on the same frozen
expert validation banks; external queues are reserved for final-model
evaluation.

```bash
HCC_SEMPATH_ABLATION_BASE_CONFIG=/path/to/best_trial/config.yaml \
HCC_SEMPATH_ABLATION_OUTPUT_ROOT=/path/to/formal_ablations \
  bash experiments/scripts/ablation/run_ablations.sh
```
