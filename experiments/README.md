# Experiments

This directory contains the active HCC-SemPath study protocols. Generated
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
5. run the matched 10%-population, three-epoch mechanism study under
   [`ablation/`](ablation/README.md).

The first Results subsection reports corpus flow, the fixed 2,400-tile classification
prototype bank, final spatial tile/slide/geometry counts, and the prespecified
information-saturation decision in all four teacher spaces. Model performance
is not used to decide whether annotation is sufficient.

The terminal model is fixed by schedule rather than test performance. classification and
spatial comparisons use paired slide-level bootstrap intervals. The terminal model
is trained once at full duration. The selected three-epoch Optuna trial is the
A0 reference for one fixed-seed 10% mechanism matrix; every other condition
inherits its subset and hyperparameters without retuning. Decoder calibration
and locked spatial testing use different slides, and neither cohort may be
optimizer-visible.
