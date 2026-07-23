# Matched Full-Population Reduced-Duration Ablation

The tracked A0-A6 configurations define the planned V2 mechanism study. Every
condition uses the complete population stream, the complete L1/L2 expert union,
the same one-tenth-duration training schedule, seed, and evaluation protocol.
This tests whether the intended gradients descend and the mechanisms separate
under a matched design without introducing a cohort-subsampling confound.
A1/A3 mask L1 labels from the objective, but the same L1 tiles remain in the
replay stream so image distribution and replay frequency stay matched.

The planned contrasts are:

- A1 versus A3: multi-teacher contribution without prototype supervision;
- A2 versus A1: prototype supervision without adjudicated filtering;
- A4 versus A3: prototype contribution in a single-teacher background;
- A6 versus A2: full reliability-filter contribution;
- A0 versus A5: dynamic prototype refresh.

No V2 ablation has been run. Existing tables, reports, and numerical values in
this directory are V1 historical evidence based partly on the removed
tile-level L2 attribute route. They cannot be attributed to the current spatial
model.

The tracked base configuration is an open-source example and contains
placeholder paths. Production runs must supply the resolved local base through
`HCC_SEMPATH_ABLATION_BASE_CONFIG`; the runner overlays only the named
tenth-duration full-population schedule and condition, retains local asset paths, always
executes in the `hcc-camoe` conda environment, and removes its temporary
resolved configs on exit. With no condition arguments it runs A0-A6:

```bash
HCC_SEMPATH_ABLATION_BASE_CONFIG=configs/local/server/train_full.yaml \
  bash experiments/ablation/scripts/run_ablations.sh
HCC_SEMPATH_ABLATION_BASE_CONFIG=configs/local/server/train_full.yaml \
  bash experiments/ablation/scripts/run_ablations.sh a0 a2 a4
HCC_SEMPATH_ABLATION_BASE_CONFIG=configs/local/server/train_full.yaml \
  bash experiments/ablation/scripts/run_ablations.sh a6
```
