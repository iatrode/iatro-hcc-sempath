# Local Artifacts

This directory contains large or private research assets that are not committed
to the source repository.

```text
artifacts/
  models/
    hcc-sempath-full/   complete final training run and release export
    teachers/           frozen external teacher weights
  prototypes/           teacher and z_hcc prototype assets
  experiments/
    ablation/           A0-A4 raw runs and checkpoints
    search/             hyperparameter-search runs
    manuscript/         generated outputs from tracked experiments/
  caches/
    local_cache/        reusable local indexes and feature caches
  diagnostics/          local QC figures and exploratory analyses
  smoke/
    real_iac/           retained real-data smoke inputs
```

Tracked reproduction code lives in `experiments/`. Temporary contract-smoke
data and outputs are rebuilt under `outputs/smoke/`.
