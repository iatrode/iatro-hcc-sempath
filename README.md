# HCC-SemPath

HCC-SemPath learns an HCC-specific pathology representation from four cached
pathology teachers and a small, fixed expert-supervision stream. The active V2
task is:

- four-class global L1 tissue-state classification;
- nine-component L2 spatial morphometry from class-routed point, circle, and
  brush annotations.

L2 is not a tile-level multi-label classifier. It exposes only measurements
supported by each component's annotation semantics: instance count, local
density, occupied area, or bile-pigment focus density. Unsupported
measurements are invalid rather than zero.

The scientific and implementation source of truth is
[`docs/HCC_SEMPATH_V2_DESIGN.md`](docs/HCC_SEMPATH_V2_DESIGN.md).

## Environment

This repository uses only the `hcc-camoe` conda environment:

```bash
conda activate hcc-camoe
python -m pip install --no-deps -e .
hcc-sempath --help
```

## Data flow

```text
HCC image-tile IAC packages + four teacher-feature IAC streams
  -> shared DINOv2-S/14 HCC representation
  -> L1 four-class global readout
  -> L2 nine-component spatial instance/measurement maps
  -> independently calibrated count/density/area outputs
```

IatroCache (`.iac`) is an offline cache implementation detail. Private WSIs,
tile packages, teacher features, checkpoints, patient-identifiable manifests,
and local paths do not belong in this public repository.

## Main commands

Contract smoke test:

```bash
bash scripts/run_contract_smoke.sh
```

Train or resume:

```bash
hcc-sempath train --config configs/server/train_full.example.yaml

hcc-sempath train \
  --config configs/server/train_full.example.yaml \
  --resume outputs/hcc_sempath_v2/checkpoints/last.pt
```

Evaluate retained teacher alignment and L1:

```bash
hcc-sempath evaluate \
  --config configs/server/train_full.example.yaml \
  --checkpoint outputs/hcc_sempath_v2/checkpoints/last.pt \
  --split val
```

The terminal checkpoint is the spatial candidate. Its decoder is frozen only
on an independent slide-separated asset whose tile/component records explicitly
declare `roi_count_complete` and `roi_measurement_complete`:

```bash
python scripts/calibrate_spatial_decoder.py \
  --checkpoint outputs/hcc_sempath_v2/checkpoints/last.pt \
  --annotation /path/to/hcc_l2_spatial_validation.json \
  --validation-split val \
  --output-calibration outputs/hcc_sempath_v2/spatial_calibration.json \
  --output-report outputs/hcc_sempath_v2/spatial_validation_report.json

python scripts/export_release_sempath.py \
  --checkpoint outputs/hcc_sempath_v2/checkpoints/last.pt \
  --spatial-calibration outputs/hcc_sempath_v2/spatial_calibration.json \
  --output-dir artifacts/release/hcc_sempath_v2
```

Calibration and release verify the frozen optimizer-visible cohort,
supervision-asset digests, terminal model state, research contract, validation
annotation, protocol, and cohort. Ordinary weak training marks never imply
exhaustive validation truth.

## Maintained documentation

- [`docs/HCC_SEMPATH_V2_DESIGN.md`](docs/HCC_SEMPATH_V2_DESIGN.md): active
  scientific, supervision, validation, and release contract.
- [`TODO.md`](TODO.md): unfinished empirical and release gates only.
- [`annotations/README.md`](annotations/README.md): annotation workspace,
  state files, and information-curve operations.
- [`artifacts/README.md`](artifacts/README.md): local/private artifact boundary.
- [`experiments/README.md`](experiments/README.md): V1/V2 evidence boundary and
  experiment sequence.
