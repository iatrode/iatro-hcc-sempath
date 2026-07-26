# HCC-SemPath

HCC-SemPath learns an HCC-specific pathology representation from four cached
pathology teachers and a small, fixed expert-supervision stream. The model
combines:

- four-class global L1 tissue-state classification;
- nine-component L2 spatial morphometry from class-routed point, circle, and
  brush annotations.

The L2 branch maps nine components into annotation-grounded instance count,
local density, occupied area, and bile-pigment focus-density measurements.
Each component exposes the measurements defined by its biological and
annotation semantics.

The scientific and implementation source of truth is
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md).

This repository contains research code and reproducibility protocols. Private
pathology data, teacher caches, supervision records, and model artifacts are
managed outside the source distribution.

## Installation

HCC-SemPath supports Python 3.10 and later. Create an isolated environment and
install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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

IatroCache (`.iac`) provides the offline tile and teacher-feature streams.
Deployment supplies its own WSIs, packages, checkpoints, and manifests.

## Main commands

Run the unit and contract suite:

```bash
python -m pytest -q
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

## Annotation workspace

The local `annotations/` directory stores review state and may contain local
paths or case identifiers, so the complete directory is excluded from Git. The
active files are:

- `hcc_prototype_review.final_3000_inflammatory_stromal.json`: stable L1
  supervision;
- `hcc_shared_priority_tiles.json`: shared L1/L2 priority boundary;
- `hcc_l2_roi_v2.json`: nine-component point/circle/brush L2 state.

Seed the shared boundary once:

```bash
hcc-sempath build-priority-list \
  --annotations annotations/hcc_prototype_review.final_3000_inflammatory_stromal.json \
  --output annotations/hcc_shared_priority_tiles.json
```

Start the combined annotation service:

```bash
hcc-sempath annotate-prototypes \
  --input /path/to/image_tile_iac_root \
  --l1-state annotations/hcc_prototype_review.final_3000_inflammatory_stromal.json \
  --l2-state annotations/hcc_l2_roi_v2.json \
  --priority-manifest annotations/hcc_shared_priority_tiles.json \
  --roi-candidate-manifest annotations/hcc_l2_roi_v2_candidates.json
```

The interface exposes separate L1 classification and L2 ROI workspaces over a
shared priority boundary. Existing component-presence labels prioritize the L2
queue, and the four-teacher information report increases priority for spatial
components that still need coverage.

The command-line state is the `Main` version. UI-created versions are stored
under `<state-stem>.versions/<version-id>.json`, with
`<state-stem>.versions.json` as their index. Each version owns its marks, skips,
progress, labels, JSON, and CSV exports. Stable label IDs survive display-name
changes; CSV exports include both IDs and display names.

Random navigation filters tiles below 30% estimated tissue by default. The
`--min-tissue-fraction` option changes this threshold, and `0` disables tissue
filtering. L2 navigation first consumes component-presence candidates, ordered
by the current information-curve deficit. The interface reports exhausted
candidate inventories that still require additional coverage.

The fixed L2 classes are:

1. `hepatocellular-parenchyma-present`
2. `necrosis-present`
3. `hemorrhage-present`
4. `bile-pigment-present`
5. `inflammatory-cell-present`
6. `fibrous-stroma-present`
7. `steatosis-vacuolation-present`
8. `vascular-structure-present`
9. `ductular-portal-present`

Point, circle, and brush semantics are defined in
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md). **Find similar
marks** compares same-tile H&E patches around manual point/circle seeds;
accepted preview crosses become editable points. Candidate spacing is estimated
per class and respects existing point/circle centres.

Independent decoder calibration uses a slide/patient-separated asset with
explicit per-component completeness:

```json
{
  "roi_count_complete": ["inflammatory-cell-present"],
  "roi_measurement_complete": [
    "inflammatory-cell-present",
    "fibrous-stroma-present"
  ]
}
```

Each completeness field accepts a component-to-boolean map or `true` for all
components. These fields provide the exhaustive endpoint truth used by
calibration.

Run the annotation stopping audit from the repository root:

```bash
python scripts/check_annotation_information_curves.py
```

The audit recomputes L1 and per-component L2 curves in all four frozen teacher
spaces and writes aggregate reports under
`artifacts/diagnostics/annotation_information_curve_current/`. Annotation
stops after repeated low-information-gain tails and geometry/slide QC pass in
every teacher space. The fixed-probe curve supplies the stopping decision;
new-batch novelty remains a secondary discovery diagnostic.

## Maintained documentation

- [`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md): active
  scientific, supervision, validation, and release contract.
- [`TODO.md`](TODO.md): unfinished empirical and release gates only.
- [`experiments/README.md`](experiments/README.md): tracked experiment
  protocols and generated-output boundary.
