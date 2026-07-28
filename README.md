# HCC-SemPath

HCC-SemPath learns an HCC-specific pathology representation from four cached
pathology teachers and a small, fixed expert-supervision stream. The model
combines:

- six-class global classification prototype supervision;
- eleven-component spatial prototype supervision from class-routed point, circle, and
  brush annotations.

The spatial branch maps eleven components into annotation-grounded instance count,
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
  -> six-class classification prototype readout
  -> eleven-component spatial prototype and measurement maps
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

Evaluate retained teacher alignment and classification:

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

Annotation states may contain local paths or case identifiers and must remain
outside the public repository. Use an ignored local workspace or a controlled
private study repository.

Seed the shared boundary once:

```bash
hcc-sempath build-priority-list \
  --annotations /private/study/classification_state.json \
  --output /private/study/shared_priority_tiles.json
```

Start the combined annotation service:

```bash
hcc-sempath annotate-prototypes \
  --input /path/to/image_tile_iac_root \
  --l1-state /private/study/classification_state.json \
  --l2-state /private/study/spatial_state.json \
  --priority-manifest /private/study/shared_priority_tiles.json \
  --roi-candidate-manifest /private/study/spatial_candidates.json \
  --roi-information-report /private/study/spatial_information_report.json
```

The interface exposes separate classification and spatial-annotation workspaces over a
shared priority boundary. Existing component candidates prioritize the spatial
queue, and the four-teacher information report increases priority for spatial
components that still need coverage.

An explicit re-review pass can give each workspace its own ordered, read-only
tile boundary:

```bash
hcc-sempath annotate-prototypes \
  --input /path/to/image_tile_iac_root \
  --l1-state /private/study/classification_review.json \
  --l2-state /private/study/spatial_state.json \
  --priority-manifest /private/study/shared_priority_tiles.json \
  --roi-candidate-manifest /private/study/spatial_candidates.json \
  --l1-review-manifest /private/study/classification_review_manifest.json \
  --l2-review-manifest /private/study/spatial_review_manifest.json
```

Each review manifest contains a stable `review_id` and the ordered
`tile_id`/`iac`/`row` records. Review completion is stored by `review_id` in
the corresponding annotation state. Existing marks remain visible during the
pass, skipped review items retain their source annotations, and navigation
stops at the end of the supplied list. Ordinary runs that omit these arguments
retain the mutable shared-priority behavior.

The command-line state is the `Main` version. UI-created versions are stored
under `<state-stem>.versions/<version-id>.json`, with
`<state-stem>.versions.json` as their index. Each version owns its marks, skips,
progress, labels, JSON, and CSV exports. Stable label IDs survive display-name
changes; CSV exports include both IDs and display names.

Random navigation filters tiles below 30% estimated tissue by default. The
`--min-tissue-fraction` option changes this threshold, and `0` disables tissue
filtering. Spatial navigation first consumes existing component candidates, ordered
by the current information-curve deficit. The interface reports exhausted
candidate inventories that still require additional coverage.

The fixed spatial components are:

1. `hepatocellular-parenchyma`
2. `necrosis`
3. `hemorrhage`
4. `bile-pigment`
5. `inflammatory-cell`
6. `fibroblast`
7. `fibrous-stroma`
8. `steatosis-vacuolation`
9. `small-vessel`
10. `large-vessel`
11. `ductular-portal`

Fibroblasts are a cell-localization and density target, distinct from the
continuous extracellular fibrous-stroma area target. Existing vascular ROI
annotations define the small-vessel class; large vessels form a separate
structure class because their identity and extent require multi-grid context.

Point, circle, and brush semantics are defined in
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md). **Find similar
marks** compares same-tile H&E patches around manual point/circle seeds;
accepted preview crosses become editable points. Candidate spacing is estimated
per class and respects existing point/circle centres.

Independent decoder calibration uses a slide/patient-separated asset with
explicit per-component completeness:

```json
{
  "roi_count_complete": ["inflammatory-cell"],
  "roi_measurement_complete": [
    "inflammatory-cell",
    "fibrous-stroma"
  ]
}
```

Each completeness field accepts a component-to-boolean map or `true` for all
components. These fields provide the exhaustive endpoint truth used by
calibration.

Run the reusable spatial stopping audit from the repository root:

```bash
python scripts/roi_information_curve.py \
  --annotation-json /private/study/spatial_state.json \
  --teacher-feature-packages \
    'gigapath=/features/gigapath/*.iac,h_optimus_1=/features/h1/*.iac,uni2_h=/features/uni2/*.iac,virchow2=/features/virchow2/*.iac' \
  --output-root /private/study/spatial_information_curve
```

The audit recomputes per-component curves in all four frozen teacher spaces.
Annotation
stops after the pooled four-teacher-by-resample plateau reaches the same
fixed-probe threshold used by the classification audit and geometry/slide QC
passes. Per-teacher support remains diagnostic; new-batch novelty remains a
secondary discovery diagnostic.
Each spatial report records the source annotation SHA-256 and generation time;
a report is current only when that digest matches the annotation file.

## Maintained documentation

- [`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md): active
  scientific, supervision, validation, and release contract.
- [`experiments/README.md`](experiments/README.md): tracked experiment
  protocols and generated-output boundary.
