# HCC-SemPath

HCC-SemPath learns an HCC-specific pathology representation from four cached
pathology teachers and a small, fixed expert-supervision stream. The model
combines:

- seven-class global classification prototype supervision;
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
  -> seven-class classification prototype readout
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

# To extend a completed run, increase train.epochs in a tracked config and
# resume from its checkpoint.
hcc-sempath train \
  --config configs/server/train_full.example.yaml \
  --resume outputs/hcc_sempath_v2/checkpoints/last.pt
```

Checkpoints retain the resolved configuration, optimizer and scheduler states,
restart-relevant optimizer hyperparameters, the explicit scheduler contract,
RNG state, dynamic-prototype refresh positions, and the absolute terminal
epoch. A resumed run takes its terminal epoch only from `train.epochs` in the
supplied configuration; increasing that value records a continuation while
preserving the optimizer and scheduler state.

Evaluate retained teacher alignment and classification:

```bash
hcc-sempath evaluate \
  --config configs/server/train_full.example.yaml \
  --checkpoint outputs/hcc_sempath_v2/checkpoints/last.pt \
  --split val
```

Run the A0 search only after the fixed train/validation expert assets pass its
preflight:

```bash
python scripts/optuna_a0_search.py \
  --base-config configs/server/train_a0_optuna.example.yaml \
  --n-trials 0
```

Run the formal study with one coordinator; rerunning the same command resumes
the contract-bound study with the same trial-seeded TPE trajectory and without
exceeding its 20-trial global budget:

```bash
python scripts/optuna_a0_search.py \
  --base-config configs/server/train_a0_optuna.example.yaml \
  --n-trials 20 \
  --study-trials 20
```

On a multi-GPU host, keep one coordinator and bind one independent trial to
each GPU. Scheduling is asynchronous: a GPU that finishes early immediately
receives the next trial, while constant-liar TPE accounts for configurations
that remain in flight:

```bash
python scripts/optuna_a0_search.py \
  --base-config configs/server/train_a0_optuna.example.yaml \
  --n-trials 20 \
  --study-trials 20 \
  --parallel-trials 4 \
  --devices 0,1,2,3
```

After one complete preflight, later processes on the same unchanged host can
reuse its frozen-asset result with `--verified-preflight-manifest`. The
coordinator writes a stat-guarded receipt for the training workers; unchanged
IAC packages are not read and hashed again, while any path, size, timestamp,
device, or inode change falls back to full SHA-256 verification.

The A0 checkpoint minimizes a prespecified normalized joint score. Direct
fixed-teacher feature/relation retention receives weight `0.26`; complete-bank
seven-class balanced cross entropy receives `0.28`; and eleven-component
spatial loss receives `0.46`. Every term is divided by the first trial's
shared-initialization epoch-0 value; later trials must reproduce that baseline
before optimization.
Selection and pruning start only after every active supervision ramp. The
study contract hashes source, all resolved tile and four-teacher IAC packages,
model initialization, supervision manifests, and prototype registries.
An interrupted `RUNNING` or failed trial invalidates that formal study instead
of being silently counted as a completed search configuration.
After all 20 records are complete or pruned, `best_config.yaml` binds the
selected trial number and hyperparameters, raw trial-config digest, selected
epoch, and `best.pt` digest. Formal ablations reject a provisional config from
an incomplete study or any config that is not that exported winner.

The finalized `checkpoints/best.pt` selected by that score is the spatial
candidate. Its decoder is frozen only on an independent slide-separated asset
whose tile/component records explicitly
declare `roi_count_complete` and `roi_measurement_complete`:

```bash
python scripts/calibrate_spatial_decoder.py \
  --checkpoint outputs/hcc_sempath_v2/checkpoints/best.pt \
  --annotation /path/to/hcc_spatial_validation.json \
  --validation-split val \
  --output-calibration outputs/hcc_sempath_v2/spatial_calibration.json \
  --output-report outputs/hcc_sempath_v2/spatial_validation_report.json

python scripts/export_release_sempath.py \
  --checkpoint outputs/hcc_sempath_v2/checkpoints/best.pt \
  --spatial-calibration outputs/hcc_sempath_v2/spatial_calibration.json \
  --output-dir artifacts/release/hcc_sempath_v2
```

Calibration and release verify the frozen optimizer-visible cohort,
supervision-asset digests, finalized selected model state, research contract, validation
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
  --classification-state /private/study/classification_state.json \
  --spatial-state /private/study/spatial_state.json \
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
  --classification-state /private/study/classification_review.json \
  --spatial-state /private/study/spatial_state.json \
  --priority-manifest /private/study/shared_priority_tiles.json \
  --roi-candidate-manifest /private/study/spatial_candidates.json \
  --classification-review-manifest /private/study/classification_review_manifest.json \
  --spatial-review-manifest /private/study/spatial_review_manifest.json
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
