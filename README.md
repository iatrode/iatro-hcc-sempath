# HCC-SemPath

[![Source](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/iatrode/iatro-hcc-sempath) [![Hugging Face](<https://img.shields.io/badge/Hugging%20Face-gated%20model-ffcc4d?logo=huggingface&logoColor=black>)](https://huggingface.co/iatrode/iatro-hcc-sempath) [![ModelScope](<https://img.shields.io/badge/ModelScope-gated%20model-624aff>)](https://modelscope.cn/models/iatrode/iatro-hcc-sempath) [![PyPI](https://img.shields.io/pypi/v/hcc-sempath?include_prereleases)](https://pypi.org/project/hcc-sempath/) [![Python](https://img.shields.io/pypi/pyversions/hcc-sempath)](https://pypi.org/project/hcc-sempath/) [![CI](https://github.com/iatrode/iatro-hcc-sempath/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/iatrode/iatro-hcc-sempath/actions/workflows/ci.yml)

[English](README.md) | [简体中文](README.zh-CN.md)

**Gated model weights:** [Hugging Face](https://huggingface.co/iatrode/iatro-hcc-sempath) | [ModelScope](https://modelscope.cn/models/iatrode/iatro-hcc-sempath).
Both hubs distribute the same SemPath model artifact. The PyPI package contains
the modelling code and CLI; clinical assets, teacher weights, and patient-level
outputs are not distributed.

HCC-SemPath is an HCC-specific pathology representation model. It distils four
frozen pathology foundation models into one DINOv2-S/14 student, then anchors
that representation with a small, fixed pathologist-supervised prototype bank.
The released package is a modelling toolkit: it builds tile and teacher-feature
assets, supports classification and spatial annotation, constructs fixed
supervision banks, trains and evaluates the student, and exports recoverable
tile-level predictions.

The model has two parallel supervised outputs:

- a seven-class HCC tissue/classification readout;
- an eleven-component spatial readout grounded by point, circle, and brush
  annotations.

The scientific and implementation source of truth is
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md). The tracked paper
experiments and their public/private boundary are documented in
[`experiments/README.md`](experiments/README.md).

> **Research use only.** HCC-SemPath is not a diagnostic device and must not be
> used for clinical decision-making without independent validation and the
> required institutional approval.

## Contents

- [Scientific contract](#scientific-contract)
- [Model contract](#model-contract)
- [Release and data boundary](#release-and-data-boundary)
- [Installation](#installation)
- [Command surface](#command-surface)
- [Quickstart: released-model inference](#quickstart-released-model-inference)
- [Complete modelling workflow](#complete-modelling-workflow)
- [Annotation semantics](#annotation-semantics)
- [Training and checkpoint selection](#training-and-checkpoint-selection)
- [Prediction schema and spatial reconstruction](#prediction-schema-and-spatial-reconstruction)
- [Configuration](#configuration)
- [Testing](#testing)
- [Repository layout](#repository-layout)
- [License](#license)

## Scientific contract

HCC-SemPath tests one central hypothesis: a compact, HCC-specific student can
retain the complementary morphology encoded by several large pathology
teachers when teacher distillation is coupled to a small expert bank that
covers the relevant classification and spatial feature spaces.

The contract has five fixed parts.

1. **Four frozen teacher coordinates.** GigaPath, H-optimus-1, UNI2-h, and
   Virchow2 are cached offline. Teacher weights and teacher feature extractors
   are never optimized during SemPath training.
2. **One shared student.** DINOv2-S/14 produces the common HCC representation.
   At approximately 20x and 0.5 micrometres per pixel, one 14-pixel patch spans
   about 7 micrometres, placing the student token scale near an immune-cell
   diameter.
3. **Two independent supervision axes.** Classification prototypes describe
   global HCC/tissue identity. Spatial prototypes describe local biological
   components. Neither axis is a surrogate for the other.
4. **Fixed complete-bank prototypes.** PAMT-D prototypes are exact current-
   student centroids over the complete frozen expert banks. They are not
   minibatch exponential-moving averages.
5. **Independent checkpoint supervision.** The validation expert bank is used
   to select a checkpoint. It is not added to the training optimizer and is not
   teacher-distillation training data.

The training classification bank establishes that the small expert-labelled
set has reached the required feature-space coverage. The validation bank gives
the otherwise self-refreshing distillation trajectory an external supervised
stopping signal. Private study data are therefore not interchangeable with the
public code paths that operate on them.

## Model contract

### Classification classes

The global readout uses seven mutually exclusive classes:

1. well-differentiated HCC;
2. moderately differentiated HCC;
3. poorly differentiated HCC;
4. background liver;
5. inflammatory stroma;
6. hemorrhage/necrosis;
7. artifact/contamination.

### Spatial components

The spatial readout uses eleven non-exclusive components:

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

Fibroblast is a cell-localization/density target. Fibrous stroma is a
continuous extracellular-area target. Small and large vessels remain separate
because the latter require multi-grid structural context.

### Training phases

Training begins with teacher-only representation alignment. Classification and
spatial supervision then enter together through the configured step ramp. The
implementation does not silently enable validation supervision as an optimizer
loss. Checkpoint selection consumes validation readouts after the forward pass.

## Release and data boundary

The repository intentionally separates reusable modelling code from controlled
study assets.

| Asset | Public source repository | Supplied by the operator |
|---|---:|---:|
| CLI, model, losses, training engine, prediction reader | Yes | No |
| Example configuration and schema contracts | Yes | No |
| Annotation UI and supervision builders | Yes | No |
| Paper-specific protocol scripts/configurations | Yes, under `experiments/` | No |
| Diagnostic WSIs and derived image tiles | No | Yes |
| Patient/case identifiers and annotation states | No | Yes |
| Four-teacher weights and gated access grants | No | Yes |
| Teacher-feature IAC caches | No | Yes |
| Frozen train/validation supervision records | No | Yes |
| Student checkpoints and release bundle | Separate gated distribution | Yes |

Annotation states can contain local paths or case identifiers. Keep them in an
ignored local workspace or an institutionally controlled private repository.
The package never downloads or redistributes the four teacher models; access
must be obtained from each original provider under its own terms.

## Installation

HCC-SemPath supports Python 3.10 and later. Install the latest public preview
from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install --pre hcc-sempath
hcc-sempath --help
```

For development from a source checkout, install the editable package and its
complete developer toolchain:

```bash
python -m pip install -e ".[dev]"
```

The two IatroCache dependencies are constrained to their compatible public
release series in `pyproject.toml`. Install the resolved project dependencies
together rather than independently overriding either package.

For WSI ingestion, OpenSlide must be usable in the selected Python environment.
GPU training additionally requires a PyTorch/CUDA combination appropriate for
the host. HCC-SemPath does not create or select a CUDA environment on behalf of
the operator.

## Command surface

The installed CLI exposes eight stable top-level workflows:

```text
hcc-sempath build       Build reusable tiles, features, manifests, and supervision
hcc-sempath download    Download the gated release into the local model cache
hcc-sempath annotate    Run the classification/spatial annotation workspace
hcc-sempath train       Train or resume a SemPath student
hcc-sempath evaluate    Evaluate a checkpoint against a resolved configuration
hcc-sempath export      Create the inference-only release bundle
hcc-sempath infer       Export recoverable tile-level predictions
hcc-sempath benchmark   Benchmark the released inference contract
```

The build namespace contains:

```text
hcc-sempath build tiles
hcc-sempath build teacher-features
hcc-sempath build training-cache
hcc-sempath build manifest
hcc-sempath build supervision
```

Every command provides its authoritative options through `--help`:

```bash
hcc-sempath build tiles --help
hcc-sempath build teacher-features --help
hcc-sempath build manifest --help
hcc-sempath annotate --help
hcc-sempath train --help
hcc-sempath infer --help
```

## Quickstart: released-model inference

After gated access is approved, download the release once. The command selects
ModelScope for a China public IP and Hugging Face otherwise; `--hub` can make
that choice explicit.

```bash
hcc-sempath download

hcc-sempath infer \
  --input /path/to/case.svs \
  --output /path/to/predictions
```

`infer` resolves the downloaded release from the local model cache. A release
stored elsewhere can be selected with `--model`; `--hub {hf,modelscope}` and
`--cache-dir` select a particular local cache.

The input may be:

- a `<name>.tile.path.iac` package;
- one 224x224 PNG, JPEG, WebP, or BMP image;
- one WSI or a directory containing `.svs`, `.mrxs`, `.ndpi`, `.scn`, `.tif`,
  or `.tiff` slides.

A WSI is first downsampled to `--target-mpp`, segmented by the low-resolution
tissue mask and per-tile tissue test, and retained as
`<name>.tile.path.iac`. Inference then writes `<name>.pred.path.iac`. Both the
WSI-to-IAC stage and the model stage show progress by default.

Use float16 to retain the model outputs without an additional integer
quantization step, or a bounded integer encoding for smaller files:

```bash
hcc-sempath infer \
  --input /path/to/case.mrxs \
  --output /path/to/predictions \
  --target-mpp 0.5 \
  --min-tissue-fraction 0.10 \
  --spatial-dtype float16 \
  --batch-size 128 \
  --workers 8
```

Use `--native-mpp`/`--native-mpp-y` when trustworthy WSI MPP cannot be read.
Existing tile and prediction outputs are protected unless `--overwrite` is
supplied. `--no-progress` suppresses progress bars.

## Complete modelling workflow

This section describes the full reusable chain. Paths are placeholders; no
controlled study asset is bundled with the repository.

### 1. Build lossless image-tile packages

Input may be a WSI, a directory of WSIs, or an input form accepted by the tile
builder. MPP must be available in slide metadata or supplied explicitly.

```bash
hcc-sempath build tiles \
  --input /controlled/wsis \
  --output /assets/tiles \
  --target-mpp 0.5 \
  --tile-size 224 \
  --min-tissue-fraction 0.30 \
  --workers 8 \
  --lossless
```

Important inputs:

- `--patient-id`, `--slide-id`, and `--split` attach stable study identity;
- `--native-mpp`/`--native-mpp-y` resolve slides without trustworthy MPP;
- `--max-tiles` and `--limit` bound deliberate pilot runs;
- `--tcga-patient-id` enables the supported TCGA identifier convention;
- `--overwrite` is required to replace existing packages.

Outputs:

- one `<name>.tile.path.iac` package per processed slide/source;
- `packages.csv`, `batch_summary.json`, and `batch_progress.json` in directory
  mode;
- optional per-slide QC images when `--qc` is enabled.

Acceptance gate:

- package record count matches the retained tile count;
- patient/slide identity and level-0 `x`/`y` coordinates are present;
- MPP, tile size, and source-level geometry are coherent;
- tissue filtering and any QC sample are reviewed before feature extraction.

### 2. Build the merged four-teacher feature package

The public builder runs the fixed GigaPath, H-optimus-1, UNI2-h, and Virchow2
teachers, verifies their row alignment, and writes one merged package per tile
package.

```bash
hcc-sempath build teacher-features \
  --input /assets/tiles \
  --output /assets/features \
  --device cuda \
  --precision bf16 \
  --batch-size 128 \
  --num-workers 8 \
  --validate-output
```

`--pretrained`, `--compile`, precision, feature dtype, and device are explicit
operator choices. Single-teacher packages are staging files only. They are
removed after the merged output passes record-count, tile-ID, dimension, byte,
and sampled exact-value checks; interrupted staging is retained for restart.

Outputs:

- one `<name>.feat.path.iac` containing all four teacher vectors per tile;
- `feature_build_manifest.json` describing the merged packages and dimensions.

Acceptance gate:

- the merged package has the same record count and tile-ID order as its tile
  package;
- all four declared teacher dimensions are present;
- the merged bytes reproduce the verified staging vectors exactly.

Virchow2 uses its class token concatenated with the mean patch-token response;
this is part of the frozen feature contract, not a runtime switch.

### 3. Optionally prepare the merged cache for training order

`teacher-features` already emits the merged feature contract. `training-cache`
is only needed when the training run also requires the deterministic row order
preparation used by the study pipeline:

```bash
hcc-sempath build training-cache \
  --tile-root /assets/tiles \
  --feature-root /assets/features \
  --seed 13 \
  --workers 8
```

Packages remain at `<feature-root>/<dataset>/<name>.feat.path.iac`.
The command validates record count, tile-ID order, dimensions, byte size, and
sampled exact feature equality before preparing the paired row order. Use
`--validate-only` for an existing cache.

### 4. Construct patient/slide-separated manifests

```bash
hcc-sempath build manifest \
  --dev-source internal=/assets/tiles/internal \
  --public-source external=/assets/tiles/external \
  --feature-root /assets/features \
  --teacher gigapath \
  --teacher h_optimus_1 \
  --teacher uni2_h \
  --teacher virchow2 \
  --split-key patient_id \
  --val-frac 0.10 \
  --seed 13 \
  --check-artifacts \
  --output /controlled/manifests/hcc_sempath.yaml
```

`--split-key patient_id` is the preferred boundary when patient identity is
available; `slide_id` and `stem` are supported for datasets with different
metadata. Groups remain intact across train/validation assignment.

Acceptance gate:

- no grouping key appears in more than one development split;
- external validation is kept distinct from development data;
- `--check-artifacts` resolves the tile and four-teacher packages for every
  manifest row;
- the resulting YAML summary matches the intended cohort sizes.

### 5. Create controlled annotation states

Launch the combined workspace over the same tile source:

```bash
hcc-sempath annotate \
  --input /assets/tiles \
  --classification-state /controlled/annotations/classification.json \
  --spatial-state /controlled/annotations/spatial.json \
  --host 127.0.0.1 \
  --port 8765
```

Remote access should be provided by an authenticated institutional tunnel or
reverse proxy. `--no-auth` is for a deliberately isolated temporary network,
not a public deployment.

The UI maintains classification and spatial states separately. UI-created
versions live under `<state-stem>.versions/<version-id>.json` with a sibling
version index. Stable label IDs survive display-name changes, and CSV exports
contain both IDs and labels.

For a frozen re-review boundary, provide ordered manifests explicitly:

```bash
hcc-sempath annotate \
  --input /assets/tiles \
  --classification-state /controlled/annotations/classification.json \
  --spatial-state /controlled/annotations/spatial.json \
  --classification-review-manifest /controlled/review/classification.json \
  --spatial-review-manifest /controlled/review/spatial.json \
  --review-existing
```

Each review manifest binds a stable `review_id` and ordered tile/IAC/row
records. Completion is stored by `review_id`; navigation stops at the supplied
boundary rather than silently drawing new tiles.

### 6. Build the fixed classification supervision bank

```bash
hcc-sempath build supervision \
  --annotation-json /controlled/annotations/classification.json \
  --validation-annotation-json /controlled/annotations/classification_val.json \
  --training-manifest /controlled/manifests/hcc_sempath.yaml \
  --source-split train \
  --target-per-class 400 \
  --output-dir /controlled/supervision/classification
```

The builder uses restricted greedy facility coverage in all four teacher spaces
to select a fixed, balanced bank from accepted labels. It writes:

- four `{teacher}_hcc_semantic_prototypes.pt` registries;
- `hcc_prototype_supervision_manifest.csv`;
- `prototype_assets_summary.json`.

The optional validation annotation is schema-checked and kept as an independent
checkpoint-selection asset. It is not merged into the training bank.

Acceptance gate:

- all seven classes are present under the expected stable label IDs;
- each accepted tile resolves uniquely to the training manifest and four
  teacher records;
- train and validation supervision boundaries have no unintended tile overlap;
- the information-gain curve is recomputed only from accepted positive members
  against a fixed eligible reference pool.

Spatial annotations remain geometry-bearing JSON and are consumed through the
training configuration's spatial manifest path; `build supervision` does not
flatten them into the classification prototype registry.

### 7. Audit spatial-bank coverage

The paper protocol includes a tracked audit utility:

```bash
python experiments/scripts/roi_information_curve.py \
  --annotation-json /controlled/annotations/spatial.json \
  --teacher-feature-packages \
    'gigapath=/assets/features/gigapath/*.iac,h_optimus_1=/assets/features/h_optimus_1/*.iac,uni2_h=/assets/features/uni2_h/*.iac,virchow2=/assets/features/virchow2/*.iac' \
  --output-root /controlled/audits/spatial_information_curve
```

The curve measures the marginal feature-space information added by accepted
positive tiles under a fixed reference definition. It must not change when an
unrelated negative or overall candidate count changes. Per-teacher curves are
diagnostic; the prespecified pooled plateau is the stopping readout. Each report
records the annotation SHA-256 and generation time so stale curves can be
detected.

### 8. Resolve and train a student configuration

Copy the templates and replace every placeholder path:

```bash
cp configs/manifest.example.yaml /controlled/configs/manifest.yaml
cp configs/train.example.yaml /controlled/configs/train.yaml

hcc-sempath train --config /controlled/configs/train.yaml
```

Resume from the complete training state when required:

```bash
hcc-sempath train \
  --config /controlled/configs/train.yaml \
  --resume /outputs/hcc_sempath/checkpoints/last.pt
```

A checkpoint retains the resolved configuration, model, optimizer, scheduler,
RNG state, dynamic-prototype refresh position, and terminal epoch contract.
Extending a completed run requires increasing `train.epochs` in the supplied
configuration; resume does not silently invent a new terminal epoch.

The output directory contains:

- `resolved_config.json`;
- `step_metrics.csv`, `development_metrics.csv`, `selection_metrics.csv`, and
  epoch-level `metrics.csv`;
- TensorBoard events;
- `checkpoints/last.pt`, `checkpoints/best.pt`, and configured diagnostic
  checkpoints;
- `summary.json`.

Acceptance gate:

- the resolved manifest and every supervision digest match the intended run;
- the teacher-only and joint ramps begin at their configured global steps;
- training and validation expert rows are disjoint in optimizer use;
- selection probes produce non-empty teacher, classification, and spatial
  readouts;
- `best.pt` is selected by the declared joint score, not by training loss alone.

### 9. Evaluate and calibrate the retained checkpoint

```bash
hcc-sempath evaluate \
  --config /controlled/configs/train.yaml \
  --checkpoint /outputs/hcc_sempath/checkpoints/best.pt \
  --split val
```

Direct teacher-alignment losses are comparable only when cohort, teachers,
feature contract, and normalization are identical. Classification and spatial
validation losses are reported on their fixed supervised banks. Unmarked
spatial regions are unknown and must not be converted into negatives.

If calibrated biological measurements are required, use an independent
slide/patient-separated asset whose records explicitly declare count and
measurement completeness:

```bash
python experiments/scripts/calibrate_spatial_decoder.py \
  --checkpoint /outputs/hcc_sempath/checkpoints/best.pt \
  --annotation /controlled/annotations/spatial_calibration.json \
  --validation-split val \
  --output-calibration /outputs/hcc_sempath/spatial_calibration.json \
  --output-report /outputs/hcc_sempath/spatial_validation_report.json
```

Ordinary weak point/circle/brush marks do not imply exhaustive count truth.
Calibration therefore requires explicit `roi_count_complete` and
`roi_measurement_complete` declarations for the applicable components.

### 10. Export the inference-only bundle

```bash
hcc-sempath export \
  --checkpoint /outputs/hcc_sempath/checkpoints/best.pt \
  --spatial-calibration /outputs/hcc_sempath/spatial_calibration.json \
  --output /outputs/hcc_sempath/release
```

The release directory contains:

- `model.safetensors`;
- `config.json`.

Teacher heads and optimizer state are removed. The shared encoder,
classification head, spatial head, class/component schema, output geometry,
and optional calibration contract remain. Export validates calibration against
the selected model digest and spatial stride before writing the bundle.

## Annotation semantics

Point, circle, and brush are not interchangeable raster tools.

- **Point:** a reliable biological centre. For inflammatory cells, blood cells,
  fibroblasts, and similar cellular targets, it ordinarily marks the cell/nuclear
  centre. A bile-pigment point may represent a very small local focus.
- **Circle:** a centre plus an approximate local extent, suitable for a compact
  object such as a vacuole or vascular lumen.
- **Brush:** an outlined region or dense field. For dense inflammatory cells it
  means that individual centres are impractical to enumerate; for fibrous
  stroma it represents continuous area.

Loss construction preserves these distinctions. Point supervision is not
expanded using the same rule as a brush polygon, and a brush does not imply a
set of repeated point instances.

Unmarked components are **unknown by default**. A component becomes negative
only through an explicit negative annotation. Exhaustive evaluation additionally
requires a completeness declaration. This prevents a partially annotated tile
from penalizing plausible predictions in regions the reader did not review.

The complete geometry and loss semantics are specified in
[`docs/HCC_SEMPATH_DESIGN.md`](docs/HCC_SEMPATH_DESIGN.md).

## Training and checkpoint selection

### Default selection principle

Dynamic complete-bank prototype refresh makes optimizer-visible loss capable of
continuing to decrease after spatial usefulness has saturated. HCC-SemPath
therefore retains both kinds of evidence:

- teacher alignment confirms representation retention;
- fixed validation classification and spatial supervision determines whether
  the model remains useful on expert-defined targets.

The reported A0 protocol uses a prespecified normalized joint score over direct
teacher retention, seven-class validation loss, and eleven-component spatial
validation loss. Terms are normalized to a shared initialization baseline, and
selection/pruning starts only after all supervision ramps are active. The exact
paper-study weights, hashes, search budget, and ablation boundary live under
[`experiments/`](experiments/README.md); they are not implicit defaults for an
unrelated dataset.

### Paper A0 search

Preflight the fixed assets without launching a trial:

```bash
python experiments/scripts/optuna_a0_search.py \
  --base-config experiments/configs/train_a0_optuna.example.yaml \
  --n-trials 0
```

Run the frozen study budget with one coordinator:

```bash
python experiments/scripts/optuna_a0_search.py \
  --base-config experiments/configs/train_a0_optuna.example.yaml \
  --n-trials 10 \
  --study-trials 10
```

Independent trials can occupy separate GPUs while sharing one asynchronous TPE
coordinator:

```bash
python experiments/scripts/optuna_a0_search.py \
  --base-config experiments/configs/train_a0_optuna.example.yaml \
  --n-trials 10 \
  --study-trials 10 \
  --parallel-trials 4 \
  --devices 0,1,2,3
```

The formal study binds source code, resolved IAC packages, initialization,
supervision manifests, and prototype registries by digest. Failed or abandoned
trials are not silently accepted as complete evidence. The exported
`best_config.yaml` records the selected trial, parameters, selected checkpoint,
and relevant digests for downstream ablations.

## Prediction schema and spatial reconstruction

`hcc-sempath infer` writes IatroCache packages with payload schema
`hcc_sempath_tile_predictions` and a versioned header. For every source tile,
the package stores:

- stable package, dataset, split, slide, row, and tile identifiers;
- level-0 tile `x`/`y`, source MPP, and coordinate orientation;
- seven final classification probabilities in float16;
- eleven-component spatial instance-response and abundance-response grids;
- grid dimensions, stride, patch size, padding, and model-pixel-to-level-0 scale;
- model/checkpoint digests and the source index digest.

Spatial grids may be encoded as `float16`, `uint16`, or `uint8`. Integer modes
use the package's declared bounded encoding; they are smaller but are not a
substitute for float16 when irreversible quantization is unacceptable.

Read predictions through the package API rather than treating the payload as an
unversioned NumPy blob:

```python
from hcc_sempath.inference.predictions import (
    PredictionPackageReader,
    grid_cell_center_level0,
)

with PredictionPackageReader("slide.pred.path.iac") as reader:
    prediction = reader.read_at(0)
    index = reader.index_table.slice(0, 1).to_pylist()[0]
    center_x, center_y = grid_cell_center_level0(
        reader.header,
        tile_x=int(index["tile_x"]),
        tile_y=int(index["tile_y"]),
        row=4,
        column=7,
    )
```

The reader validates the IAC container and reconstructs encoded spatial arrays.
`grid_cell_center_level0` applies the stored geometry, so downstream spatial
analysis does not need to guess tile offsets, stride, or level scaling.

## Configuration

Two templates are maintained:

- [`configs/manifest.example.yaml`](configs/manifest.example.yaml): cohort,
  split, tile, and teacher-feature locations;
- [`configs/train.example.yaml`](configs/train.example.yaml): model, loss,
  supervision, optimization, checkpoint, and validation-probe settings.

Copy templates outside the source tree for a controlled run. Do not commit
machine-specific absolute paths, access tokens, patient identifiers, cache
inventories, or private annotation locations.

Configuration resolution is written to `resolved_config.json` at run start.
That file—not an edited template—is the exact runtime record. Resume checks the
stored contract rather than silently accepting an incompatible model or data
layout.

## Testing

Run the complete contract suite in the project environment:

```bash
python -m pytest -q
```

For CLI discovery without constructing assets:

```bash
hcc-sempath --help
hcc-sempath build --help
hcc-sempath infer --help
```

Before a long run, the minimum acceptance sequence is:

1. tile and four-teacher IAC validation passes;
2. manifest group separation passes;
3. classification and spatial supervision digests resolve;
4. one short training probe writes non-empty teacher/classification/spatial
   metrics;
5. a small exported prediction package can be read locally and mapped back to
   its source coordinates.

## Repository layout

```text
src/hcc_sempath/     installed model, data, training, annotation, and I/O code
configs/             public manifest and training templates
docs/                maintained scientific and implementation contract
experiments/         paper-specific protocols, configs, and derived-study tools
tests/               unit, schema, resume, CLI, and integration contracts
```

`experiments/` contains the retained protocol needed to understand the reported
study. It is not the default installed command surface and must not contain
host-specific launch scripts, checkpoints, private annotations, teacher caches,
or transient reports.

## License

Source code and documentation are released under
[CC BY-NC-ND 4.0](LICENSE) for non-commercial use with attribution. Modified
versions may be created for non-commercial use but may not be redistributed.

Third-party dependencies and teacher models remain governed by their original
licenses and access agreements. A SemPath checkpoint does not grant access to,
or redistribute, any gated teacher model. Final student weights, when released,
will be distributed separately through a gated model repository under the
declared release terms. The release contract, intended use, output semantics,
and internal checkpoint readout are documented in [MODEL_CARD.md](MODEL_CARD.md).
