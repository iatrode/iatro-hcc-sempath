# Annotation Assets and Operations

This is the single operational reference for HCC-SemPath human annotation.
Annotation payloads can contain local paths or case identifiers and are ignored
by Git. Commit only code, schemas, and public-safe documentation.

Keep completed human review JSON/CSV files here. Generated scores, plots,
memmaps, logs, and selection caches belong under ignored artifact/output
directories.

## Active assets

- `hcc_prototype_review.final_3000_inflammatory_stromal.json`: stable L1
  supervision.
- `hcc_shared_priority_tiles.json`: shared L1/L2 priority boundary seeded from
  the stable L1 asset.
- `hcc_l2_roi_v2.json`: active nine-component point/circle/brush L2 state.
- `reviews/teacher_disagreement/exval_1000/review.{json,csv}`: completed
  historical V1 expert-review asset.

## Unified L1/L2 workspace

Seed the shared boundary once:

```bash
conda run -n hcc-camoe hcc-sempath build-priority-list \
  --annotations annotations/hcc_prototype_review.final_3000_inflammatory_stromal.json \
  --output annotations/hcc_shared_priority_tiles.json
```

Start the annotation UI:

```bash
conda run -n hcc-camoe hcc-sempath annotate-prototypes \
  --input /path/to/image_tile_iac_root \
  --l1-state annotations/hcc_prototype_review.final_3000_inflammatory_stromal.json \
  --l2-state annotations/hcc_l2_roi_v2.json \
  --priority-manifest annotations/hcc_shared_priority_tiles.json \
  --roi-candidate-manifest annotations/hcc_l2_roi_v2_candidates.json
```

Both state arguments and the priority manifest are required. The UI always
exposes separate L1 classification and L2 ROI workspaces. Both exhaust the
same priority boundary before full-corpus sampling; a newly saved
out-of-boundary tile is appended atomically for both modes.

The old tile-level L2 classifier is the primary L2 navigation prior. The
current four-teacher information report weights deficient spatial components,
and reviewed tiles update the observed mapping from old labels to current
spatial positives. Historical labels are never shown as current marks,
pre-filled, or used as spatial targets.

## State and navigation

The command-line state is the `Main` version. UI-created versions start empty
under `<state-stem>.versions/<version-id>.json`; the adjacent
`<state-stem>.versions.json` is the index. Each version has independent marks,
skips, progress, labels, JSON, and CSV.

Stable label IDs survive display-name changes. Referenced labels may be
archived but not deleted. CSV export retains `l1`/`l2` IDs and adds
`l1_name`/`l2_names`.

Random navigation excludes tiles below 30% estimated tissue without recording
a human skip. Change this with `--min-tissue-fraction`; `0` disables the
filter. L2 navigation exhausts the old classification-L2 candidate pool first.
Within that pool it prioritizes components not yet stable in the latest
information report. Direct old-L2 labels for those components are exhausted
before related historical labels are used through their observed spatial
positive yield. The UI reports any component whose direct old-L2 inventory is
fully reviewed while its information curve remains unready. No raw positive
count stops annotation.

## L2 tools

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

`hyaline-change-present` remains only in historical tile-level assets.

Point, circle, and brush meanings are component-specific and are defined in
[`../docs/HCC_SEMPATH_V2_DESIGN.md`](../docs/HCC_SEMPATH_V2_DESIGN.md).

**Find similar marks** uses only same-tile H&E image patches around the selected
class's manual point/circle seeds. It does not use teacher or student features.
Preview crosses are not persisted; **Accept visible matches** converts the
visible subset into editable points. Candidate spacing is estimated per class
and excludes existing manual point/circle centres. **Start from scratch**
discards the preview and clears the current tile.

## Independent validation state

Decoder calibration requires a separate slide/patient-separated asset.
Every tile/component endpoint used as exhaustive truth must declare:

```json
{
  "roi_count_complete": ["inflammatory-cell-present"],
  "roi_measurement_complete": [
    "inflammatory-cell-present",
    "fibrous-stroma-present"
  ]
}
```

Each field may also be a component-to-boolean map or `true` for all components.
These flags mean exhaustive endpoint truth. They are never inferred from
`roi_reviewed`, a positive mark, or an absent mark.

## Information curves

Run the joint stopping audit from the repository root:

```bash
python scripts/check_annotation_information_curves.py
```

The entrypoint has no analysis switches. It reads the current L1/L2 annotation
assets and the local Mac feature manifest, then recomputes the L1
teacher-feature curve and every slide-aware L2 component curve separately in
all four frozen teacher spaces. It writes:

```text
artifacts/diagnostics/annotation_information_curve_current/
  annotation_stop_report.json
  annotation_stop_summary.csv
  l1/
  l2/
```

The primary curve evaluates the same slide-separated probe at every nested
reference count, so remaining novelty can only stay level or fall. The overall
decision is `stop_annotation` only when this tail has repeated low information
gain for the global L1 curve, all four L1 classes, and all nine L2 components;
all teacher-space L1 centres and L2 geometry/slide QC must also pass. Every L2
component must pass independently in every teacher space; averaging cannot hide
one under-covered teacher, and L2 requires three consecutive low-gain
increments. New-batch novelty is retained only as a secondary L1
discovery diagnostic and never changes the stop decision. Both levels use the
same cached teacher features consumed by training; the audit never substitutes
raw RGB or untrained DINO features for the distillation task space.
