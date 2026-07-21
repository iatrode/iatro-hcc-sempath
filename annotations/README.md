# Annotation Assets

This directory is reserved for human annotation assets and review outputs.

Do not treat these files as disposable caches. Generated model scores, NPZ
exports, memmaps, logs, and temporary selection caches should stay under
`artifacts/caches/`; completed human review JSON/CSV files should be moved here.

The annotation payloads themselves are intentionally ignored by Git because they
can contain local paths, case identifiers, and large derived tables. Commit only
code, schemas, or documentation unless a dataset release is explicitly planned.

Current local layout:

- `hcc_prototype_review.final_3000_inflammatory_stromal.json`: final prototype annotation state.
- `hcc_shared_priority_tiles.json`: shared mutable L1/L2 tile boundary seeded from the stable 3000-tile L1 review (generated; not committed).
- `hcc_l2_roi_v2.json`: V2 nine-class complete-review ROI annotation state (created by the UI).
- `reviews/teacher_disagreement/exval_1000/review.json`: completed 1000-tile repeated review state.
- `reviews/teacher_disagreement/exval_1000/review.csv`: CSV export of the completed repeated review.

## Unified L1/L2 annotation workspace

Seed the shared boundary from the stable L1 review once:

```bash
conda run -n hcc-camoe hcc-sempath build-priority-list \
  --annotations annotations/hcc_prototype_review.final_3000_inflammatory_stromal.json \
  --output annotations/hcc_shared_priority_tiles.json
```

Run both annotation workflows behind one browser UI and switch with the L1/L2 navigation:

```bash
conda run -n hcc-camoe hcc-sempath annotate-prototypes \
  --input /path/to/image_tile_iac_root \
  --l1-state annotations/hcc_prototype_review.json \
  --l2-state annotations/hcc_l2_roi_v2.json \
  --priority-manifest annotations/hcc_shared_priority_tiles.json
```

Both state arguments and the shared priority manifest are required: the UI always exposes the L1
classification and L2 ROI drawing workspaces. Both modes exhaust the same priority tile boundary
before sampling the full corpus. A newly sampled or manually saved out-of-boundary tile is appended
atomically so the other mode prioritizes it too. The former tile-level L2 classification is not an
annotation task and does not seed labels in the ROI workspace. Label display names and active labels are stored in
each state file. Stable label IDs remain unchanged when a label is renamed;
referenced labels can be archived but cannot be deleted. CSV export retains the original `l1` and
`l2` ID columns and adds `l1_name` and `l2_names` display-name columns.

The `Main` annotation version uses the state path supplied on the command line. New versions are
created from the UI with empty annotations under `<state-stem>.versions/<version-id>.json`; each
version has its own JSON, CSV, skipped records, progress, and label configuration. The adjacent
`<state-stem>.versions.json` file is the version index.

Random navigation estimates tissue coverage from downsampled RGB values and excludes tiles below
30% tissue without recording a human skip. Override the threshold with
`--min-tissue-fraction`, or set it to `0` to disable automatic filtering.

In L2 mode, **Preview suggested plan** runs the configured release classifier once, derives
class-specific spatial evidence by backpropagating through its feature readout, and combines that
evidence with H&E hematoxylin peaks for cell-center proposals. High-contrast outlined preview marks are never written
to the annotation state. **Continue from plan** converts them to ordinary editable ROI marks;
**Start from scratch** discards the preview and clears the current tile. Override the default release
assets with `--roi-plan-config`, `--roi-plan-checkpoint`, and `--roi-plan-device`.



  L1（4 类，互斥）：

1. HCC-tumor
3. Background-liver
4. Inflammatory-stromal
5. Degenerative-material

  L2 ROI（9 类，可并存）：

1. hepatocellular-parenchyma-present
2. necrosis-present
3. hemorrhage-present
4. bile-pigment-present
5. inflammatory-cell-present
6. fibrous-stroma-present
7. steatosis-vacuolation-present
8. vascular-structure-present
9. ductular-portal-present

`hyaline-change-present` is retained only in legacy tile-level L2 assets and is not an ROI task.

L1 expansion:
HCC-tumor further classification

WHO grading system (3 tiered system)
Well differentiated: tumor cells resemble mature hepatocytes; minimal to mild nuclear atypia
Moderately differentiated: tumor cells appear malignant on H&E and morphology suggests hepatocellular differentiation; moderate nuclear atypia
Poorly differentiated: tumor cells appear malignant on H&E and often cannot be distinguished from other poorly differentiated neoplasms; marked nuclear atypia

Modified Edmondson-Steiner grading system (4 tiered system) (Cancer 1954;7:462)
Grade I: tumor cells are difficult to differentiate from hyperplastic liver cells
Grade II: tumor cells resemble mature hepatocytes with slightly larger and more hyperchromatic nuclei; sharp and clear cut cell borders; frequent acini formation
Grade III: tumor cells are larger and have more hyperchromatic nuclei with less acidophilic cytoplasms; trabecular distortion; numerous tumor giant cells
Grade IV: tumor cells are intensely hyperchromatic, with scant and less granular cytoplasm; tumor cells appear less cohesive and can appear giant, spindled or short and plump; medullary growth pattern with loss of trabeculation; less acini

Background-liver
Further expansion
hepatocyte
Portal triad / portal tract
Central vein
