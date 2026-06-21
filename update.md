# HCC-SemPath V2 — ROI-Guided Level-2 Upgrade

Status: active implementation ledger for `develop`.

Design of record: [`docs/HCC_SEMPATH_V2_DESIGN.md`](docs/HCC_SEMPATH_V2_DESIGN.md).
Historical baseline: [`docs/HCC_SEMPATH_DESIGN_V1_DEPRECATED.md`](docs/HCC_SEMPATH_DESIGN_V1_DEPRECATED.md).

## Progress Summary

| Workstream | State | Evidence / next gate |
| --- | --- | --- |
| ROI annotation geometry and state schema | implemented | complete nine-attribute tile review; point/brush/circle UI; polygon backend |
| Tri-state geometry-to-token conversion | implemented and unit-tested | production coordinate audit pending |
| Patch-token L2 branch | implemented and unit-tested | real ROI calibration pending |
| Masked ROI token loss | implemented and unit-tested | gradient-budget audit pending |
| Directional local-to-global transfer | implemented and unit-tested | matched V1/V2 experiment pending |
| B0/B1 detach and ramp schedule | implemented and unit-tested | production schedule selection pending |
| Attribute-wise teacher adjudication | implemented, optional, default off | cross-fitting and loss-scale match pending |
| ROI candidate queue and quota scheduler | implemented | 402 existing candidates; supplement vascular +45 and ductular +2, then freeze |
| Training ROI asset | ready for annotation | complete review until every retained attribute reaches 100 ROI-positive tiles |
| Post-training external localization evaluation | not run | Gate R2 after model freeze |
| External L2 and retrieval non-degradation | not run | Gate R2 after model freeze |
| Manuscript integration | not started for V2 | Gate R3 |

Last code validation: 163 passed, 1 skipped; two pre-existing hanging package-shuffle tests were
deselected, while ROI package-scatter behavior was tested separately.

## Objective

V2 uses expert ROI annotations to explicitly supervise where each non-exclusive Level-2
morphology attribute appears. Point, brush, circle, and polygon annotations are converted to
ViT patch-token targets. The spatial evidence is transferred one-way into the deployable global
`z_hcc` Level-2 readout; global predictions and teacher consensus never act as spatial truth.

## Implemented

- ROI annotation UI supports point, brush, and circle input per L2 attribute, undo/clear, and
  attribute-wise completely-reviewed state. The backend also accepts polygon geometry.
- V2 ROI mode excludes hyaline change, enforces all nine review markers on save, derives tile-level
  L2 positives from drawn geometry, supports reopening reviewed tiles, and reports live class quotas.
- `build-roi-queue` creates the deterministic quota-driven queue; `annotate-prototypes
  --roi-candidate-manifest` restricts annotation to that frozen queue.
- Annotation state records tile ID, split, attribute, geometry, state, and review completeness.
- JSON, JSONL, and the annotation UI state JSON are accepted as ROI manifests.
- Geometry is rasterized to the backbone patch grid (`14 x 14` for 224px/16px).
- Token supervision is tri-state:
  - annotated positive geometry -> positive;
  - explicit negative geometry or completely reviewed background -> negative;
  - all unreviewed tokens -> ignore.
- `StudentEncoder` exposes patch tokens without changing the public `encode()` result.
- A separate patch projector and ROI-anchored attribute queries produce per-attribute spatial
  logits and Top-Q pooled local logits.
- `L_roi` is normalized only by valid ROI tokens. Tiles without ROI produce exactly zero ROI loss.
- Directional local-to-global consistency is active only for ROI-supervised tile/attribute pairs;
  the local target is stop-gradient.
- B0/B1 scheduling is implemented: the patch backbone is detached during head warm-up, then ROI
  gradients enter the backbone after `roi_backbone_start_step`; consistency starts separately.
- ROI and consistency losses have independent low weights and ramps, and remain separate from the
  optional attribute-wise teacher adjudication pilot.
- ROI validity, supervised-pair count, and activation fraction are logged.
- Existing release checkpoints remain compatible because the ROI branch is created only when an
  ROI manifest is configured.

## Configuration

```yaml
data:
  roi_manifest_path: /path/to/hcc_l2_roi_annotations.json
  roi_train_splits: [train]

model:
  roi_patch_size: 16
  roi_patch_dim: 1536
  roi_top_q: 0.1
  roi_patch_temperature: 0.1

loss:
  roi_weight: 0.1
  roi_consistency_weight: 0.05
  roi_start_step: 0
  roi_ramp_steps: 1000
  roi_backbone_start_step: 1000
  roi_consistency_start_step: 1000
```

Example manifest record:

```json
{
  "tile_id": "slide_a_0000123",
  "split": "train",
  "attribute": "necrosis-present",
  "state": "positive",
  "review_complete": false,
  "geometry": {
    "type": "brush",
    "coordinate_space": "normalized",
    "points": [[0.31, 0.42], [0.37, 0.46]],
    "width": 0.035
  }
}
```

## Annotation commands

```bash
hcc-sempath build-roi-queue \
  --annotations annotations/hcc_prototype_review.final_3000_inflammatory_stromal.json \
  --annotations /path/to/vascular_ductular_supplement.json \
  --output annotations/hcc_l2_roi_v2_candidates.json \
  --overwrite

hcc-sempath annotate-prototypes \
  --input /path/to/tile-iac-root \
  --state annotations/hcc_l2_roi_v2.json \
  --roi-candidate-manifest annotations/hcc_l2_roi_v2_candidates.json
```

The current generated queue contains 402 candidates and explicitly reports unresolved source
inventory deficits of vascular `45` and ductular/portal `2`. Add supplemental classified positives
and regenerate once; after ROI annotation starts, do not overwrite that queue.

## Acceptance evidence still requiring real data

- annotate the frozen training-side queue completely across all nine ROI attributes until each
  retained attribute has 100 confirmed ROI-positive tiles;
- verify ROI rendering/coordinate mapping and report per-attribute independent-slide coverage;
- compare V1, ROI local-only, and ROI local-to-global under matched fixed training budgets;
- freeze training, then externally sample and report localization, L1/retrieval non-degradation,
  and per-objective gradient norms;
- keep attribute-wise teacher adjudication disabled in the primary ROI experiment unless its
  independent ablation demonstrates added value without changing effective L2 loss scale.
