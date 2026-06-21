# HCC-SemPath V2 Scientific and Implementation Design

Status: active design of record for the `develop` branch.

This document defines the complete V2 research design. `update.md` is the implementation
ledger, not a substitute for this design. The former V1 foundation is retained only as
`HCC_SEMPATH_DESIGN_V1_DEPRECATED.md` for historical traceability.

## 1. Research Objective

V1 organizes each pathology tile through one global HCC embedding, one mutually exclusive
Level-1 tissue-state axis, and non-exclusive tile-level Level-2 morphology attributes. Its
Level-2 supervision indicates that a morphology is present somewhere in the tile but does not
identify where it appears.

V2 asks:

> Can expert ROI annotations explicitly direct the student toward the local tissue regions that
> instantiate each Level-2 morphology attribute, and can that independently anchored local
> evidence improve the deployable global HCC representation without degrading Level-1 structure,
> morphology retrieval, or retained teacher knowledge?

The primary intervention is ROI-guided local morphology learning. Attribute-wise teacher
adjudication is an optional secondary ablation and is not a prerequisite for V2.

## 2. Core Hypothesis

Level-2 attributes such as necrosis, bile pigment, inflammatory cells, fibrous stroma, or
vascular structures are spatially sparse and may coexist within one tile. Global pooling can
learn contextual shortcuts because the correct tile-level label does not identify the causal
region.

The V2 hypothesis is:

1. expert ROI annotations provide an independent spatial anchor for each Level-2 attribute;
2. patch-level attribute supervision forces local features to respond at the annotated
   morphology rather than at correlated global context;
3. a one-way, stop-gradient local-to-global constraint transfers validated local evidence into
   the reusable global embedding;
4. staged optimization and a bounded loss budget prevent sparse ROI supervision from dominating
   Level-1 organization and multi-teacher distillation.

## 3. Scientific Object and Claims

The released scientific object remains the global normalized embedding `z_hcc`. The ROI branch
is a training-time spatial supervision path and an optional interpretation output. V2 does not
turn HCC-SemPath into a diagnostic detector or a segmentation model.

The spatial map is an attribute-specific patch evidence map. It may be described as
ROI-guided local attention only when `attention` is explicitly defined as this supervised
attribute evidence. V2 does not claim direct supervision of every native Transformer
self-attention matrix.

Primary claim:

> ROI-anchored local morphology evidence reshapes the global HCC representation and improves
> Level-2 morphology organization while preserving global Level-1 and retrieval performance.

Secondary claim:

> The local branch produces spatial evidence aligned with held-out expert ROIs and avoids
> all-zero, all-one, and full-tile broadcast shortcuts.

## 4. Semantic Axes

### 4.1 Level 1

Level 1 remains a mutually exclusive global tissue-state axis. V2 does not introduce local L1
supervision and does not change its taxonomy or loss.

### 4.2 Level 2

Level 2 remains non-exclusive and multi-label. A tile may contain multiple attributes, and each
attribute may occupy only a small subset of patch tokens. Tile-level L2 presence and spatial L2
validity are distinct:

- tile-level expert or teacher evidence can supervise global presence;
- only ROI annotations can supervise spatial location;
- global confidence, teacher consensus, or unlocalized tile labels never become patch targets.

## 5. ROI Annotation Contract

### 5.1 Supported geometry

The annotation contract supports:

- point annotations;
- free brush/polyline annotations;
- circles;
- polygons/freehand closed regions;
- explicit negative geometry where required;
- attribute-wise complete review with no positive region.

Coordinates may be recorded in pixels or normalized tile coordinates. Every record includes at
least tile ID, split, L2 attribute, state, geometry or complete-review status, and annotation
version/identity in the production manifest.

### 5.2 Tri-state token target

For tile `x`, patch token `j`, and attribute `k`:

```text
y_roi[x,j,k] in {0,1}
v_roi[x,j,k] in {0,1}
```

- positive ROI geometry: `y_roi=1`, `v_roi=1`;
- explicit reviewed negative geometry: `y_roi=0`, `v_roi=1`;
- completely reviewed tile: all nine retained ROI attributes initially become valid negatives across the tile,
  then every annotated positive geometry is overlaid for its attribute;
- unreviewed, ambiguous, out-of-tissue, or unmarked regions around partial point/brush
  annotations: `v_roi=0` and contribute no loss.

The production asset uses complete tile review across all nine retained Level-2 attributes. Hyaline
change is removed from the V2 ROI taxonomy because its 35 tile-level positives do not justify a
separate spatial objective. If a tile is
saved as an incomplete draft, point or partial brush annotations never convert unmarked patches
into negatives and that draft is excluded from training.

### 5.3 Coordinate-to-token conversion

Geometry is rasterized at patch centers on the actual backbone grid. At 224 px with patch size
16, the expected grid is `14 x 14`. Training aborts if the configured ROI grid differs from the
backbone grid. Rendering overlays and token conversion are audited before a manifest is frozen.

### 5.4 Training asset and annotation volume

ROI annotation follows the existing Level-1 prototype workflow: one training-side prototype
asset is annotated completely. There is no separately annotated validation or test subset during
training. Model selection does not consume an ROI validation score. After training is frozen,
localization and representation quality are evaluated by independent external sampling.

Every selected tile is reviewed once for all nine ROI Level-2 attributes. All visible positive foci
are marked; an unmarked attribute is therefore a complete negative for that tile. Separate
negative queues are neither needed nor permitted.

Selection is quota-driven by positive attribute coverage, not by nine independent passes over
every tile and not by a fixed total tile count. The production target is 100 ROI-positive tiles
per retained attribute. A deterministic multi-label cover of the existing 3,000 tile-level
annotations yields 402 candidates: it covers at least 100 source-positive tiles for seven
attributes, all 98 ductular/portal candidates, and all 55 vascular candidates. The queue must be
supplemented with at least 2 ductular/portal-positive and 45 vascular-positive tiles before it is
frozen. Supplemental tiles are still reviewed completely for all nine attributes, so one tile can
close multiple class deficits.

Selection prioritizes independent slides and caps repeated tiles from one slide for the same
attribute. Per-attribute feasibility is reported by independent positive slide count. Attributes
without adequate slide diversity remain exploratory even if their tile quota is exhausted.

## 6. Model Architecture

### 6.1 Global path

The existing encoder produces the reusable embedding:

```text
z_hcc = normalize(project(global_pool(ViT(x))))
```

The public `encode()` contract is unchanged. Existing release checkpoints remain loadable because
the ROI branch is instantiated only when an ROI manifest is configured.

### 6.2 Local patch path

The encoder exposes the ViT patch tokens before global pooling:

```text
h_patch[j] = ViT_patch_token_j(x)
u[j]       = normalize(P_patch(h_patch[j]))
```

`P_patch` is separate from the global projector. Each L2 attribute has a normalized query
`q_patch[k]`:

```text
a[j,k] = bounded_cosine(u[j], q_patch[k]) / tau_patch
```

The queries are learnable but are explicitly anchored by valid ROI-token loss. They receive no
global pseudo-spatial target. This is the primary V2 implementation. Positive-token EMA
prototypes from the earlier upgrade plan remain a possible ablation, not part of the primary
method, because adding both mechanisms would make the main claim unnecessarily complex.

### 6.3 Local aggregation

Patch logits are aggregated with Top-Q pooling:

```text
ell_local[k] = mean(top_q_j(a[j,k]))
```

Top-Q allows sparse morphology to influence tile-level local evidence without the instability of
hard max or the dilution of global average pooling. `q`, temperature, activation area, and
degenerate-map rate are frozen before confirmatory evaluation.

### 6.4 Directional local-to-global transfer

For an ROI-supervised tile/attribute pair:

```text
ell_global[k] = bounded_cosine(z_hcc, p_global[k]) / tau_global
L_local_global = BCEWithLogits(
    ell_global[k],
    stopgrad(sigmoid(ell_local[k]))
)
```

The direction is local to global only. Gradients from this consistency term update the global
embedding path but do not turn global predictions into patch supervision. A pair participates
only if it has at least one valid ROI token or a complete spatial review.

## 7. Loss Design and Gradient Budget

### 7.1 ROI token loss

```text
L_roi = sum(v_roi * BCEWithLogits(a, y_roi)) / max(sum(v_roi), 1)
```

Tiles without valid ROI tokens produce exactly zero ROI loss. Normalization is by valid tokens,
not by batch size times all attributes.

### 7.2 Total objective

```text
L_total = L_teacher
        + lambda_relation * L_relation
        + lambda_semantic * L_semantic
        + lambda_global_l2 * L_global_l2
        + lambda_roi * L_roi
        + lambda_lg * L_local_global
```

Initial ROI values are deliberately small:

```text
lambda_roi = 0.10
lambda_lg  = 0.05
```

They are ramps, not immediate constants. Confirmatory runs must log per-objective gradient norms
on the shared backbone. A V2 run is invalid if ROI-related gradient norm persistently dominates
the combined teacher/global objective or materially degrades L1/retrieval.

### 7.3 Attribute-wise teacher adjudication

The implemented `r[m,k]`, soft target `q[k]`, uncertainty gate, negative mask, and asymmetric L2
loss are retained behind `loss.l2_attribute_adjudication: false` by default. They address global
teacher disagreement, not ROI localization.

They are excluded from the primary ROI experiment unless all of the following pass:

- slide-cross-fitted teacher-attribute reliability is available;
- effective global L2 gradient scale is matched to the symmetric baseline;
- the mechanism improves external L2 evidence beyond ROI-V2 alone;
- it does not expand the central paper claim beyond what the data can support.

## 8. Optimization Schedule

### Stage A — existing global representation

Train the teacher-prior/global path under the existing schedule. ROI code may be present but its
weight is zero before `roi_start_step`.

### Stage B0 — ROI head warm-up

- detach ViT patch tokens before the ROI projector;
- train only the patch projector and attribute queries from valid ROI tokens;
- keep local-to-global consistency at zero;
- verify finite loss, non-degenerate activation area, and attribute-specific response variance.

### Stage B1 — joint ROI shaping

- enable ROI gradients into the backbone at `roi_backbone_start_step`;
- ramp `lambda_roi` rather than switching it abruptly;
- start and ramp local-to-global consistency separately;
- retain gradient clipping and monitor objective-specific gradient ratios.

Default implementation schedule:

```text
roi_start_step               0
roi_ramp_steps               1000
roi_backbone_start_step      1000
roi_consistency_start_step   1000
```

Production values may shift relative to the teacher-prior plateau, but the B0-before-B1 ordering
is fixed.

## 9. Validation Design

### 9.1 Controlled models

Use the same training tiles, optimizer budget, seeds, external query/gallery, and fixed training
schedule:

1. V1 global model;
2. V2 ROI local head with detached backbone;
3. V2 ROI joint backbone, without local-to-global consistency;
4. full V2 ROI joint backbone plus local-to-global consistency;
5. optional full V2 plus attribute-wise teacher adjudication.

The primary comparison is 1 versus 4. Model 5 is auxiliary and cannot replace that comparison.

### 9.2 Spatial evidence

- pointing accuracy on an independently sampled external annotation set after training;
- token F1 and soft IoU on externally sampled, completely reviewed regions;
- per-attribute activation-area distribution;
- all-zero, all-one, and full-map broadcast rates;
- external slide-clustered confidence intervals;
- qualitative overlays selected without using model identity or test performance.

### 9.3 Global Level-2 evidence

- external macro and per-attribute average precision;
- macro AUC;
- calibration and coverage;
- performance stratified by ROI size and attribute prevalence.

### 9.4 Primary representation evidence

- blinded morphology retrieval using the existing frozen query/gallery protocol;
- precision@k, mean relevance@k, NDCG@k, and paired model win rate;
- comparison with V1 and all frozen teachers;
- slide-clustered bootstrap confidence intervals.

### 9.5 Non-degradation requirements

- Level-1 accuracy and balanced accuracy;
- teacher-prior retention;
- global embedding stability;
- retrieval performance;
- throughput, memory, and release-interface compatibility;
- per-objective gradient norms and total ROI gradient share.

## 10. Decision Gates

### Gate R0 — training annotation contract

- geometry rendering matches source coordinates;
- tri-state masks preserve ignore regions;
- all selected tiles carry complete nine-attribute review state;
- positive quotas and independent-slide coverage are documented.

### Gate R1 — optimization validity

- B0 is numerically stable;
- training diagnostics remain finite and maps are non-degenerate;
- no ROI validation metric is used for model selection.

### Gate R2 — frozen external evaluation

- after training is frozen, full V2 improves external L2/localization evidence over V1 and
  local-head-only controls;
- L1 and blinded retrieval show no material degradation;
- ROI gradients remain within the frozen budget.

### Gate R3 — manuscript inclusion

- multiple seeds and paired slide-clustered intervals are complete;
- localization and representation claims agree;
- only mechanisms that pass independent ablation enter the manuscript method.

## 11. Expected Contribution

V2 contributes an ROI-guided extension of HCC-SemPath in which independently annotated local
morphology evidence shapes a compact global HCC embedding. The contribution is not merely a heat
map: the scientific test is whether validated local evidence improves the reusable representation
while preserving its global tissue organization and retrieval value.

## 12. Implementation Map and Current Status

Implemented:

- ROI point/brush/circle UI and polygon-capable backend;
- JSON/JSONL/UI-state manifest loading;
- tri-state geometry-to-token conversion;
- patch-token exposure with unchanged `encode()`;
- patch projector, attribute queries, Top-Q local pooling;
- masked ROI loss and directional local-to-global loss;
- B0/B1 detach and ramp schedule;
- ROI diagnostics and configuration validation;
- unit/integration tests for geometry, masks, gradients, scheduling, dataset collation, and scatter.

Pending real-data evidence:

- complete nine-class ROI annotation asset reaching 100 positive tiles per retained attribute;
- annotation rendering audit and per-attribute independent-slide feasibility report;
- gradient-norm instrumentation and budget audit;
- multi-seed V1/V2 ablations;
- post-training external localization and L2 evaluation;
- retrieval/L1 non-degradation analysis;
- manuscript integration after Gate R3.
