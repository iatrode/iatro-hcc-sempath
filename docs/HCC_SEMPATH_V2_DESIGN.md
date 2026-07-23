# HCC-SemPath V2: L1 Classification and L2 Spatial Morphometry

Status: active design of record. This is the single binding scientific,
supervision, implementation, validation, and release contract for V2.

## 1. Research problem

HCC-SemPath V2 learns two different pathology objects from one fixed
DINOv2-S/14 tile encoder:

1. a four-class, mutually exclusive Level-1 tissue-state classification;
2. weakly supervised spatial Level-2 morphometry for nine HCC components,
   supporting component location, instance count, local abundance, and
   calibrated area where the supervision permits it.

Level 2 is not a tile-classification task. The former deployable global L2
scores, attribute-classification loss, Top-Q pooling, and local/global score
consistency are outside the V2 path. Global component centroids exist only as
a no-gradient PAMT-D reliability coordinate during training; they are neither
an L2 output nor a spatial pseudo-label.

## 2. Core hypothesis

A 224-pixel tile contains both cell-scale objects and structures that span
several attention cells. A useful spatial head therefore needs:

- a 14-by-14-pixel local observation window for nuclei and other small objects;
- denser coordinate sampling than the native 16-by-16 DINO token grid;
- semantic context from the final Transformer tokens;
- explicit multi-cell aggregation for fat vacuoles, ducts, vessels, and other
  extended structures;
- component-aware interpretation of point, circle, and brush marks.

The hypothesis is that fusing these signals can preserve the pretrained
teacher representation while sparse expert L1/L2 constraints reshape the
shared `z_hcc` toward scientifically interpretable global and local outputs.

## 3. Fixed semantic contract

### Level 1

The global L1 classes, in fixed order, are:

1. `HCC-tumor`
2. `Background-liver`
3. `Inflammatory-stromal`
4. `Degenerative-material`

The output is one four-way softmax.

### Level 2

The spatial components, in fixed order, are:

1. `hepatocellular-parenchyma-present`
2. `necrosis-present`
3. `hemorrhage-present`
4. `bile-pigment-present`
5. `inflammatory-cell-present`
6. `fibrous-stroma-present`
7. `steatosis-vacuolation-present`
8. `vascular-structure-present`
9. `ductular-portal-present`

Hyaline change belongs only to the deprecated tile-level V1 asset.

## 4. Component-and-geometry annotation contract

Tool meaning is resolved jointly with component biology:

| Components | Point | Circle | Brush | Identifiable measurements |
| --- | --- | --- | --- | --- |
| Hepatocellular parenchyma, hemorrhage, inflammatory cells | One instance | One larger instance | Dense-cell bag with unknown exact count | Instance count plus local density |
| Steatosis/vacuolation, vascular structure, ductular/portal structure | One instance with unresolved extent | One large instance with approximate extent | One connected marked structure with approximate extent; overlapping strokes merge | Structure count plus area |
| Necrosis, fibrous stroma | Invalid for positive annotation | Positive extent | Positive extent | Area/coverage only |
| Bile pigment | Small positive pigment seed, not an instance | Larger positive focus with approximate extent | Irregular/fused positive pigment extent | Pigment burden/area; derived focus density only |

A circle is never treated as a generic brush bag. It is a large point for
countable components and supplies one countable centre; for bile pigment it
supplies focus extent without claiming a biological instance. A brush is
class-routed: dense-cell bag, connected large discrete structure, or
continuous positive area. Brush input events are annotation mechanics:
overlapping strokes on the same discrete structure form one instance.

Explicit negatives are strong negative evidence. Ordinary unmarked space is
usually background but can contain deliberately unresolved mixtures, so it
must not receive the same confidence as an explicit negative.

`roi_reviewed` records that a tile was visited and saved; it is not promoted to
a complete nine-component review. A geometry-free `state=negative,
review_complete=true` record supplies a strong tile-wide negative. Unmentioned
components remain ignored unless an explicit completeness field says
otherwise.

## 5. Model

### Shared encoder and teachers

The student architecture and initialization remain fixed to pretrained
DINOv2-S/14 at native 224-pixel input.
All existing teacher feature caches remain usable. The four teachers shape the
shared representation through feature and relation distillation. The stable
L1 annotations define teacher-specific four-class prototypes and online
student-space class centroids. Their responses provide both HCC semantic
supervision and per-tile PAMT-D teacher reliability, while direct L1
cross-entropy anchors the human boundary.

L2 follows the same small-to-large premise: sparse expert spatial constraints
update local positive/negative component centroids and reshape the local
features and shared encoder while four-teacher distillation anchors `z_hcc`.
Teacher-space component centroids participate only in per-tile reliability
adjudication. Teacher features never generate L2 centres, brush masks, or
spatial pseudo-labels.

### L1 head

The normalized global embedding is read against four no-gradient,
expert-updated student-space centroids:

```text
z = normalize(project(CLS))
l1_logits[k] = cosine(z, centroid_l1[k]) / temperature
```

Human L1 labels update the centroids and use cross entropy on the resulting
response. Semantic distillation and PAMT-D use only the four primary teacher
prototypes; legacy L2 attributes in old prototype packages are ignored.

### Dense local branch

The pretrained 14-by-14 DINO patch projection is reused with stride 7 and
padding 4:

```text
local = conv2d(image, pretrained_patch_projection, kernel=14, stride=7, padding=4)
```

For a 224-pixel input this produces a 32-by-32 local grid without replacing or
invalidating the teacher caches.

Final 16-by-16 Transformer patch tokens are bilinearly projected onto the
32-by-32 grid and fused with the local features. Residual context blocks with
dilations 1, 2, and 4 aggregate evidence across multiple grid cells. This is
the route used for structures whose identity is not visible in one local
window.

### Spatial outputs

For each of nine components the head emits:

- `l2_instance_logits [B, 9, 32, 32]`;
- `l2_abundance_logits [B, 9, 32, 32]`.

These maps are positive-versus-negative local prototype responses over the
fused spatial features. One supervised tile/component contributes one centroid
observation, so a large brush cannot dominate merely by covering more grid
cells. Point/circle centres update instance prototypes. Cell points and dense
cell brushes update density prototypes; circle/brush extent updates discrete
structure area prototypes; continuous and pigment extent updates burden/area
prototypes. A structure point never supplies an inferred extent.

A single prototype is a semantic readout coordinate after the trainable local,
semantic, fusion, and nonlinear context transformations. It is not a claim
that one component has a single raw-image morphology. Distinct appearances,
including heterogeneous hepatocellular nuclei, receive independent spatial
loss at their annotated locations and may map to the same component direction.
The L2 annotation gate likewise measures fixed-probe coverage rather than
distance to one global centre. A multi-prototype morphology bank is not part of
the primary model because it would partition the small expert asset into
unidentified latent subtypes without changing the required component output.

The instance tensor retains fixed nine-class topology, but non-countable
channels are deterministically suppressed and marked invalid in both model
output and decoder metadata; they are not trained as latent pseudo-counts.

Decoded outputs are capability-masked:

- NMS coordinates and counts only for cell-instance and discrete-structure
  components;
- density mass/mean only for the three cell/density components;
- area fraction/pixels only for continuous, pigment, and discrete-structure
  components;
- thresholded bile-pigment focus density as a secondary morphology descriptor,
  not an instance count or direct loss target.

Count and density/area remain separate measurements. They are not summed into a
fabricated global L2 confidence score, and unsupported measurements are emitted
as invalid rather than zero.

## 6. Targets and losses

Countable instance centres use a tolerant one-to-one peak objective: adjacent
clicks cannot be satisfied by the same response cell. Matching maximizes
cardinality before response score, and non-matched cells within the union of
click-tolerance regions are negative for the instance objective, so one click
cannot train two decoded peaks. Dense-cell brushes use an aggregate
positive-bag objective. Area-capable components use positive occupied-area
support. Explicit negatives and ordinary unmarked background remain distinct
confidence sources in both the direct loss and the local prototype readout.

The point tolerance represents click/grid uncertainty, not nucleus diameter.
A point never supplies an inferred object area or an exact sub-grid centre.
Positive support is removed from implicit-negative masks in every geometry
route. Mixed point/brush cell annotations supervise resolved instances and
unresolved abundance separately; they are not converted into contradictory
dense instance truth.

PAMT-D combines teacher base weights and per-tile reliability as
`w_m * alpha_mi`. Feature and L1-semantic losses are normalized jointly over
teacher-by-tile mass; relation loss uses `w_m * alpha_mi * alpha_mj` over
teacher-by-pair mass. The student and teachers enter adjudication through the
same fixed-temperature cosine coordinate, with the temperature applied once.
The shared response KL is further weighted by each tile's normalized teacher
reliability mass. Teacher features, response targets, and reliability weights
are gradient stops.

The total objective is:

```text
L = L_four_teacher(alpha_PAMTD)
  + lambda_response * L_student_prototype_response
  + lambda_l1 * CE(L1)
  + lambda_spatial * (
        L_instance_point_peak
      + lambda_abundance_point * L_abundance_point_peak
      + L_brush_density_bag
      + L_positive_area
      + L_explicit_negative
      + L_implicit_background
    )
```

Each term is normalized per supervised tile/component pair before pairs are
averaged. Cell brushes remain density bags; structure and continuous-component
extent marks supervise the area head. Negative loss reaches the instance head
only for countable components. Ordinary unmarked background cannot dominate
sparse positive or explicit-negative supervision mechanically.

The explicit-negative and implicit-background mechanisms have intentionally
different roles. Explicit negatives receive unit direct-loss weight and define
the prototype boundary whenever available. Unmarked background receives a
0.05 direct-loss weight; its pair-averaged EMA centroid is only the fallback
contrastive coordinate when no explicit-negative centroid exists. The
coefficient of a contrastive coordinate is not a label-confidence weight, so
the 0.05 direct-loss weight must not be reapplied to centroid subtraction and
the fallback centroid must not be removed on the premise that it is strong
per-cell supervision.

## 7. Optimization

The spatial head starts with detached backbone signals, then the spatial
weight ramps and the backbone is released. These counters advance only on
L2-supervised updates:

```text
spatial_start_step           0
spatial_ramp_steps           1000
spatial_backbone_start_step  1000
```

After release, the global and spatial gradient norms, spatial gradient share,
and gradient cosine are measured on the final shared Transformer block. These
are diagnostics, not dynamic loss weights.

The fixed L1/L2 expert-tile union is replayed at a configured interval among
population batches. This keeps both dynamic prototype systems and direct
human supervision active throughout training without changing the full-corpus
four-teacher objective.

Online student prototypes are updated only after the corresponding optimizer
step. Thus a batch cannot first enter a centroid and then use that same
centroid to supervise itself. Explicit-negative centroids define the local
decision boundary when available; weak implicit-background centroids are used
only as a fallback.

## 8. Data organization

- L1 uses the stable 3,000-tile expert classification asset.
- L2 uses the current nine-class spatial annotation manifest.
- Both assets are intentionally small expert interventions on the
  population-scale four-teacher representation; neither is expected to label
  the full corpus.
- The union of their training tiles is replayed throughout population
  distillation; newly annotated tiles enter the same union automatically.
- Reduced-scale ablations subsample only the population stream. They retain the
  complete fixed L1/L2 expert union so supervision coverage is identical across
  matched conditions.
- The shared priority list serves the stable 3,000 tiles first and expands
  outside that boundary only when still-growing component curves require more
  examples.
- L2 has no preset class quota or total tile count. Per-component sufficiency is
  measured separately in each of the four frozen teacher feature spaces on one
  fixed, slide-separated probe while a nested reference set grows. Remaining
  task-space novelty is therefore monotone non-increasing. Annotation stops
  only after every teacher has consecutive low-information-gain tail
  increments and slide/geometry QC passes; a teacher average cannot hide an
  under-covered teacher.
- The L2 curve x-axis is unique component-positive tiles rather than clicks or
  strokes. Point/circle/brush counts, component-specific measurement
  capabilities, slide balance, rasterization failures, and annotation
  conflicts are independent QC. Raw RGB or an untrained DINO representation is
  never used as a substitute for the four-teacher distillation task space.
- The fixed-probe plateau must repeat across slide-aware resamples and
  consecutive tail increments. New-batch discovery novelty may rebound and is
  reported only as a secondary diagnostic, never as a stopping signal.
- The final L2 tile count is the union required for all nine component curves
  to pass. Early-saturating components stop consuming annotation effort;
  still-growing components drive subsequent tile selection.
- Old L2 tile labels may prioritize which image is shown next; they never enter
  the target tensor.
- Training consumes all configured image IAC packages and the four existing
  teacher-feature IAC streams.
- Patient/slide separation remains the split unit.

## 9. Validation

Training records validation loss, L1 accuracy when validation labels exist,
and retained teacher alignment. None of these substitutes for spatial
validation. Consequently, teacher-alignment plateau stopping is disabled for
the spatial route, and the terminal checkpoint after the prescribed schedule
is the pre-validation spatial candidate. The training-side L2 annotation asset
is not reused as a spatial validation set.

After checkpoint freezing, spatial validation is performed on an independent,
slide-separated expert sample:

- instance localization and count metrics only for count-capable components;
- density calibration for hepatocellular, hemorrhagic, and inflammatory-cell
  components;
- area metrics for necrosis, fibrous stroma, bile pigment, vacuolation,
  vascular structures, and ductular/portal structures;
- bile-pigment focus-density repeatability only under a frozen threshold,
  connectivity, minimum-area, and spatial-scale definition;
- results stratified by component mode and annotation geometry;
- per-component and macro results with independent-slide counts.

`scripts/calibrate_spatial_decoder.py` implements this gate. Exact point/count
pairs require explicit per-component `roi_count_complete`; measurement pairs
likewise require `roi_measurement_complete`. These flags are validation-only
claims and are never inferred from ordinary weak training marks. Dense-cell
brushes are evaluated as top-fraction MIL bags, not exact positive pixels.
Threshold scoring excludes incomplete tile/component pairs, retains explicit
and weak implicit negatives only inside complete pairs, and includes
complete-negative tiles when selecting bile minimum-focus size.

Validation tile IDs must resolve inside the requested manifest `val`/`exval`
partition, which must be patient/slide-disjoint from the entire
optimizer-visible population plus L1/L2 expert replay cohort. The terminal
checkpoint freezes the exact optimizer-visible package list, an aggregate
package/cohort digest, and SHA-256 digests of mutable L1/L2/prototype
supervision assets. Resume, independent evaluation, and calibration never
reconstruct that contract from a later directory scan. The aggregate
report contains component/mode, point/circle/brush/mixed strata, slide-macro
metrics, and independent-slide counts. The versioned decoder asset binds the
ordered thresholds, NMS kernels, bile minimum-focus size, and output stride to
the exact terminal model-state digest, research contract, annotation,
protocol, and validation cohort. The release exporter rejects a calibration
from any other checkpoint.

The comparison is against the deprecated global-L2 baseline only as a
historical control. The confirmatory claim concerns spatial measurement, not
tile-level attribute classification.

## 10. Release and downstream boundary

The released pathology tower owns the frozen encoder, normalized `z_hcc`, L1
readout, spatial instance/measurement maps, capability masks, calibrated
measurements, preprocessing metadata, and provenance. Downstream systems may
consume these outputs through a one-way, versioned interface. Their token,
language, routing, or task losses cannot modify the released V2 checkpoint or
be required to reproduce it.

Continuous-token adapters, language alignment, whole-slide aggregation,
patient-level diagnosis, staging, prognosis, and treatment recommendation are
separate research objects. They do not change V2 training or its confirmatory
evidence boundary.

## 11. Expected contribution

V2 contributes one HCC-specific representation with:

- retained multi-teacher foundation features;
- stable four-class global tissue state;
- nine-component, geometry-aware spatial output;
- native support for class-routed point/circle/brush annotations;
- cell-scale localization plus multi-grid structural context;
- count, density, and validated area measurements for downstream spatial
  analysis.

## 12. Implementation and acceptance map

- `modeling/models.py`: dynamic L1/local L2 prototype readouts, dense fused
  spatial features, and decoder.
- `spatial_schema.py`: fixed nine-class measurement capabilities.
- `training/roi.py`: instance centres, density bags, positive area, explicit
  negatives, and weak implicit-background targets.
- `training/spatial_losses.py`: tolerant instance peaks, abundance point peaks,
  dense-cell brush bags, positive-area loss, and capability-routed negatives.
- `training/pamtd.py`: per-tile four-teacher adjudication and shared semantic
  response target.
- `training/engine.py`: active objective and L2-supervised-step warm-up.
- `training/train.py`: L1/L2 ingestion, fixed expert-tile replay, and frozen
  optimizer/supervision contracts.
- `training/spatial_validation.py`: independent completeness-aware calibration
  and aggregate spatial validation.
- `training/evaluate.py`: frozen-contract verification and evaluation-cohort
  exclusion.
- `scripts/roi_information_curve.py`: executable four-teacher,
  component-wise annotation sufficiency/QC curve.
- `scripts/calibrate_spatial_decoder.py`: terminal-checkpoint decoder freeze.
- `scripts/export_release_sempath.py`: provenance-bound V2 release without
  teacher heads.

The implementation is acceptable only while:

1. point, circle, and brush follow the component table above;
2. area-only and pigment components never export biological instance counts;
3. unresolved mixtures are not promoted to strong negative truth;
4. L2 reaches the shared encoder after supervised-step warm-up while
   four-teacher distillation remains active;
5. unsupported measurements remain invalid;
6. annotation sufficiency is determined by component-wise information
   plateaus, not a preset tile quota;
7. reduced-duration ablations retain the full population and complete expert
   union;
8. only an independently calibrated terminal checkpoint can become a release.
