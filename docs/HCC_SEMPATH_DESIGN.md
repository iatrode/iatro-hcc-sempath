# HCC-SemPath: L1 Classification and L2 Spatial Morphometry

Status: active design of record. This is the scientific, supervision,
implementation, validation, and release contract.

## 1. Research problem

HCC-SemPath learns two complementary pathology representations from one fixed
DINOv2-S/14 tile encoder:

1. a four-class, mutually exclusive Level-1 tissue-state classification;
2. weakly supervised spatial Level-2 morphometry for nine HCC components,
   supporting component location, instance count, local abundance, and
   calibrated area where the supervision permits it.

The L2 branch produces spatial component maps and calibrated morphometric
measurements. No-gradient global component centroids provide the PAMT-D
reliability coordinate during training.

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

## 4. Component-and-geometry annotation contract

Tool meaning is resolved jointly with component biology:

| Components | Point | Circle | Brush | Identifiable measurements |
| --- | --- | --- | --- | --- |
| Hepatocellular parenchyma, hemorrhage, inflammatory cells | One instance | One larger instance | Dense-cell bag with unknown exact count | Instance count plus local density |
| Steatosis/vacuolation, vascular structure, ductular/portal structure | One instance with unresolved extent | One large instance with approximate extent | One connected marked structure with approximate extent; overlapping strokes merge | Structure count plus area |
| Necrosis, fibrous stroma | Invalid for positive annotation | Positive extent | Positive extent | Area/coverage only |
| Bile pigment | Small positive pigment seed, not an instance | Larger positive focus with approximate extent | Irregular/fused positive pigment extent | Pigment burden/area; derived focus density only |

A circle supplies one countable centre for countable components and supplies
focus extent for bile pigment. A brush is
class-routed: dense-cell bag, connected large discrete structure, or
continuous positive area. Brush input events are annotation mechanics:
overlapping strokes on the same discrete structure form one instance.

Explicit negatives are strong negative evidence. Ordinary unmarked space is
usually background but can contain deliberately unresolved mixtures, so it
must not receive the same confidence as an explicit negative.

`roi_reviewed` records that a tile was visited and saved. A geometry-free `state=negative,
review_complete=true` record supplies a strong tile-wide negative. Unmentioned
components remain ignored unless an explicit completeness field says
otherwise.

## 5. Model

### Shared encoder and teachers

The student architecture and initialization remain fixed to pretrained
DINOv2-S/14 at native 224-pixel input.
All existing teacher feature caches remain usable. The four teachers shape the
shared representation through feature and relation distillation. The stable
L1 annotations define teacher-specific four-class prototypes and periodically
recomputed student-space class centroids. Every student refresh re-encodes the
complete fixed 3,000-tile bank with the current encoder; a compute mini-batch
is only a memory chunk and never defines the prototype pool. Their responses provide both HCC semantic
supervision and per-tile PAMT-D teacher reliability, while direct L1
cross-entropy supervises the human boundary.

L2 follows the same small-to-large premise: sparse expert spatial constraints
define exact full-bank local positive/negative component centroids and reshape the local
features and shared encoder while four-teacher distillation stabilizes `z_hcc`.
Teacher-space component centroids participate in per-tile reliability
adjudication. Expert point, circle, and brush geometry supplies the L2 spatial
targets.

### L1 head

The normalized global embedding is read against four no-gradient,
expert-updated student-space centroids:

```text
z = normalize(project(CLS))
l1_logits[k] = cosine(z, centroid_l1[k]) / temperature
```

Human L1 labels define the complete centroid bank and use cross entropy on the
resulting response. Centroids are no-gradient coordinates recomputed from the
current student between optimizer steps; gradients from the response update
`z_hcc`. Semantic distillation and PAMT-D use the four primary teacher
prototypes.

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

A component prototype is a semantic readout coordinate after the trainable
local, semantic, fusion, and nonlinear context transformations. Morphological
variants, including heterogeneous hepatocellular nuclei, receive independent
spatial loss at their annotated locations and map into the shared component
coordinate. The L2 annotation gate measures fixed-probe coverage across the
four teacher spaces.

The instance tensor retains fixed nine-class topology. Capability masks
deterministically suppress count outputs for non-countable channels in model
output and decoder metadata.

Decoded outputs are capability-masked:

- NMS coordinates and counts only for cell-instance and discrete-structure
  components;
- density mass/mean only for the three cell/density components;
- area fraction/pixels only for continuous, pigment, and discrete-structure
  components;
- thresholded bile-pigment focus density as a secondary morphology descriptor.

Count, density, and area remain separate measurements. Capability masks mark
undefined component/measurement pairs as invalid.

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

Explicit negatives receive unit direct-loss weight and define the prototype
boundary whenever available. Unmarked background receives a 0.05 direct-loss
weight; its exact full-bank pair-averaged centroid supplies the fallback
contrastive coordinate. Contrastive centroid subtraction uses the unscaled
coordinate.

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
and gradient cosine are measured on the final shared Transformer block as
training diagnostics.

The fixed L1/L2 expert-tile union is replayed at a configured interval among
population batches. This keeps both dynamic prototype systems and direct
human supervision active throughout training without changing the full-corpus
four-teacher objective. The union is decoded once into a host-memory bank;
replay and prototype refreshes use deterministic views of that same bank, so
random package reads cannot stall the GPU between expert updates.

Student prototypes are refreshed periodically from their complete fixed banks
using the current encoder state between optimizer steps. Replay mini-batches
never perform EMA updates and never redefine the bank. Thus the semantic
coordinate is independent of replay batch composition while still following
the evolving student. Explicit-negative centroids define the local
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
- Reduced-duration ablations retain the full population stream and complete
  fixed L1/L2 expert union. Only training duration and the named mechanism
  differ across matched conditions.
- The shared priority list serves the stable 3,000 tiles first and expands
  outside that boundary only when still-growing component curves require more
  examples.
- Per-component annotation sufficiency is measured separately in each of the
  four frozen teacher feature spaces on one
  fixed, slide-separated probe while a nested reference set grows. Remaining
  task-space novelty is therefore monotone non-increasing. Annotation stops
  only after every teacher has consecutive low-information-gain tail
  increments and slide/geometry QC passes in every teacher space.
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
- Existing component-presence labels prioritize the annotation queue.
- Training consumes all configured image IAC packages and the four existing
  teacher-feature IAC streams.
- Patient/slide separation remains the split unit.

## 9. Validation

Training records validation loss, L1 accuracy when validation labels exist,
and retained teacher alignment. The terminal checkpoint after the prescribed
schedule enters independent spatial validation on a slide-separated expert
asset.

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

The confirmatory comparison evaluates the spatial measurement endpoint against
the prespecified baseline.

## 10. Release and downstream boundary

The released pathology tower owns the frozen encoder, normalized `z_hcc`, L1
readout, spatial instance/measurement maps, capability masks, calibrated
measurements, preprocessing metadata, and provenance. Downstream systems
consume these outputs through a one-way, versioned interface while the released
checkpoint remains frozen.

Continuous-token adapters, language alignment, whole-slide aggregation,
patient-level diagnosis, staging, prognosis, and treatment recommendation are
separate research objects with independent evidence boundaries.

## 11. Expected contribution

HCC-SemPath contributes one HCC-specific representation with:

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
- `training/engine.py`: active objective, exact full-bank dynamic prototype
  refresh, and L2-supervised-step warm-up.
- `training/train.py`: L1/L2 ingestion, fixed expert-tile replay and prototype
  loaders, and frozen
  optimizer/supervision contracts.
- `training/spatial_validation.py`: independent completeness-aware calibration
  and aggregate spatial validation.
- `training/evaluate.py`: frozen-contract verification and evaluation-cohort
  exclusion.
- `scripts/roi_information_curve.py`: executable four-teacher,
  component-wise annotation sufficiency/QC curve.
- `scripts/calibrate_spatial_decoder.py`: terminal-checkpoint decoder freeze.
- `scripts/export_release_sempath.py`: provenance-bound release package.

Implementation conformance requires:

1. point, circle, and brush follow the component table above;
2. capability masks exclude biological instance counts from area-only and
   pigment components;
3. unresolved mixtures retain weak-background semantics;
4. L2 reaches the shared encoder after supervised-step warm-up while
   four-teacher distillation remains active;
5. undefined component/measurement pairs remain invalid;
6. annotation sufficiency is determined by component-wise information
   plateaus, not a preset tile quota;
7. reduced-duration ablations retain the full population and complete expert
   union;
8. only an independently calibrated terminal checkpoint can become a release.
