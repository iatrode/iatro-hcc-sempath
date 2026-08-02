# HCC-SemPath: Parallel Classification and Spatial Prototype Supervision

Status: active design of record. This is the scientific, supervision,
implementation, validation, and release contract.

## 1. Research problem

HCC-SemPath learns two complementary pathology representations from one fixed
DINOv2-S/14 tile encoder:

1. a seven-class, mutually exclusive classification task
   \(\mathcal{T}_{\mathrm{cls}}\);
2. weakly supervised spatial morphometry task
   \(\mathcal{T}_{\mathrm{spatial}}\) for eleven HCC components,
   supporting component location, instance count, local abundance, and
   calibrated area where the supervision permits it.

The two tasks are parallel expert-supervision axes over the shared representation.
Neither head consumes the other head's features, logits, labels, or outputs.

```text
image -> shared DINOv2-S/14 encoder
      -> z_hcc -> four teacher projection heads
               -> classification prototype readout
      -> patch/local features -> spatial prototype readouts
```

The spatial branch produces component maps and calibrated morphometric
measurements. No-gradient global component centroids provide the PAMT-D
reliability coordinate during training.

## 2. Core hypothesis

A 224-pixel tile at 20x-equivalent, 0.5 micrometres per pixel contains both
cell-scale objects and structures that span several attention cells.
DINOv2-S/14 separates two
design choices: S is the compact student capacity, while the 14-pixel patch
projection covers 7 by 7 micrometres, approximately one small lymphocyte. A
useful spatial head therefore needs:

- a 14-by-14-pixel local observation window for nuclei and other small objects;
- denser coordinate sampling than the native 16-by-16 DINO token grid;
- semantic context from the final Transformer tokens;
- explicit multi-cell aggregation for fat vacuoles, ducts, vessels, and other
  extended structures;
- component-aware interpretation of point, circle, and brush marks.

The hypothesis is that fusing these signals can preserve the pretrained
teacher representation while sparse expert classification and spatial constraints reshape the
shared `z_hcc` toward scientifically interpretable global and local outputs.

## 3. Fixed semantic contract

### Classification task

The global classification classes, in fixed order, are:

1. `HCC-tumor-well-differentiated`
2. `HCC-tumor-moderately-differentiated`
3. `HCC-tumor-poorly-differentiated`
4. `Background-liver`
5. `Inflammatory-stromal`
6. `Hemorrhage-necrosis`
7. `Artifact-contamination`

The output is one seven-way softmax.

### Spatial task

The spatial components, in fixed order, are:

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

Fibroblast is a cell class. It is separated from fibrous stroma so spindle-cell
localization and density do not alter the continuous extracellular-matrix
target, and fibroblasts are not absorbed into hepatocellular parenchyma.
Small vessel denotes the predominantly circular cross-sectional vascular ROIs
in the current asset. Large vessel is a separate extended-structure class whose
wall, lumen, and context may span several attention cells.

## 4. Component-and-geometry annotation contract

Tool meaning is resolved jointly with component biology:

| Components | Point | Circle | Brush | Identifiable measurements |
| --- | --- | --- | --- | --- |
| Hepatocellular parenchyma, hemorrhage, inflammatory cells, fibroblasts | One instance | One larger instance | Dense-cell bag with unknown exact count | Instance count plus local density |
| Steatosis/vacuolation, small vessel, large vessel, ductular/portal structure | One instance with unresolved extent | One large instance with approximate extent | One connected marked structure with approximate extent; overlapping strokes merge | Structure count plus area |
| Necrosis, fibrous stroma | Positive area seed with unresolved extent | Positive extent | Positive extent | Area/coverage only |
| Bile pigment | Small positive pigment seed, not an instance | Larger positive focus with approximate extent | Irregular/fused positive pigment extent | Pigment burden/area; derived focus density only |

A circle supplies one countable centre for countable components and supplies
focus extent for bile pigment. A brush is
class-routed: dense-cell bag, connected large discrete structure, or
continuous positive area. Brush input events are annotation mechanics:
overlapping strokes on the same discrete structure form one instance.

Explicit negatives are strong negative evidence. Ordinary unmarked space can
contain deliberately unresolved mixtures and remains ignored.

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
classification annotations define teacher-specific seven-class prototypes and periodically
recomputed student-space class centroids. Every student refresh re-encodes the
complete fixed 2,800-tile bank with the current encoder; a compute mini-batch
is only a memory chunk and never defines the prototype pool. Their responses provide both HCC semantic
supervision and per-tile PAMT-D teacher reliability, while direct classification
cross-entropy supervises the human boundary.

In parallel, sparse spatial constraints define exact full-bank local
positive/negative component prototypes and reshape local features and the
shared encoder while four-teacher distillation stabilizes `z_hcc`.
Teacher-space component centroids participate in per-tile reliability
adjudication. Expert point, circle, and brush geometry supplies the spatial
targets.

### Classification head

The normalized global embedding is read against seven no-gradient,
expert-updated student-space centroids:

```text
z = normalize(project(CLS))
classification_logits[k] = cosine(z, centroid_classification[k]) / temperature
```

Human classification labels define the complete centroid bank and use cross entropy on the
resulting response. Centroids are no-gradient coordinates recomputed from the
current student between optimizer steps; gradients from the response update
`z_hcc`. Semantic distillation and PAMT-D use the seven class-conditioned
teacher prototypes.

### Dense local branch

The pretrained 14-by-14 DINO patch projection, corresponding to a 7-by-7
micrometre observation window, is reused with stride 7 and padding 4:

```text
local = conv2d(image, pretrained_patch_projection, kernel=14, stride=7, padding=4)
```

For a 224-pixel input this produces a 32-by-32 local grid with a 3.5-micrometre
sampling interval, without replacing or invalidating the teacher caches.

Final 16-by-16 Transformer patch tokens are bilinearly projected onto the
32-by-32 grid and fused with the local features. Residual context blocks with
dilations 1, 2, and 4 aggregate evidence across multiple grid cells. This is
the route used for structures whose identity is not visible in one local
window.

### Spatial outputs

For each of eleven components the head emits:

- `spatial_instance_logits [B, 11, 32, 32]`;
- `spatial_abundance_logits [B, 11, 32, 32]`.

These maps are positive-versus-negative local prototype responses over the
fused spatial features. One supervised tile/component contributes one centroid
observation, so a large brush cannot dominate merely by covering more grid
cells. Point/circle centres update instance prototypes. Cell points, dense
cell brushes, and cell-circle extent update density prototypes;
circle/brush extent updates discrete-structure area prototypes; continuous
and pigment extent updates burden/area prototypes. A structure point never
supplies an inferred extent.

A component prototype is a semantic readout coordinate after the trainable
local, semantic, fusion, and nonlinear context transformations. Morphological
variants, including heterogeneous hepatocellular nuclei, receive independent
spatial loss at their annotated locations and map into the shared component
coordinate. The spatial annotation gate measures fixed-probe coverage across the
four teacher spaces.

The instance tensor retains fixed eleven-class topology. Capability masks
deterministically suppress count outputs for non-countable channels in model
output and decoder metadata.

Decoded outputs are capability-masked:

- NMS coordinates and counts only for cell-instance and discrete-structure
  components;
- density mass/mean only for the four cell/density components;
- area fraction/pixels only for continuous, pigment, and discrete-structure
  components;
- thresholded bile-pigment focus density as a secondary morphology descriptor.

Count, density, and area remain separate measurements. Capability masks mark
undefined component/measurement pairs as invalid.

## 6. Targets and losses

Countable point marks supervise the grid cell containing the annotated centre.
The configured neighbourhood is exclusion support: nearby non-centre cells are
negative for the instance objective, so one click cannot train multiple decoded
peaks. A point never supplies inferred object area. Dense-cell brushes and
circle/brush contours supervise every selected grid cell as positive occupied
support. A circle additionally supplies instance-exclusion support so one
large object cannot produce multiple centres. Mixed point/brush cell
annotations supervise resolved instances and unresolved abundance separately.
Explicit negatives supervise confirmed absence; all other unmarked cells remain
outside the loss.

PAMT-D combines teacher base weights and per-tile reliability as
`w_m * alpha_mi`. Feature and classification-semantic losses are normalized jointly over
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
  + lambda_classification * CE(T_cls)
  + lambda_spatial * (
        L_instance_point_peak
      + lambda_abundance_point * L_abundance_point_peak
      + L_brush_density_bag
      + L_positive_area
      + L_explicit_negative
    )
```

Each routed objective is averaged within component and then across its active
components. The fixed objective weights combine count, density, and area
endpoints. Cell brushes remain density bags; structure and continuous-component
extent marks supervise the area head. Negative loss reaches the instance head
only for countable components.

Explicit negatives define the prototype boundary whenever available.

## 7. Optimization

Training begins with four-teacher feature and relation distillation alone.
After the teacher-only interval, prototype-semantic, classification, and spatial supervision
enter together and ramp on one global optimizer-step clock:

```text
expert_supervision_start_step  3000
expert_supervision_ramp_steps  1000
```

Both expert tasks reach the shared encoder from their first non-zero supervised
update. PAMT-D reliability filtering and student-response matching
begin after the common expert ramp and then increase to their configured
strength. Global and spatial gradient norms, spatial gradient share, and
gradient cosine are measured on the final shared Transformer block.

`step_metrics.csv` stores every optimizer step through buffered GPU-to-host
transfers. `development_metrics.csv` stores loss components on the same fixed
development subset every 1,000 optimizer steps. Epoch summaries remain in
`metrics.csv`; they are not the sole source for convergence or stopping
analysis.

The fixed classification/spatial expert-tile union is replayed at a configured interval among
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
decision boundary when available.

## 8. Data organization

- The classification task uses the frozen 2,800-tile expert asset, with 400 tiles
  per class.
- The spatial task uses the current eleven-component annotation manifest.
- Both assets are intentionally small expert interventions on the
  population-scale four-teacher representation; neither is expected to label
  the full corpus.
- The union of their training tiles is replayed throughout population
  distillation; newly annotated tiles enter the same union automatically.
- A0 Optuna search uses one deterministic 10% population subset, the complete
  training expert union, and independent complete L1/L2 validation banks. It
  searches learning rate, weight decay, and the global spatial-task weight for
  at most 16 epochs. Multi-GPU execution assigns one independent trial to each
  GPU under one coordinator. Asynchronous device reuse prevents paid GPU
  idling after early stopping; constant-liar TPE includes in-flight
  configurations when proposing the next trial. The fixed 10% population is
  the compute-matched study
  population for A0 and every formal ablation; it is not a checkpoint-selection
  device. Every formal mechanism ablation inherits the A0 seed,
  hyperparameters, maximum budget, checkpoint-selection rule, schedule, and
  subset without retuning. Because an intervention can change the
  initialization-scale of a validation term, each ablation freezes its own
  epoch-0 \(T_0,C_0,S_0\) denominators while retaining the A0 weights, ramp
  boundary, patience, and relative-improvement rule. The A0 ramp boundary is
  explicit, so disabling a late-ramped mechanism cannot create earlier
  checkpoint opportunities. On this fixed 10% study population, every pass is
  a common approximately 1,284-update checkpoint across A0 and all ablations;
  those completed results retain that matched grid. Full-population training
  evaluates the identical joint criterion on a fixed global-step grid so an
  epoch containing ten times as many updates cannot multiply the stopping
  budget. Its cosine learning-rate horizon is likewise inherited as the A0
  study's absolute optimizer-step count rather than recomputed from full-data
  epochs. Conditions that remove classification training
  still retain the complete L1 validation bank. Each condition receives an
  exact source/asset contract over its active teacher/prototype subset, and a
  checkpoint from an obsolete shorter epoch plan cannot be extended into the
  formal run. The ablation base is accepted only when the completed Optuna
  study export binds it to the winning trial's hyperparameters, raw config,
  selected epoch, and checkpoint digest; training revalidates the complete
  resolved condition digest at startup.
- The shared priority list serves the stable 3,000 tiles first and expands
  outside that boundary only when still-growing component curves require more
  examples.
- Per-component annotation sufficiency is measured on fixed,
  slide-separated probes while nested reference sets grow in each of the four
  frozen teacher feature spaces. Remaining task-space novelty is therefore
  monotone non-increasing. The stopping decision pools teacher-by-resample
  curves exactly as the classification audit does; per-teacher support remains
  diagnostic. Annotation stops after the pooled plateau is confirmed and
  slide/geometry QC passes.
- The spatial curve x-axis is unique component-positive tiles rather than clicks or
  strokes. Point/circle/brush counts, component-specific measurement
  capabilities, slide balance, rasterization failures, and annotation
  conflicts are independent QC. Raw RGB or an untrained DINO representation is
  never used as a substitute for the four-teacher distillation task space.
- The fixed-probe plateau must repeat across slide-aware resamples and
  consecutive tail increments. New-batch discovery novelty may rebound and is
  reported only as a secondary diagnostic, never as a stopping signal.
- The final spatial tile count is the union required for all eleven component curves
  to pass. Early-saturating components stop consuming annotation effort;
  still-growing components drive subsequent tile selection.
- Existing component candidates prioritize the annotation queue.
- Training consumes all configured image IAC packages and the four existing
  teacher-feature IAC streams.
- Patient/slide separation remains the split unit.

## 9. Validation

Training keeps ordinary teacher validation, seven-class expert validation, and
eleven-component spatial expert validation as separate streams. A0 checkpoint
selection uses three fixed validation terms:

- \(T_e\): direct four-teacher retention on a deterministic 65,536-tile probe
  (the unchanged first 128 batches at batch size 512) drawn from the fixed 10%
  population-validation packages. It is computed from mean cosine distance
  on the full probe plus relation MSE on its deterministic evenly spaced
  4,096-tile subset, using the configured relation weight. Dynamic PAMT-D
  targets and student prototypes do not enter this term. Every L1/L2 expert
  train and validation tile is excluded from this label-blind probe; ordinary
  population-validation diagnostics are accumulated on the same 128 batches.
- \(C_e\): class-balanced cross entropy over the complete seven-class expert
  validation bank.
- \(S_e\): one complete-bank, component-balanced reduction of the
  eleven-component spatial objective, including explicit negatives.

The first trial evaluates the shared initialization before optimization to
freeze \(T_0,C_0,S_0\). Every later trial recomputes and must reproduce those
values before using the stored denominators. Checkpoint \(k\), indexed by
global optimizer step, minimizes
\[
J_k=w_TT_k/T_0+w_CC_k/C_0+w_SS_k/S_0,
\]
with the formal A0 weights \((w_T,w_C,w_S)=(0.26,0.28,0.46)\). The weights must be positive,
sum to one, and are configuration-bound for the prespecified three-cell
sensitivity analysis. Selection and patience begin only after every active
expert/PAMT-D supervision ramp has completed. Full-population patience counts
fixed-step joint-validation probes, never epochs; the reduced A0/ablation
matrix retains its completed common pass-end grid, which is also a fixed step
grid because every condition uses the same 10% population. Macro-F1, balanced accuracy,
ordinary population loss, and individual teacher cosines remain diagnostics.
No validation expert tile enters replay or any optimizer-visible bank; when
historical IAC package splits differ from finalized annotation splits, the
exact validation rows are removed from population training.
The formal search uses one coordinator and one strict global trial budget.
Optuna pruning steps count only post-ramp joint-validation observations, so a
late ramp cannot silently consume the pruning warm-up.

After checkpoint freezing, spatial validation is performed on an independent,
slide-separated expert sample:

- instance localization and count metrics only for count-capable components;
- density calibration for hepatocellular, hemorrhagic, inflammatory-cell, and
  fibroblast components;
- area metrics for necrosis, fibrous stroma, bile pigment, vacuolation,
  small vessels, large vessels, and ductular/portal structures;
- bile-pigment focus-density repeatability only under a frozen threshold,
  connectivity, minimum-area, and spatial-scale definition;
- results stratified by component mode and annotation geometry;
- per-component and macro results with independent-slide counts.

`experiments/scripts/calibrate_spatial_decoder.py` implements this gate. Exact point/count
pairs require explicit per-component `roi_count_complete`; measurement pairs
likewise require `roi_measurement_complete`. These flags are validation-only
claims and are never inferred from ordinary weak training marks. Brush and
circle contours use their full selected support under the formal contract.
Threshold scoring excludes incomplete tile/component pairs, retains explicit
and includes complete-negative tiles when selecting bile minimum-focus size.

Validation tile IDs must resolve inside the requested manifest `val`/`exval`
partition, which must be patient/slide-disjoint from the entire
optimizer-visible population plus the classification/spatial expert replay
cohort. The finalized joint-selection checkpoint records its selected global
step, containing epoch, completed run terminal step/epoch, and freezes the exact
optimizer-visible package list, an aggregate
package/cohort digest, and SHA-256 digests of mutable classification/spatial/prototype
supervision assets. Resume, independent evaluation, and calibration never
reconstruct that contract from a later directory scan. The aggregate
report contains component/mode, point/circle/brush/mixed strata, slide-macro
metrics, and independent-slide counts. The versioned decoder asset binds the
ordered thresholds, NMS kernels, bile minimum-focus size, and output stride to
the exact selected model-state digest, research contract, annotation,
protocol, and validation cohort. The release exporter rejects a calibration
from any other checkpoint.

The confirmatory comparison evaluates the spatial measurement endpoint against
the prespecified baseline.

## 10. Release and downstream boundary

The released pathology tower owns the frozen encoder, normalized `z_hcc`, classification
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
- stable seven-class global tissue state;
- eleven-component, geometry-aware spatial output;
- native support for class-routed point/circle/brush annotations;
- cell-scale localization plus multi-grid structural context;
- count, density, and validated area measurements for downstream spatial
  analysis.

## 12. Implementation and acceptance map

- `modeling/models.py`: dynamic classification/local spatial prototype readouts, dense fused
  spatial features, and decoder.
- `spatial_schema.py`: fixed eleven-class measurement capabilities.
- `training/roi.py`: instance centres, instance-exclusion support, density
  bags, positive area, explicit negatives, and ignored unmarked regions.
- `training/spatial_losses.py`: exact-centre instance peaks, abundance point
  peaks, contour-faithful brush support, positive-area loss, and
  capability-routed negatives.
- `training/pamtd.py`: per-tile four-teacher adjudication and shared semantic
  response target.
- `training/engine.py`: active objective, exact full-bank dynamic prototype
  refresh, common classification/spatial intervention schedule, step-level
  metrics, fixed intra-epoch joint-selection probes, and exact step
  checkpoints.
- `training/train.py`: classification/spatial ingestion, fixed expert-tile replay and prototype
  loaders, and frozen
  optimizer/supervision contracts.
- `training/spatial_validation.py`: independent completeness-aware calibration
  and aggregate spatial validation.
- `training/evaluate.py`: frozen-contract verification and evaluation-cohort
  exclusion.
- `experiments/scripts/roi_information_curve.py`: executable four-teacher,
  component-wise annotation sufficiency/QC curve.
- `experiments/scripts/calibrate_spatial_decoder.py`: finalized-selection decoder freeze.
- `experiments/scripts/export_release_sempath.py`: provenance-bound release package.

Implementation conformance requires:

1. point, circle, and brush follow the component table above;
2. capability masks exclude biological instance counts from area-only and
   pigment components;
3. unmarked cells in unresolved mixtures remain ignored; only explicit
   negative marks or complete-negative review create negative targets;
4. teacher-only distillation precedes a simultaneous classification/spatial ramp, and both
   expert objectives reach the shared encoder from their first active update;
5. undefined component/measurement pairs remain invalid;
6. annotation sufficiency is determined by component-wise information
   plateaus, not a preset tile quota;
7. formal ablations retain the same fixed 10% population subset and complete
   expert union, and differ only by the named mechanism; A1 is the package-level
   global expert-intervention control, whereas A2 and A11 isolate
   prototype-adjudicated reliability and student-response matching;
8. only an independently calibrated, finalized joint-selection checkpoint can
   become a release.
