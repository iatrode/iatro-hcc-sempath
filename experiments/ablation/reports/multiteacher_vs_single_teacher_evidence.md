# Multi-teacher vs Single-teacher Prototype Evidence

## Scope

This note fixes the current evidence for the training/validation-stage semantic
fitting comparison between:

- A0: full multi-teacher + prototype + filter
- A2: multi-teacher + prototype, filter disabled
- A3: single-teacher Virchow2, no prototype
- A4: single-teacher Virchow2 + prototype

The purpose is to settle the semantic-fitting evidence in two steps:

1. A3 vs A4 shows the single-teacher floor and the effect of adding the
   prototype mechanism.
2. A4 vs A0/A2 shows the multi-teacher gain under prototype-supervised
   conditions.

Filter/adjudication is intentionally left as a separate question and should be
read mainly from A0 vs A2.

## Data Integrity

The A0 and A4 `metrics.csv` files changed schema after epoch 3. The rows are
recoverable:

- A0: epochs 1-3 use the original 96-column header; epochs 4-20 use the
  224-key order in `summary.json`.
- A4: epochs 1-3 use the original 81-column header; epochs 4-20 use the
  113-key order in `summary.json`.
- A2 and A3 have consistent CSV files.

The figures and summary table in this directory use this mixed-schema recovery.

## Main Result

For ZHCC semantic fitting, A3 is the negative/control floor: without prototype
supervision, the prototype-bank semantic fitting readouts are not defined. A4
turns on the prototype mechanism under the same single-teacher setting and
produces measurable semantic fitting. A0/A2 then improve the main semantic
fitting readouts over A4 under multi-teacher prototype training.

| condition | prototype supervision | L1 accuracy e20 | L1 purity e20 | L2 macro AUC e20 | top-k precision e20 |
|---|---|---:|---:|---:|---:|
| A0 full multi-teacher + prototype | on | 0.8730 | 0.8725 | 0.7831 | 0.6542 |
| A2 multi-teacher + prototype, no filter | on | 0.8733 | 0.8724 | 0.7818 | 0.6525 |
| A3 single-teacher, no prototype | off | n/a | n/a | n/a | n/a |
| A4 single-teacher + prototype | on | 0.8527 | 0.8724 | 0.7557 | 0.6480 |

The most defensible primary statement is based on L1 accuracy because Level-1 is
approximately mutually exclusive:

> Multi-teacher prototype training improves ZHCC Level-1 semantic fitting over
> single-teacher prototype training by about 2 percentage points at epoch 20.

A3 is still part of the argument: it establishes that the single-teacher,
no-prototype floor does not instantiate the prototype-bank semantic fitting
task. Therefore A4 is the appropriate single-teacher comparator for A0/A2, and
A3 documents what is added when prototype supervision is introduced.

L2 is non-mutually-exclusive, so macro AUC should be interpreted as label
ranking/separation rather than hard classification accuracy:

> Multi-teacher prototype training also improves Level-2 semantic label ranking,
> with epoch-20 macro AUC about 0.026-0.027 higher than the single-teacher
> prototype condition.

## Trend

A0 and A2 both rise steadily in L1 accuracy:

- A0: 0.4183 -> 0.7083 -> 0.8217 -> 0.8730 at epochs 1, 3, 10, 20.
- A2: 0.4183 -> 0.7083 -> 0.8247 -> 0.8733 at epochs 1, 3, 10, 20.
- A3: prototype-bank semantic readouts are not defined because prototype
  supervision is disabled.
- A4: 0.4627 -> 0.7473 -> 0.7893 -> 0.8527 at epochs 1, 3, 10, 20.

A4 starts higher early, but the multi-teacher conditions overtake and finish
higher by the late epochs. This supports a training-stage semantic fitting
benefit from multi-teacher prototype training rather than a single noisy final
point.

## Filter Boundary

A0 and A2 are nearly identical on these semantic fitting metrics. A0 has a small
edge on L2 macro AUC and top-k precision, while A2 is marginally higher on L1
accuracy. Therefore, this evidence supports multi-teacher over single-teacher,
but it does not settle filter/adjudication.

Filter should be discussed separately, ideally in teacher-disagreement or
low-margin strata where a reliability filter is expected to matter most.

## Teacher-Imitation QC

Teacher feature cosine is included as QC, not as the primary semantic evidence.
It verifies that the student models learned the teacher representation targets.
For multi-teacher models, per-teacher feature cosine should be read separately
because the aggregate score mixes four different targets. A3/A4 are
Virchow2-only and therefore have no GigaPath, H-optimus-1, or UNI2-h readout.

## Figures

- `figures/zhcc_l1_accuracy_trend.png`
- `figures/zhcc_l1_purity_trend.png`
- `figures/zhcc_l2_macro_auc_trend.png`
- `figures/prototype_topk_precision_trend.png`
- `figures/semantic_fitting_final_bar.png`
- `figures/semantic_fitting_final_bar_with_a3_na.png`
- `figures/ablation_evidence_availability_matrix.png`
- `figures/teacher_feature_cosine_heatmap.png`
- `figures/teacher_feature_cosine_aggregate_trend.png`
- `figures/teacher_feature_cosine_trend_gigapath.png`
- `figures/teacher_feature_cosine_trend_h_optimus_1.png`
- `figures/teacher_feature_cosine_trend_uni2_h.png`
- `figures/teacher_feature_cosine_trend_virchow2.png`

The numeric source tables are:

- `multiteacher_vs_single_teacher_summary.csv`: measured A0/A2/A4 prototype
  comparisons.
- `ablation_semantic_fitting_summary.csv`: four-condition matrix including A3
  as the no-prototype floor.
- `teacher_imitation_qc_summary.csv`: final teacher feature cosine, relation
  MSE, and retrieval-overlap QC.
