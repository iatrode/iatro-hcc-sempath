# Experiments

This directory contains post-training workflows and manuscript evidence for the
fixed HCC-SemPath model. Not every subdirectory is a manuscript experiment:
some are historical scaffolds or local engineering checks retained for
provenance.

## Current Manuscript Evidence

The current scientific evidence path is:

1. `10_teacher_disagreement_review`: completed expert-reviewed
   teacher-disagreement asset and main random500/top500 analysis.
2. `ablation`: completed matched 10%-scale A0-A4 mechanism study.
3. `09_representation_audit`: non-copy / teacher-retention support.
4. `07_full_exval_cache`: external representation-cache QC.
5. `01_checkpoint_comparison`: checkpoint provenance.
6. `06_attention_qc`: optional qualitative support.

## Current Expert Asset

```text
annotations/reviews/teacher_disagreement/exval_1000/review.csv
```

It contains 1,000 external-validation tiles with cross-blinded expert
adjudication:

- `random500`: 500 fully random external-validation tiles.
- `top500`: 500 non-degenerate external-validation tiles prioritized by
  teacher-only disagreement score.

The high-conflict queue is a teacher-defined stress set. Expert labels and
HCC-SemPath predictions are not used to construct it.

## Historical / Engineering Workflows

The following folders are retained for provenance or local workflow history, but
are not the current manuscript experiments:

- `00_local_eval`
- `02_embedding_export`
- `03_retrieval_benchmark`
- `04_blinded_review_package`
- `05_review_analysis`
- `08_pre_review_gate`
- `shared`

Use `experiments/reports/current_experiment_status.md` for the current status
of completed evidence and remaining gaps.
