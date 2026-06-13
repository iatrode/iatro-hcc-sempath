# Experiments

This directory contains the tracked post-training reproduction workflows based
on the complete HCC-SemPath run in
`artifacts/models/hcc-sempath-full`.

The goal is to turn the fixed final model into a reproducible manuscript
evidence workflow.  The model is no longer being selected or tuned here; these
experiments document local reproducibility, external-set inference, retrieval
candidate generation, representation structure, interpretability QC, and the
final expert-review endpoint.

## Evidence Flow

The manuscript evidence flow is:

1. **Checkpoint documentation**: `00_local_eval`, `01_checkpoint_comparison`.
2. **External sampled representation cache**: `07_full_exval_cache`.
3. **External retrieval benchmark and candidate generation**:
   `08_pre_review_gate`.
4. **Representation structure audit**: `09_representation_audit`.
5. **Interpretability support**: `06_attention_qc`.
6. **Blinded expert review and scoring**: `04_blinded_review_package`,
   `05_review_analysis`.

`02_embedding_export` and `03_retrieval_benchmark` are retained as early smoke
tests for the export/retrieval code path. They are not the primary manuscript
evidence now that the 200k exval IAC cache and downstream retrieval audit exist.

## Directory Policy

Use experiment-centered subdirectories. Each numbered experiment keeps its own
`configs/`, `scripts/`, `results/`, `logs/`, and `reports/` folders.

This is preferred over a top-level `configs/scripts/results` split because each
experiment has a different data contract and output lifecycle. Keeping assets
next to the experiment that produced them makes reruns, review, and cleanup less
ambiguous.

Shared helpers that are reused across experiments go in `shared/`.

```text
experiments/
  00_local_eval/
    configs/
    scripts/
    results/
    logs/
    reports/
  01_checkpoint_comparison/
    configs/
    scripts/
    results/
    logs/
    reports/
  02_embedding_export/
    configs/
    scripts/
    results/
    logs/
    reports/
  03_retrieval_benchmark/
    configs/
    scripts/
    results/
    logs/
    reports/
  04_blinded_review_package/
    configs/
    scripts/
    results/
    logs/
    reports/
  05_review_analysis/
    configs/
    scripts/
    results/
    logs/
    reports/
  06_attention_qc/
    configs/
    scripts/
    results/
    logs/
    reports/
  07_full_exval_cache/
    scripts/
    results/
    logs/
    reports/
  08_pre_review_gate/
    scripts/
    results/
    logs/
    reports/
  09_representation_audit/
    scripts/
    results/
    logs/
    reports/
  shared/
    configs/
    scripts/
  reports/
```

## Fixed Inputs

Final checkpoint:

```text
artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt
```

Intermediate checkpoint for training-stage comparison:

```text
artifacts/models/hcc-sempath-full/checkpoints-61/best_scientific_score.pt
```

Local manifest:

```text
configs/local/mac/manifest.yaml
```

Local prototype assets:

```text
artifacts/prototypes/
```

## Target Checklist

### 00 Local Eval

Research question: does the final checkpoint reproduce stable teacher-alignment
and HCC prototype diagnostics on local `val` and `exval` splits?

Core outputs:

- localized final evaluation config
- `eval_val.json`
- `eval_exval.json`
- short metric summary

Primary checkpoint:

```text
artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt
```

### 01 Checkpoint Comparison

Research question: did the run improve from the epoch-61 intermediate state to
the final epoch-100 state under the same local evaluation protocol?

Core outputs:

- epoch 61 vs epoch 100 metric table
- delta summary for teacher alignment, scientific score, prototype-bank metrics
- recommendation of which checkpoint is the manuscript default

Compared checkpoints:

```text
artifacts/models/hcc-sempath-full/checkpoints-61/best_scientific_score.pt
artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt
```

### 02 Embedding Export

Role: smoke test for embedding export. The manuscript-scale cache is produced by
`07_full_exval_cache`.

Research question: can the final model produce a reusable `z_hcc` embedding
table with enough metadata for retrieval and audit?

Core outputs:

- student `embedding_norm` arrays
- tile metadata table
- slide/package provenance table
- optional prototype-response table

Required metadata:

- tile id
- slide id
- dataset/source cohort
- split
- tile package path or package id
- row index
- image coordinates when available

### 03 Retrieval Benchmark

Role: smoke test for retrieval-table generation. The manuscript-scale retrieval
benchmark is produced by `08_pre_review_gate`.

Research question: can `z_hcc` and teacher baselines generate fixed top-k
retrieval tables for blinded review packaging?

Core models:

- HCC-SemPath `z_hcc`
- gigapath
- h_optimus_1
- uni2_h
- virchow2
- teacher-average/fused baseline if technically available

Core outputs:

- fixed query set
- fixed gallery set
- top-k retrieval tables for every model
- merged query-result table for blinded review
- machine-readable retrieval run manifest

### 04 Blinded Review Package

Research question: can expert reviewers score retrieval relevance without model
identity leakage?

Core outputs:

- deduplicated query-result pair table
- randomized review order
- reviewer-facing CSV
- reviewer-facing image/HTML package when tile images are available
- hidden answer key mapping review items to model sources

Review labels should support:

- morphology relevance score
- optional failure reason
- optional dominant morphology note

### 05 Review Analysis

Research question: under blinded expert judgment, does HCC-SemPath outperform
teacher baselines in morphology retrieval?

Core outputs:

- precision@k
- mean relevance@k
- NDCG@k
- model win rate
- bootstrap confidence intervals
- reviewer agreement diagnostics if multiple reviewers are available

### 06 Attention QC

Research question: do final-model saliency, attention rollout, and prototype
responses align with interpretable HCC morphology?

Core outputs:

- sampled tile attention sheets
- saliency overlays
- attention rollout overlays when available
- prototype-response summary table
- high-confidence and failure-case examples

### 07 Full Exval Cache

Role: primary external representation cache.

Research question: can the fixed final model produce a stable, IAC-native,
sampled external-test `z_hcc` feature cache?

Core outputs:

- 200,000 sampled exval tile embeddings
- one `hcc_sempath_z_hcc.features.iac` package per exval tile package
- tile/source-row metadata
- cache manifest and QC summary

### 08 Pre-review Gate

Role: primary retrieval-candidate generation before expert review.

Research question: can `z_hcc` and teacher baselines generate complete,
same-slide-filtered retrieval candidates from the sampled external cache?

Core outputs:

- 200 query / 50,000 gallery benchmark
- top-10 retrieval tables for `z_hcc` and four teacher baselines
- embedding QC
- balanced draft blinded-review candidate set

### 09 Representation Audit

Role: automatic representation-structure evidence before expert review.

Research question: does `z_hcc` retain multi-teacher/prototype structure while
forming a retrieval neighborhood distinct from any single teacher?

Core outputs:

- cross-model retrieval overlap
- teacher agreement per retrieval pair
- prototype-primary agreement summary
- query confidence/disagreement strata

## Recommended Goal Order

1. Document local checkpoint behavior with `00_local_eval` and
   `01_checkpoint_comparison`.
2. Generate the sampled external IAC cache with `07_full_exval_cache`.
3. Generate retrieval candidates with `08_pre_review_gate`.
4. Audit representation structure with `09_representation_audit`.
5. Generate selected interpretability examples with `06_attention_qc`.
6. Freeze the final blinded package.
7. After expert scores are available, run `05_review_analysis`.

## Current Known State

The final training run completed at epoch 100 and global step 297300.

The current local evidence package includes a 200,000-tile sampled exval
`z_hcc` IAC cache, five-model retrieval benchmark, representation audit, and a
draft balanced review candidate set. Expert scoring has not started.

Checkpoint-internal metrics for
`artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt`:

- scientific score: 0.816994
- teacher alignment score: 0.817271
- prototype-bank Level-1 accuracy: 0.964333
- prototype-bank Level-2 macro AUC: 0.778199
- prototype-bank top-k precision: 0.734811
- prototype-bank Level-1 neighborhood purity: 0.947867
- prototype-bank Level-2 neighborhood purity: 0.634289

Use `summary.json` and checkpoint `best_metrics` as authoritative run summaries.
The historical `metrics.csv` may contain early rows from before the CSV column
padding fix and should not be parsed naively by header alone.
