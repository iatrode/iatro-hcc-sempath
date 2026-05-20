# HCC-SemPath technical framework

This document records the current technical direction for HCC-SemPath. The project is positioned as an expert-anchor-guided compact pathology representation model, not as a task-specific clinical prediction model.

## Core technical hypothesis

A small expert-initialized hepatobiliary morphology anchor bank can organize large-scale HCC pathology patches into a disease-aware semantic coordinate system, and this organization can be distilled into an open-weight compact encoder.

The method should therefore be evaluated as a representation-learning and database-organization framework rather than as a conventional clinical endpoint model.

## Design principles

1. Use H-optimus-1 as a high-quality general pathology teacher, not as the final scientific contribution.
2. Use expert anchors as low-cost semantic control points that organize the large HCC patch database.
3. Treat anchor labels as dominant morphology anchors rather than mutually exclusive histological classes.
4. Keep private institutional WSIs as the main development resource, while using TCGA-LIHC as a public reproducibility benchmark.
5. Release code, configs, trained compact weights when allowed, anchor schema, and public benchmark scripts to compensate for non-releasable institutional raw data.

## Data roles

### Institutional HCC WSI cohort

Role:

- Main training and development cohort.
- Source for large-scale HCC patch database construction.
- Source for expert-selected morphology anchor seeds.
- Source for compact student distillation.

Requirements:

- Keep patient-level or slide-level split discipline.
- Avoid tile-level leakage across train, validation, and test partitions.
- Store all tiles through a reproducible manifest or HCCSPK package contract.
- Record scanning, magnification, MPP, and tissue-filtering settings whenever available.

### TCGA-LIHC WSI cohort

Role:

- Public reproducibility benchmark.
- Cross-cohort anchor-response stability test.
- Public demonstration of released weights and scripts.
- Public database for retrieval, anchor-response, and efficiency benchmarks.

TCGA should not be framed primarily as a clinical validation cohort for OS, DFS, RFS, or Cox modeling in this technical paper.

## Module 1: WSI tiling and data contract

Goal:

Build a reproducible large-scale patch database from institutional and TCGA HCC WSIs.

Tasks:

1. Tile WSIs at the selected magnification and tile size.
2. Apply basic tissue filtering and artifact-aware quality control.
3. Write a tile manifest containing tile ID, patient ID, slide ID, coordinates, split, and tile path.
4. Optionally package tiles into HCCSPK for portable teacher inference and public benchmarking.
5. Generate summary statistics: number of slides, patients, tiles, tissue-pass rate, and per-slide tile distribution.

Expected outputs:

- `data/manifests/*.csv`
- `data/packages/*.hccspk`
- tile QC summary tables

## Module 2: Teacher feature cache

Goal:

Use H-optimus-1 to create a high-quality teacher feature space for all eligible tiles.

Tasks:

1. Run teacher inference on institutional and TCGA tiles.
2. Save one feature vector per tile using the manifest tile ID.
3. Validate teacher feature dimensionality and completeness.
4. Record teacher model name, version, preprocessing, batch size, and device.

Expected outputs:

- `data/teacher_cache/h_optimus_1/<tile_id>.npy`
- teacher cache validation report

## Module 3: Expert-initialized morphology anchor bank

Goal:

Construct a small, auditable anchor bank that represents dominant hepatobiliary morphology directions.

Anchor design:

- Anchors are semantic control points, not exhaustive class labels.
- Anchors may overlap biologically and morphologically.
- A tile assigned to one anchor means the anchor is the dominant visible morphology for prototype construction.
- Initial anchor categories should be broad and stable at 20x / 224-pixel tile resolution.

Candidate anchor groups:

1. Tumor cell-rich area
2. Hepatocyte-like / low-atypia tumor area
3. High-atypia tumor area
4. Necrosis
5. Fibrous stroma / collagen-rich area
6. Non-tumor liver parenchyma
7. Sinusoid-rich / blood-rich area
8. Portal tract / bile duct-like area

Tasks:

1. Select high-confidence seed tiles or small regions for each anchor.
2. Encode seed tiles with the teacher model.
3. Build one or more prototype vectors per anchor by averaging or clustering seed features.
4. Store anchor metadata: name, definition, seed tile IDs, source slide IDs, number of seeds, feature extractor, and creation date.
5. Save the anchor bank as a payload containing both tensors and metadata.

Expected outputs:

- `data/anchors/hcc_semantic_anchors.pt`
- `data/anchors/anchor_schema.json`
- representative anchor tile sheets for manual review

## Module 4: Anchor-guided semantic database organization

Goal:

Use the anchor bank to organize the large HCC patch database into a semantic coordinate system.

Tasks:

1. Compute cosine similarity between every teacher feature and every anchor vector.
2. Store each tile's anchor-response vector.
3. Retrieve top-K patches for each anchor from institutional and TCGA cohorts.
4. Generate anchor heatmaps on selected WSIs using patch coordinates.
5. Generate anchor-response distribution summaries by cohort.

Expected outputs:

- tile-level anchor-response tables
- top-K retrieval sheets per anchor
- anchor heatmaps for representative WSIs
- institutional vs TCGA anchor-response distribution summaries

## Module 5: Compact student encoder training

Goal:

Train a compact encoder that inherits both the teacher morphology space and the anchor-organized semantic coordinate system.

Training objectives:

1. Feature distillation: preserve teacher morphology features.
2. Relation distillation: preserve pairwise or neighborhood structure among tiles.
3. Anchor-response distillation: preserve anchor response geometry and ranking.
4. Optional prototype contrastive objective: strengthen dominant-anchor neighborhoods without treating anchors as strictly mutually exclusive classes.
5. Optional momentum-constrained anchor adaptation: allow mild anchor adjustment while preventing semantic drift.

Tasks:

1. Train feature-only student baseline.
2. Train feature + relation student baseline.
3. Train anchor-response distillation model.
4. Train full model with all selected objectives.
5. Save checkpoints, resolved configs, training curves, and validation metrics.

Expected outputs:

- `outputs/hcc_sempath_v*/checkpoints/*.pt`
- `outputs/hcc_sempath_v*/metrics.csv`
- `outputs/hcc_sempath_v*/summary.json`

## Module 6: Baselines and controls

Goal:

Show that the method is not merely teacher imitation, random projection, or unsupervised clustering.

Required comparisons:

1. H-optimus-1 raw feature retrieval.
2. Random anchor bank.
3. Unsupervised prototype bank from clustering.
4. Student with feature-only distillation.
5. Student with feature + relation distillation.
6. Full expert-anchor-guided HCC-SemPath model.

Optional comparisons:

- Different student backbones.
- Different anchor seed counts per category.
- Fixed anchors vs trainable momentum-constrained anchors.
- Institutional-only anchors vs mixed institutional/TCGA anchors.

## Module 7: Technical validation metrics

Goal:

Validate database organization and compact-model inheritance without relying on clinical endpoint prediction.

Primary metrics:

1. Anchor retrieval precision@K: expert review of top-K patches retrieved by each anchor.
2. Anchor enrichment fold: top-K anchor retrieval compared with random retrieval.
3. Anchor-response rank correlation: student response ranking compared with teacher-anchor response ranking.
4. Top-K overlap: overlap between teacher-anchor and student-anchor retrieval lists.
5. Cross-cohort stability: consistency of anchor retrieval and response distributions between institutional data and TCGA.
6. Efficiency: parameters, FLOPs if available, tiles per second, GPU memory, and end-to-end WSI processing time.

Secondary metrics:

- Feature cosine similarity between teacher and student.
- Relation MSE between teacher and student embedding neighborhoods.
- Anchor-response MSE or KL divergence.
- Spatial coherence of anchor heatmaps.

Clinical survival, recurrence, or Cox models are not primary validation targets for this technical paper.

## Module 8: Public reproducibility package

Goal:

Make the technical contribution reviewable despite institutional raw WSI restrictions.

Tasks:

1. Release training and inference code.
2. Release configs and environment files.
3. Release compact model weights if permitted.
4. Release anchor schema and, if permitted, anchor payload or derived anchor vectors.
5. Release TCGA benchmark scripts.
6. Provide a smoke-test dataset or synthetic contract test.
7. Provide documentation for reproducing TCGA feature extraction, anchor response, retrieval, and benchmark tables.

Expected outputs:

- `MODEL_CARD.md`
- `REPRODUCIBILITY.md`
- `BENCHMARKS.md`
- public TCGA benchmark scripts

## Immediate implementation priorities

Priority 1:

- Finalize broad anchor categories.
- Add anchor metadata schema.
- Add scripts for anchor-response computation and top-K retrieval export.

Priority 2:

- Add random-anchor and unsupervised-prototype baselines.
- Add anchor-response rank correlation and top-K overlap metrics.
- Add TCGA public benchmark config.

Priority 3:

- Add feature-only, relation-only, and full anchor-guided ablation configs.
- Add real WSI throughput benchmark.
- Add model card and reproducibility guide.

Priority 4:

- Run full institutional training.
- Run TCGA public benchmark.
- Prepare expert blind review sheets for top-K anchor retrieval.

## What this project should not become

- It should not be framed as a clinical OS, DFS, RFS, or Cox prediction paper.
- It should not claim that the compact student universally outperforms the teacher.
- It should not treat broad morphology anchors as mutually exclusive diagnostic labels.
- It should not rely only on feature cosine similarity to claim success.
- It should not require public release of institutional raw WSIs to be technically reviewable.

## Intended technical contribution

HCC-SemPath aims to demonstrate that a small expert-defined hepatobiliary morphology anchor bank can organize large-scale HCC pathology data into a reusable semantic coordinate system, and that this organization can be distilled into a compact, open-weight pathology encoder suitable for public benchmarking and downstream technical reuse.
