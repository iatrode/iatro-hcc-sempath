# HCC-SemPath TODO

This public TODO tracks repository-level engineering work. Dataset-specific paths,
private benchmark outputs, and institution-specific notes should stay outside the
git repository.

## Open-source readiness

- Keep `README.md`, `docs/PROJECT_DIRECTION.md`, `docs/model_plan.md`,
  `docs/TECHNICAL_FRAMEWORK.md`, and `docs/SEMANTIC_PROTOTYPE_PLAN.md` aligned
  around the lightweight vertical HCC representation objective.
- Add `docs/RELATED_WORK.md` with public citations and model-boundary notes.
- Add `docs/HCC_SEMANTIC_SPACE.md` describing reusable HCC morphology semantics.
- Add schema files for tile manifests, teacher manifests, semantic prototype
  metadata, and public-safe evaluation summaries.
- Add model-card and reproducibility templates before public release.

## Training system

- Add training-manifest documentation and schema examples for per-WSI IAC
  training cohorts.
- Add sampled validation subsets for large-scale training.
- Add WSI-window or feature-cache-aware sampling for current whole-matrix
  teacher feature packages.
- Add benchmark scripts for teacher-feature storage tradeoffs before changing
  the feature package layout.
- Add minimal HCC-specific weak-supervision objectives on the shared `z_hcc`
  embedding.
- Implement semantic prototype initialization from curated concept embeddings.
- Add bounded prototype updates, prototype diagnostics, and prototype capacity
  scaling experiments.

## Evaluation

- Separate teacher-imitation metrics from HCC-specific representation metrics.
- Add morphology retrieval, clustering, cross-center stability, and prototype
  utilization diagnostics.
- Add ablations for individual teachers, single-teacher students, multi-teacher
  students without prototypes, fixed prototypes, and bounded momentum prototypes.
- Add public benchmark summaries that do not contain raw WSI paths, patient-level
  identifiers, or private artifact paths.

## Data governance

- Keep raw WSIs, large tile packages, teacher feature packages, checkpoints, and
  per-tile production tables outside git.
- Track only schemas, small synthetic fixtures, public-safe summaries, and scripts.
