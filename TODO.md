# HCC-SemPath TODO

This public TODO tracks repository-level engineering work. Dataset-specific paths,
private benchmark outputs, and institution-specific notes should stay outside the
git repository.

Prototype annotation, prototype package loading, curated concept prototype
building, prototype supervision labels, PAMT-D adjudication, and prototype
metrics are implemented. Remaining prototype-related work is evaluation and
release reporting, not core prototype-system plumbing.

## Open-source release

- Add `docs/RELATED_WORK.md` with public citations and model-boundary notes.
- Add `docs/HCC_SEMANTIC_SPACE.md` describing reusable HCC morphology semantics.
- Add machine-readable schema files for IatroCache packages, training manifests,
  prototype supervision manifests, prototype package metadata, and public-safe
  evaluation summaries.
- Add model-card and reproducibility templates before public release.

## Training operations

- Add fixed validation subset export from the training manifest for reproducible
  frequent validation.
- Add WSI-window or feature-cache-aware sampling if real dry-run profiling shows
  feature-cache locality limits throughput.
- Add benchmark scripts for teacher-feature storage tradeoffs before changing
  the feature package layout.

## Evaluation and reporting

- Separate teacher-imitation metrics from HCC-specific representation metrics in
  evaluation reports and public summaries.
- Add morphology retrieval, clustering, cross-center stability, and prototype
  utilization diagnostics.
- Add ablations for individual teachers, single-teacher students, multi-teacher
  students without prototypes, fixed prototypes, and prototype capacity scaling.
- Add public benchmark summaries that do not contain raw WSI paths, patient-level
  identifiers, or private artifact paths.
