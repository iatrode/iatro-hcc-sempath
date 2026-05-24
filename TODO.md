# HCC-SemPath TODO

This public TODO tracks repository-level engineering work. Dataset-specific paths,
private benchmark outputs, and institution-specific notes should stay outside the
git repository.

## Open-source readiness

- Keep `README.md`, `docs/PROJECT_DIRECTION.md`, `docs/model_plan.md`, and
  `docs/TECHNICAL_FRAMEWORK.md` aligned around the lightweight vertical HCC
  representation objective.
- Add `docs/RELATED_WORK.md` with public citations and model-boundary notes.
- Add `docs/HCC_SEMANTIC_SPACE.md` describing reusable HCC morphology semantics.
- Add schema files for tile manifests, teacher manifests, and HCC semantic anchors.
- Add model-card and reproducibility templates before public release.

## Training system

- Implement CUDA mixed precision training with autocast and gradient scaling.
- Add multi-teacher training with teacher-specific heads and a shared `z_hcc`
  embedding.
- Add multi-package dataset loading for per-slide or sharded tile packages.
- Add WSI-level split management to prevent tile-level leakage.
- Add sampled validation subsets for large-scale training.

## Evaluation

- Separate teacher-imitation metrics from HCC-specific representation metrics.
- Add ablations for single-teacher, multi-teacher, weak-supervision-only, and
  full multi-teacher plus HCC weak-supervision training.
- Add public benchmark summaries that do not contain raw WSI paths, patient-level
  identifiers, or private artifact paths.

## Data governance

- Keep raw WSIs, large tile packages, teacher feature packages, checkpoints, and
  per-tile production tables outside git.
- Track only schemas, small synthetic fixtures, public-safe summaries, and scripts.
