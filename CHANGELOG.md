# Changelog

## 0.2.0.dev3 — Unreleased

- Add the licence and cross-platform release badges to the repository and
  model-card entrypoints. Align the unpinned PyPI installation command across
  GitHub, Hugging Face, and ModelScope after validating prerelease discovery.

## 0.2.0.dev2 — 2026-08-04

- Add pinned GitHub Actions workflows for routine push/PR validation and
  manually dispatched, tag-bound PyPI and GitHub releases. Release preflight
  verifies source metadata, PyPI version availability, distribution identity,
  console entrypoints, and the public/private asset boundary.
- Consolidate the English and Chinese open-source entrypoints around PyPI,
  Hugging Face, and ModelScope. The developer extra now contains the complete
  test, lint, and repository-local Optuna toolchain; build and publication
  dependencies remain isolated inside the manual release workflow. The
  redundant search extra is removed before its first public release.
- Align declared dependencies with the source contract: add SciPy and the
  ModelScope client at runtime, move experiment plotting and optional
  TensorBoard logging into the developer extra, add supported minimum
  versions, and remove the obsolete direct image-codec dependency.

## 0.2.0.dev1 — 2026-08-04

- Normalize the first-release supervision terminology to
  `T_cls`/classification and `T_spatial`/spatial throughout the public
  documentation, experiment contracts, diagnostics, and examples. This is a
  terminology-only change; configuration fields and serialized assets are
  unchanged.

- Delegate prediction-record compression to the IAC 0.1.3
  `VariableRecordPack` and native ZSTD codec. SemPath now owns only its
  prediction payload schema and no longer declares or implements a private
  zlib codec convention.
- Replace the unpublished flat CLI with workflow commands. Reusable assets are
  built through the `hcc-sempath build` namespace; annotation uses
  `hcc-sempath annotate`; training, evaluation, and benchmarking remain direct
  top-level workflows. Move build and annotation implementations out of the
  CLI routing package and remove all obsolete command names before the first
  public release.
- Add `hcc-sempath infer` for released-model inference over tile IAC packages.
  Prediction IAC outputs preserve source identities, model digests, probability
  encoding, spatial grid geometry, and an explicit level-0 coordinate transform.
  Make `benchmark` consume the same gated release contract instead of internal
  training configuration and checkpoints.
- Extend released-model inference to 224-pixel raster images and WSI files,
  including tissue-aware WSI tiling, progress reporting, and canonical
  `.tile.path.iac`, `.feat.path.iac`, and `.pred.path.iac` names. Add
  `hcc-sempath download` for the local gated-model cache and make the public
  feature builder emit one verified four-teacher package per tile package.
- Store the inference-only release state in `model.safetensors`; training
  checkpoints remain private PyTorch recovery artifacts and are never part of
  the gated model release.

- Declare the public source and documentation license as
  CC-BY-NC-ND-4.0, aligned with the planned gated student-weight release.

- Add fixed-global-step joint teacher/classification/spatial validation and checkpoint
  selection for full-population training. Patience and minimum evidence count
  validation probes rather than population epochs, preventing a larger
  corpus from silently multiplying the optimization budget. Every evaluated
  model is atomically recoverable, including eligible probes in the first
  population epoch. Full-population preparation also carries forward the
  selected A0 cosine schedule's absolute step horizon. The legacy epoch-end
  mode remains available for exact
  reproduction of the completed fixed-10% A0 and ablation studies.

- Run fixed-teacher validation matrix diagnostics on the evaluation device,
  vectorize retrieval-overlap scoring, and serialize identical epoch
  checkpoint aliases once. This removes the unreported CPU pause after
  validation without changing checkpoint selection or metric definitions.
- Hash only Git-tracked source in a checkout and ignore generated editable
  install metadata in a declared source archive, so formal source receipts are
  identical before and after installation and cannot absorb host-local files.
- Normalize invalid zero-valued host thread variables before any Python/CUDA
  startup check, preserving the intended CPU quota instead of letting libgomp
  reject the AutoDL image defaults.
- Make the clean-archive NVMe preparation entry point resolve its repository
  modules without relying on an ambient working-directory `PYTHONPATH`.
- Allow spatial decoder metrics to be emitted for the checkpoint-selection
  supervision bank without falsely declaring overlapping source cohorts to be
  an independent validation cohort; strict independent calibration remains the
  default.
- Separate metrics on assigned positive/explicit-negative semantic support
  from descriptive activation in non-assigned regions, and report
  tile-component precision, recall, F1, and AUC only over known labels.
- Materialize exact-resume epoch accumulators only when a mid-epoch checkpoint
  is due, eliminating otherwise unused per-step CUDA-to-CPU scalar
  synchronizations without changing checkpoint contents or resume semantics.
- Replace the unpublished six-class classification contract with the final
  seven-class contract, splitting hemorrhage/necrosis from
  artifact/contamination and fixing the training bank at 400 tiles per class.
- Add independent full-bank classification and spatial validation streams for A0 search and
  checkpoint selection. The frozen A0 selection loss combines epoch-0-
  normalized component-balanced spatial loss, class-balanced classification
  cross entropy, and direct fixed-teacher feature/relation retention with
  explicit configurable weights.
- Exclude exact finalized expert-validation rows from population training and
  exact expert-training rows from population validation when historical IAC
  package splits and the finalized annotation split differ.
- Replace the training-loss Optuna proxy with a contract-hashed, resumable A0
  search over learning rate, weight decay, and global spatial-task weight.
  Search artifacts bind the trial, best epoch, checkpoint, configuration, and
  supervision digests.
- Make formal A1–A11 runs inherit the selected A0 maximum budget and normalized
  teacher/classification/spatial checkpoint rule, removing the obsolete three-/six-epoch
  population-loss stopping path. Classification-removal conditions retain the
  complete classification validation bank while zeroing classification training loss.
  Freeze the A0 ramp boundary, bind each condition's active source/asset
  subset, and reject continuation from a checkpoint created under a shorter
  epoch plan. Bind `best_config.yaml` to the completed study's winning trial
  and checkpoint, and revalidate each full resolved ablation config at train
  startup.
- Fix point supervision to train the annotated centre instead of selecting
  the model's current local maximum.
- Train every selected range cell when full-range supervision is configured,
  and combine point, range, and area evidence once per measurement component.
- Penalize sparse complete-negative false-positive peaks with a balanced
  global-mean and top-four hard-tail loss on both valid spatial heads.
- Exclude range-derived instance support from measurement prototype refresh.
- Add `spatial_measurement_positive` as a diagnostic metric. Existing
  checkpoints and annotation files remain load-compatible; retraining is
  required to obtain the corrected loss behaviour.

- Define the first-release classification/spatial API and asset schema after
  removing unpublished development terminology.
- Define classification prototype registries as one ordered class bank;
  spatial prototypes are represented only by the independent spatial branch.
- Use the final eleven-component spatial taxonomy without presence suffixes,
  including distinct `small-vessel` and `large-vessel` components.
- Align the combined annotation audit, documentation, and annotation queue
  with the pooled fixed-probe plateau rule shared by classification and
  spatial supervision.
- Record the source annotation digest and generation time in spatial
  information reports.
- Restore the parallel classification/spatial training contract:
  teacher-only representation shaping is followed by one simultaneous
  expert-supervision ramp, with both objectives connected to the shared
  encoder.
- Replace asynchronous task-specific schedule keys with
  `expert_supervision_start_step` and `expert_supervision_ramp_steps`.
  `spatial_detach_shared_encoder` is reserved for the matched mechanism
  ablation.
- Add buffered optimizer-step metrics and fixed intra-epoch development probes.
- Replace dataset-size-dependent `warmup_epochs` with `lr_warmup_steps`;
  matched reduced-duration ablations retain the terminal model's absolute
  intervention and LR schedules.
- Use one grouped depthwise convolution per spatial context block without
  changing module topology or checkpoint parameter names.
- Add independent, read-only classification and spatial review manifests with
  persistent per-pass completion and strict end-of-list navigation.
- Prevent native browser image dragging in the location overview and keep
  pointer-captured pan interaction responsive.
- Move the location overview into a navigation overlay that loads on demand,
  and reserve tile-wheel brush-width control for the active Brush tool so
  ordinary scrolling remains available elsewhere.
- Add visible `1`–`9` shortcuts for selecting the first nine classification
  classes and use Space to save and advance.
- Use the final classification annotation asset after tumor-differentiation
  adjudication.

This is the first release contract. Earlier local checkpoints, configurations,
and annotation schemas were development artifacts and are not supported.
