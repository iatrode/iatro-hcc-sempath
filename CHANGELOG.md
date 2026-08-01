# Changelog

## 0.2.0 — Unreleased

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
- Add independent full-bank L1 and L2 validation streams for A0 search and
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
- Make formal A1–A12 runs inherit the selected A0 maximum budget and normalized
  teacher/L1/L2 checkpoint rule, removing the obsolete three-/six-epoch
  population-loss stopping path. Classification-removal conditions retain the
  complete L1 validation bank while zeroing classification training loss.
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
