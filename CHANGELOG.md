# Changelog

## 0.2.0 — Unreleased

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
