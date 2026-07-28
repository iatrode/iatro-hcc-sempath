# Changelog

## 0.2.1

- Align the combined annotation audit, documentation, and annotation queue
  with the pooled fixed-probe plateau rule shared by classification and
  spatial supervision.
- Record the source annotation digest and generation time in spatial
  information reports.
- Retain read compatibility with historical per-teacher low-gain reports.

## 0.2.0 — 2026-07-26

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
- Replace the deprecated initial classification annotation filename with the
  final asset used after tumor-differentiation adjudication.

Model constructor signatures, checkpoint state-dict topology, spatial output
semantics, and IAC dependencies are unchanged. Configurations using the removed
asynchronous schedule keys must migrate to the common expert schedule.
