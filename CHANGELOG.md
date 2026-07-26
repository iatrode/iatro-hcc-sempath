# Changelog

## 0.2.0 — 2026-07-26

- Restore the parallel L1/L2 training contract: teacher-only representation
  shaping is followed by one simultaneous expert-supervision ramp, with both
  objectives connected to the shared encoder.
- Replace asynchronous `l1_*` and `spatial_*` schedule keys with
  `expert_supervision_start_step` and `expert_supervision_ramp_steps`.
  `spatial_detach_shared_encoder` is reserved for the matched mechanism
  ablation.
- Add buffered optimizer-step metrics and fixed intra-epoch development probes.
- Replace dataset-size-dependent `warmup_epochs` with `lr_warmup_steps`; matched
  reduced-duration ablations retain the terminal model's absolute intervention
  and LR schedules.
- Use one grouped depthwise convolution per spatial context block without
  changing module topology or checkpoint parameter names.
- Add independent, read-only L1 and L2 review manifests with persistent
  per-pass completion and strict end-of-list navigation.
- Prevent native browser image dragging in the location overview and keep
  pointer-captured pan interaction responsive.
- Move the location overview into a navigation overlay that loads on demand,
  and reserve tile-wheel brush-width control for the active Brush tool so
  ordinary scrolling remains available elsewhere.
- Add visible `1`–`9` shortcuts for selecting the first nine L1 classes and
  use Space to save the current L1 annotation and advance.

Model constructor signatures, checkpoint state-dict topology, spatial output
semantics, and IAC dependencies are unchanged. Configurations using the removed
asynchronous schedule keys must migrate to the common expert schedule.
