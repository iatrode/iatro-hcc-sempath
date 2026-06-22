# HCC-SemPath V2 Design-vs-Implementation Audit

Status: implementation audit, updated after corrective changes. Audited against
[`HCC_SEMPATH_V2_DESIGN.md`](HCC_SEMPATH_V2_DESIGN.md) and the `update.md` ledger
on branch `develop`.

## Verdict

The V2 core mechanism and its optimization-validity instrumentation are
**structurally complete and trainable**. The student is fixed to pretrained
DINOv2-S/14 at native 224-pixel input; backbone selection, pretraining, and patch
size are not experiment configuration options. Gate R1 / R2 diagnostics are
now emitted without dynamically changing objective weights.

## Verified consistent with design

| Contract | Location | Status |
| --- | --- | --- |
| Tri-state token target; review-complete background then positive overlay; partial annotations stay ignore | `training/roi.py` | OK |
| Geometry rasterized at patch centers; grid-mismatch aborts training | `training/roi.py`, `training/train.py:982` | OK |
| Separate patch projector, normalized attribute queries, bounded-cosine/temperature, Top-Q pooling; `encode()` unchanged; ROI branch instantiated only with a manifest | `modeling/models.py` | OK |
| One-way local→global transfer; local target stop-gradient | `training/zhcc_losses.py:198`; test `test_roi_guided_loss_routes_local_signal_only_toward_global` asserts `local_logits.grad is None` | OK |
| `L_roi` normalized by valid tokens; zero loss when no ROI tokens | `training/zhcc_losses.py:194` | OK |
| B0/B1 schedule: backbone detached until `roi_backbone_start_step`; independent ramps for ROI and consistency | `training/engine.py:375-413` | OK |
| Attribute-wise teacher adjudication default off | `training/engine.py:402` | OK |
| Global/ROI gradient norms, ROI gradient share, and objective cosine on the final shared Transformer block | `training/engine.py` | OK |
| Per-attribute activation, all-zero, all-one, and broadcast rates | `training/zhcc_losses.py` | OK |
| Fixed-validation teacher-alignment stopping rule independent of training and dynamic-prototype losses | `training/engine.py` | OK |

The focused modeling, ROI, schedule, configuration, and CSV tests pass.

## Interpretation contract

Objective diagnostics distinguish natural saturation from a disconnected,
non-finite, or dominating ROI path. They are observational: no single-objective
plateau stops training or changes a loss weight. Model stopping uses only the
predefined fixed-validation teacher-alignment rule.

## Out of scope for code, tracked by gates (real-data work)

Per `update.md`, the real-data portions of Gate R0–R3 remain unrun: rendering
and coordinate audit on production tiles, per-attribute independent-slide
feasibility, frozen external L2 and retrieval/L1 evaluation, and manuscript
integration.

## Recommendation

Launch the fixed V2 training only after Gate R0 assets pass. Use the emitted
diagnostics to certify optimization validity, freeze the selected checkpoint,
then run the external evaluation without retraining V1.
