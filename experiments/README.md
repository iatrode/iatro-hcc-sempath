# Experiments and Evidence Status

HCC-SemPath V2 is the active route: four-teacher PAMT-D shapes shared
`z_hcc`; L1 is four-class classification; L2 is nine-component spatial
morphometry. The implementation contract is complete, but annotation is not
frozen and no V2 training checkpoint, calibrated decoder, external result, or
ablation result exists yet.

## Active sequence

| Claim unit | Required experiment | Status |
| --- | --- | --- |
| Four teachers shape reusable HCC representation | full-training retention, prototype alignment, and gradient diagnostics | pending V2 training |
| Small expert supervision guides population learning | full V2 training with complete fixed L1/L2 replay | pending |
| L1 supports four tissue states | patient/slide-separated external L1 evaluation | pending |
| Countable L2 components localize instances | complete point/circle localization and count validation | pending |
| Dense-cell brushes support abundance | complete region-level density calibration | pending |
| Continuous/structural/pigment components support spatial composition | complete area/burden calibration | pending |
| Bile pigment supports focus density | frozen threshold/connectivity/minimum-size validation | pending |
| PAMT-D, multiple teachers, and dynamic prototypes contribute | matched A0-A6 full-population, one-tenth-duration study | pending |
| Release reproduces the trained model | checkpoint, preprocessing, cohort, supervision, and decoder provenance audit | implemented; empirical asset pending |

V2 becomes paper-evaluable only after:

1. L2 annotation reaches confirmed component-wise information plateaus;
2. full V2 training finishes;
3. an independent slide-separated asset freezes the component decoders;
4. external evaluation and the matched mechanism study complete.

## Historical V1 boundary

The existing numerical outputs under experiment/result directories were
produced by the superseded V1 route, including its tile-level L2 attribute
readout. They remain local or recoverable from Git history for provenance and
hypothesis formation, but are not V2 localization, count, density, burden, or
area evidence.

Two maintained local protocols remain:

- [`ablation/README.md`](ablation/README.md): planned V2 A0-A6 mechanism study.
- [`10_teacher_disagreement_review/README.md`](10_teacher_disagreement_review/README.md):
  historical 1,000-tile human-review asset construction and leakage boundary.
