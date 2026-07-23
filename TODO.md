# HCC-SemPath TODO

Only unfinished work belongs here. Implemented behavior and scientific
semantics are maintained in
[`docs/HCC_SEMPATH_V2_DESIGN.md`](docs/HCC_SEMPATH_V2_DESIGN.md).

## Annotation freeze

- Continue purpose-driven L2 annotation only for components whose
  modality-aware, slide-aware information curves have not reached a confirmed
  plateau.
- Freeze the final L2 training asset and a separate exhaustive
  slide/patient-separated spatial validation asset.

## Training and evidence

- Run full V2 training with the fixed L1/L2 expert union.
- Confirm spatial gradients reshape the shared encoder while retaining L1 and
  four-teacher alignment.
- Run the matched full-population, one-tenth-duration A0-A6 mechanism study.
- Calibrate count, density, area, and bile-focus outputs only from the frozen
  independent spatial validation asset.
- Run external L1 and spatial-composition evaluation; publish only aggregate,
  de-identified results.

## Public release

- Freeze the model card, data/annotation schemas, preprocessing contract, and
  reproducibility manifest after the empirical gates pass.
- Export only a terminal checkpoint paired with its validated spatial decoder
  and complete cryptographic provenance.
