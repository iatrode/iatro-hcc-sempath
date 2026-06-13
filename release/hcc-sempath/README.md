# HCC-SemPath Release

This directory is the local release view for the locked final model.

The full training run is stored under `artifacts/models/hcc-sempath-full/`.
`model.pt` is an encoder-only release checkpoint stored through Git LFS and
hard-linked locally to the generated release artifact. The metadata files are
relative symbolic links so the release layout can be inspected without
duplicating the training archive.

Local release files:

- `model.pt`: encoder-only `z_hcc` checkpoint
- `config.json`: inference-only model and preprocessing configuration
- `metrics.csv`: complete training metrics
- `summary.json`: final run summary
- `training-note.txt`: training completion note

The full training checkpoint, optimizer state, teacher projection heads, and
training-only configuration remain local artifacts. Teacher model weights are
not required for `z_hcc` inference.
