# Remote teacher-cache workflow

## Goal

Build teacher features once on a high-performance machine, then train the
student without repeatedly running the teacher.

## Inputs copied to the remote machine

Preferred transfer artifact:

1. `tiles.iac`
2. Repository code

The package contains an IatroCache header, Arrow slide/record tables, and
JXL-compressed `224 x 224` tiles. It is a pre-inference data package, not the
teacher feature cache.

## Remote commands

```bash
conda env create -f environment.yml
conda activate hcc-sempath
python -m pip install --no-deps -e .
python scripts/download_teacher.py
hcc-sempath build-teacher-cache \
  --input data/packages \
  --output data/features/h_optimus_1 \
  --teacher h_optimus_1 \
  --batch-size 1024 \
  --precision fp16 \
  --compile \
  --num-workers 8 \
  --prefetch-factor 2 \
  --continue-on-error \
  --device cuda
```

`--num-workers` controls parallel IatroCache tile reads and JXL decode during
teacher inference. `--prefetch-factor` controls the number of prefetched batches
per worker. Keep `--batch-size` as the GPU-memory knob and tune workers only
when GPU utilization is low.
`--precision fp16` or `--precision bf16` enables CUDA autocast for teacher
inference and writes the resulting feature matrix back as float32. `--compile`
uses `torch.compile`; its first generated package includes compilation warm-up
cost, so judge throughput after the first batches have completed.

`--input` may point to one image-tile `.iac` file or a directory of
image-tile `.iac` files. Tile size is read from each input package header. A
directory input fails if discovered packages have inconsistent tile dimensions.
For directory inputs, existing valid outputs are skipped unless `--overwrite`
is passed. Each generated or skipped output is validated immediately and written
to `teacher_cache_progress.csv`; a JSON summary with total, processed, ok, and
failed counts is written beside it.
The planned supported presets are `h_optimus_1`, `gigapath`, `uni2_h`, and
`virchow2`. `--teacher` also accepts local model directories and custom timm /
`hf_hub:*` names for controlled experiments.

## Output naming

Input package naming is already fixed by the image-tile workflow. Teacher model
outputs should use:

```text
<teacher-name>.features.iac
```

Examples:

```text
slide_a.gigapath.features.iac
slide_a.h_optimus_1.features.iac
```

## Outputs copied back after teacher inference

```text
data/features/h_optimus_1/*.h_optimus_1.features.iac
data/features/h_optimus_1/teacher_cache_progress.csv
data/features/h_optimus_1/teacher_cache_progress.json
```

Training uses the `*.features.iac` package as the distillation target. Teacher
feature construction writes IatroCache directly and does not create loose `.npy`
intermediates.

## Verification before training

```bash
hcc-sempath train --config configs/distill_train.example.yaml
```

If a cache file is missing or has the wrong dimensionality, training fails during startup before the first batch is launched.

## Open-source hygiene

The public repository should contain code, schemas, documentation, examples,
and synthetic smoke-test artifacts only. Do not commit production WSIs,
production tile packages, teacher feature packages, checkpoints, patient-level
manifests, access tokens, or machine-local paths.
