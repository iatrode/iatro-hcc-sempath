# HCC-SemPath Training Manifest / 训练清单

The training manifest is the experiment-level cohort contract. It joins
per-WSI image-tile IAC packages with convention-based teacher feature packages
without changing the production IAC format.

训练清单是实验级 cohort 合同。它通过命名约定把 per-WSI image-tile IAC package
与 teacher feature package 连接起来，不改变生产 IAC 格式。

## Manifest Shape

```yaml
version: 1
tile_suffix: .tiles.iac
split_key: patient_id
seed: 13

datasets:
  internal:
    role: development
    tile_root: /path/to/internal_tiles
  tcga:
    role: public
    tile_root: /path/to/tcga_tiles

splits:
  train:
    internal: [case_a, case_b]
    tcga: [tcga_a]
  val:
    internal: [case_c]
  exval:
    tcga_heldout:
      source: tcga
      stems: [tcga_b]

summary:
  datasets:
    internal:
      role: development
      package_count: 3
      tile_count: 123456
  splits:
    train:
      internal:
        package_count: 2
        tile_count: 100000
```

The stem `case_a` resolves to:

```text
<tile_root>/case_a.tiles.iac
<feature_root>/<teacher>/case_a.<teacher>.features.iac
```

## Build

```bash
hcc-sempath build-train-manifest \
  --dev-source internal=/data/hcc_tiles/internal \
  --public-source tcga=/data/hcc_tiles/tcga \
  --public-exval-n 50 \
  --val-frac 0.15 \
  --split-key patient_id \
  --seed 13 \
  --output data/manifests/hcc_train.yaml
```

Before a training run, use artifact checking once teacher caches are complete:

```bash
hcc-sempath build-train-manifest \
  --dev-source internal=/data/hcc_tiles/internal \
  --public-source tcga=/data/hcc_tiles/tcga \
  --public-exval-n 50 \
  --val-frac 0.15 \
  --split-key patient_id \
  --teacher gigapath \
  --teacher h_optimus_1 \
  --teacher uni2_h \
  --teacher virchow2 \
  --feature-root /data/hcc_features \
  --check-artifacts \
  --output data/manifests/hcc_train.yaml
```

`--check-artifacts` fails if any expected tile package or teacher feature
package is missing.

## Production Rules

- Split by `patient_id` when available; use `slide_id` only when patient identity
  is unavailable.
- Keep private paths and patient-identifiable manifests outside public git.
- Use the generated `summary` to confirm package counts and tile counts before
  launching training.
- Large runs should use `data.dynamic_package_sampling: true`; every package in
  the configured train/val split participates, while batch construction mixes
  samples across packages from an in-memory shuffle buffer.
- For multi-million tile training, keep frequent validation bounded with
  `train.max_val_batches` and `train.max_eval_batches`; reserve full validation
  for selected checkpoints.
