# 11 Representation UMAP

Purpose: Generate a 2D UMAP projection of the student embedding space $z_{\mathrm{HCC}}$ on the 1,000 expert-adjudicated validation tiles.

## Configs and inputs

- Predictions/Annotations source: `artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv`
- Model checkpoint: `artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt`
- Model config: `artifacts/models/hcc-sempath-full/resolved_config.json`

## Execution

Run the script using the conda python environment:

```bash
python experiments/11_representation_umap/scripts/run_umap.py
```

## Outputs

- `results/zhcc_umap.png` (High-resolution PNG)
- `results/zhcc_umap.pdf` (Vector PDF for publication)
