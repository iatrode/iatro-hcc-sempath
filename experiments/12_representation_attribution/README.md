# 12 Representation Attribution

Purpose: Generate representation-level patch-occlusion sensitivity overlays for the student model $z_{\mathrm{HCC}}$ and the four teachers (GigaPath, H-optimus-1, UNI2-h, Virchow2), showing the visual feature attribution differences.

## Configs and inputs

- Selected tiles / Annotation queue metadata: `experiments/06_attention_qc/configs/reviewed_attention_cases.csv`
- Model checkpoint: `artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt`
- Model config: `artifacts/models/hcc-sempath-full/resolved_config.json`

## Execution

Run the script with the project's Python environment:

```bash
python experiments/12_representation_attribution/scripts/run_attribution.py
```

Arguments:
- `--tile-ids`: (Optional) Specify space-separated list of review IDs (e.g. `TD-0864 TD-0477`). Defaults to the 4 representative cases.
- `--output-dir`: (Optional) Specify custom output path.

## Outputs

- `results/representation_attribution_panel.png`
- `results/representation_attribution_panel.pdf`
