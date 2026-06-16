# 06 Attention QC

Purpose: generate tile-level final-decision occlusion sensitivity examples for
the final checkpoint.

Cases are drawn randomly within the random500/top500 and major Level-1 strata
from the completed blind-evaluation asset. A morphology-only candidate sheet is
used to exclude low-tissue, artifact-dominated, or visually uninformative
tiles before sensitivity is computed. Sensitivity outputs are not used for case
selection.

The visualization is not an attention map. Each heatmap measures the decrease
in the final prototype-classification margin after masking one ViT input patch.
It is a decision perturbation diagnostic, not a causal pathology localization.

Primary output:

- `results/tile_attention_sheet.png`
- `results/*.occlusion.png`
- `results/tile_attention_rows.csv`
- `reports/attention_qc_summary.md`
- `reports/reviewed_attention_candidates.png`
- `reports/attention_manuscript_panel.{png,pdf}`
