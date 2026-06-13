# 00 Local Eval

Purpose: rerun the final checkpoint on local `val` and `exval` splits using
local paths for the manifest and prototype assets.

Primary output:

- `results/eval_val.json`
- `results/eval_exval.json`
- `reports/local_eval_summary.md`

Default checkpoint:

```text
../../artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt
```
