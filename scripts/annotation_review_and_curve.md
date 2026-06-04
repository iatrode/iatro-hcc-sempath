# Annotation Review and Anchor Information Curve

This note documents two standalone research utilities under `scripts/`.
They are intentionally not wired into the main `hcc-sempath` CLI.

## Annotation review UI

Script:

```bash
scripts/annotation_review_ui.py
```

Purpose:

- Review existing tile annotations with the tile image visible.
- Save review state into JSON and a companion `.review.csv`.
- Skip entries already marked with `reviewed: true`.
- Resume from `--output-json` when that file already exists.

### L1 review

Use this when reviewing unstable L1 labels across all current classes.
With teacher features, the UI shows a feature-center suggestion, but the user still makes the final decision.

```bash
python scripts/annotation_review_ui.py \
  --annotation-json annotations/hcc_prototype_review.json \
  --output-json annotations/hcc_prototype_review.l1_review.json \
  --mode l1 \
  --teachers gigapath,uni2_h,virchow2 \
  --teacher-feature-root /path/to/teacher/features
```

Useful options:

```text
--unstable-l1 Degenerative-material
--no-open
--host 127.0.0.1
--port 0
```

### Pair review

Use this when only two L1 classes should be shown and each tile is assigned to one of them.
The two classes are arbitrary.

```bash
python scripts/annotation_review_ui.py \
  --annotation-json annotations/hcc_prototype_review.json \
  --output-json annotations/hcc_prototype_review.pair_review.json \
  --mode binary \
  --class-a HCC-tumor \
  --class-b Inflammatory-stromal
```

Change `--class-a` and `--class-b` for other two-class reviews.

### UI behavior

- Desktop: the tile list is open by default and stays open when selecting a tile.
- Mobile: the tile list is folded by default; the screen is split into tile image and controls.
- After a decision, the UI advances to the next remaining tile.
- During save, controls are disabled to avoid duplicate clicks.
- L1 mode actions are three-way: accept suggested, keep current, or use selected.
- Pair mode actions are the two class buttons.

## Anchor information curve

Script:

```bash
scripts/anchor_information_curve.py
```

Purpose:

- Estimate whether additional annotated anchors still improve prototype information.
- Use cached teacher features; it does not train a model.
- Reuse one locked validation split for all anchor counts.
- Produce summary CSVs and plots for the annotation-count curve.

Typical command:

```bash
python scripts/anchor_information_curve.py \
  --annotation-json annotations/hcc_prototype_review.json \
  --prototype-contract annotations/hcc_prototype_review.json \
  --teacher-feature-root /path/to/teacher/features \
  --output-root outputs/anchor_information_curve_real_iac_smoke \
  --seed 13
```

Help:

```bash
python scripts/anchor_information_curve.py --help
```

Main outputs:

```text
infospace_information_report.json
infospace_information_summary.csv
infospace_information_by_teacher.csv
infospace_information_by_prototype.csv
figures/infospace_information_summary.png
figures/infospace_information_teacher_audit.png
figures/infospace_novelty_distribution.png
figures/infospace_level1_prototype_curves.png
figures/infospace_level2_prototype_curves.png
figures/infospace_level1_prototype_audit.png
figures/infospace_level2_prototype_audit.png
figures/infospace_pca_qc_l1_teacher_average.png
figures/infospace_pca_qc_l1_by_teacher.png
figures/infospace_pca_qc_l2_teacher_average.png
figures/infospace_pca_qc_l2_by_teacher.png
figures/infospace_umap_qc_l1_browser_matched.png
```

Key defaults:

```text
anchor counts 100,200,400,800,1200,1600,2000,3000
seed 13
locked validation fraction 0.2
anchor group key tile_id
bootstrap iterations 500
plot formats png,pdf
```

Interpretation:

- `recommendation.recommended_anchor_count` in `infospace_information_report.json` is the selected count.
- Selection uses the first marginal-utility elbow, not complete feature exhaustion: the first low-drift count where novelty gain per 100 added anchors falls below 35% of the best observed marginal gain is treated as elbow onset, and the next count is used as the recommendation.
- `anchor_counts_requested` records requested counts.
- `anchor_counts_available` records counts that could actually be evaluated from available train anchors.
- If 3000 is requested but unavailable after locked validation is held out, it will not appear in `anchor_counts_available`.

Boundary:

- This script only computes the information curve from cached features.
- It does not modify annotation JSON.
- It does not train.
- It does not change prototype definitions.
