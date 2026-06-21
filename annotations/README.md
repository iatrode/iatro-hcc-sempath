# Annotation Assets

This directory is reserved for human annotation assets and review outputs.

Do not treat these files as disposable caches. Generated model scores, NPZ
exports, memmaps, logs, and temporary selection caches should stay under
`artifacts/caches/`; completed human review JSON/CSV files should be moved here.

The annotation payloads themselves are intentionally ignored by Git because they
can contain local paths, case identifiers, and large derived tables. Commit only
code, schemas, or documentation unless a dataset release is explicitly planned.

Current local layout:

- `hcc_prototype_review.final_3000_inflammatory_stromal.json`: final prototype annotation state.
- `hcc_l2_roi_v2_candidates.json`: local frozen V2 ROI candidate queue (generated; not committed).
- `hcc_l2_roi_v2.json`: V2 nine-class complete-review ROI annotation state (created by the UI).
- `reviews/teacher_disagreement/exval_1000/review.json`: completed 1000-tile repeated review state.
- `reviews/teacher_disagreement/exval_1000/review.csv`: CSV export of the completed repeated review.
