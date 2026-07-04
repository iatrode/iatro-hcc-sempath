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
- `hcc_l2_roi_v2_candidates.json`: local V2 ROI priority pool from existing L2 positives (generated; not committed).
- `hcc_l2_roi_v2.json`: V2 nine-class complete-review ROI annotation state (created by the UI).
- `reviews/teacher_disagreement/exval_1000/review.json`: completed 1000-tile repeated review state.
- `reviews/teacher_disagreement/exval_1000/review.csv`: CSV export of the completed repeated review.



  L1（4 类，互斥）：

1. HCC-tumor
3. Background-liver
4. Inflammatory-stromal
5. Degenerative-material

  L2（10 类，可并存）：

1. hepatocellular-parenchyma-present
2. necrosis-present
3. hemorrhage-present
4. bile-pigment-present
5. inflammatory-cell-present
6. fibrous-stroma-present
7. steatosis-vacuolation-present
8. hyaline-change-present
9. vascular-structure-present
10. ductular-portal-present

L1 expansion:
HCC-tumor further classification

WHO grading system (3 tiered system)
Well differentiated: tumor cells resemble mature hepatocytes; minimal to mild nuclear atypia
Moderately differentiated: tumor cells appear malignant on H&E and morphology suggests hepatocellular differentiation; moderate nuclear atypia
Poorly differentiated: tumor cells appear malignant on H&E and often cannot be distinguished from other poorly differentiated neoplasms; marked nuclear atypia

Modified Edmondson-Steiner grading system (4 tiered system) (Cancer 1954;7:462)
Grade I: tumor cells are difficult to differentiate from hyperplastic liver cells
Grade II: tumor cells resemble mature hepatocytes with slightly larger and more hyperchromatic nuclei; sharp and clear cut cell borders; frequent acini formation
Grade III: tumor cells are larger and have more hyperchromatic nuclei with less acidophilic cytoplasms; trabecular distortion; numerous tumor giant cells
Grade IV: tumor cells are intensely hyperchromatic, with scant and less granular cytoplasm; tumor cells appear less cohesive and can appear giant, spindled or short and plump; medullary growth pattern with loss of trabeculation; less acini

Background-liver
Further expansion
hepatocyte
Portal triad / portal tract
Central vein

