# Representation Audit

Retrieval pairs audited: 10000.
Unique tiles requiring teacher features: 6705.
Elapsed seconds: 141.0.

## Teacher Agreement By Retrieval Model

| model | pair teacher cosine | teacher disagreement | primary prototype match | all-4 primary match |
|---|---:|---:|---:|---:|
| z_hcc | 0.6603 | 0.0967 | 3.52 | 0.672 |
| gigapath | 0.6580 | 0.0860 | 3.49 | 0.658 |
| h_optimus_1 | 0.6640 | 0.0919 | 3.54 | 0.670 |
| uni2_h | 0.6627 | 0.0905 | 3.48 | 0.631 |
| virchow2 | 0.6575 | 0.1221 | 3.54 | 0.681 |

## z_hcc vs Teacher Retrieval Overlap

| teacher | overlap@10 | jaccard@10 | z_hcc unique@10 |
|---|---:|---:|---:|
| gigapath | 2.52 | 0.155 | 7.48 |
| h_optimus_1 | 2.21 | 0.136 | 7.79 |
| uni2_h | 2.12 | 0.130 | 7.88 |
| virchow2 | 2.40 | 0.147 | 7.60 |

## Failure Strata

| stratum | queries |
|---|---:|
| high_confidence_teacher_consensus | 45 |
| high_teacher_disagreement | 24 |
| intermediate | 117 |
| low_margin | 14 |

## Interpretation

`z_hcc` retains teacher/prototype semantic structure at a level comparable to
the individual teacher retrieval spaces, while its nearest-neighbor sets are
not simple copies of any single teacher. Mean overlap@10 with individual
teachers is only 2.12-2.52, leaving 7.48-7.88 unique `z_hcc` neighbors per
query.
