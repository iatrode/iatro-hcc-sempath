# Teacher-Disagreement Expert Review Summary

This is the current manuscript-level expert annotation asset for teacher-disagreement analysis.

## Reviewed Queues

- `random500`: 500 fully random exval tiles; 500 adjudicated.
- `top500`: 500 non-degenerate high teacher-conflict exval tiles; 500 adjudicated.

The queues were constructed before expert adjudication. `top500` is the deterministic top-ranked teacher-conflict stress queue; expert labels and HCC-SemPath predictions are not inputs to the selection rule.

## Conflict Separation

| source_group | tiles | disagreement_mean | disagreement_median | disagreement_min | disagreement_max |
|---|---:|---:|---:|---:|---:|
| random500 | 500 | 0.3626 | 0.2507 | 0.0902 | 1.1168 |
| top500 | 500 | 1.3072 | 1.3008 | 1.2787 | 1.4448 |
| all | 1000 | 0.8349 | 1.1978 | 0.0902 | 1.4448 |

The disagreement score is exactly reconstructed as `vote_entropy + primary_pairwise_l1 + attribute_pairwise_l1`; the maximum absolute reconstruction error is 1.20e-07.

## Teacher Agreement With Expert L1

| source_group | teacher | accuracy | balanced accuracy | macro F1 |
|---|---|---:|---:|---:|
| random500 | gigapath | 0.834 | 0.640 | 0.629 |
| random500 | h_optimus_1 | 0.874 | 0.653 | 0.648 |
| random500 | uni2_h | 0.864 | 0.628 | 0.635 |
| random500 | virchow2 | 0.864 | 0.644 | 0.638 |
| top500 | gigapath | 0.132 | 0.263 | 0.078 |
| top500 | h_optimus_1 | 0.590 | 0.280 | 0.285 |
| top500 | uni2_h | 0.684 | 0.354 | 0.334 |
| top500 | virchow2 | 0.258 | 0.293 | 0.173 |
| all | gigapath | 0.483 | 0.427 | 0.365 |
| all | h_optimus_1 | 0.732 | 0.501 | 0.502 |
| all | uni2_h | 0.774 | 0.504 | 0.512 |
| all | virchow2 | 0.561 | 0.504 | 0.451 |

## PAMT-D / Ablation Agreement With Expert L1

| source_group | model | accuracy | balanced accuracy | macro F1 |
|---|---|---:|---:|---:|
| random500 | pred_full | 0.948 | 0.844 | 0.879 |
| random500 | pred_a1 | 0.660 | 0.541 | 0.483 |
| random500 | pred_a2 | 0.672 | 0.546 | 0.494 |
| random500 | pred_a3 | 0.514 | 0.482 | 0.397 |
| random500 | pred_a4 | 0.542 | 0.441 | 0.379 |
| random500 | pred_a6 | 0.670 | 0.541 | 0.489 |
| top500 | pred_full | 0.932 | 0.876 | 0.877 |
| top500 | pred_a1 | 0.526 | 0.347 | 0.281 |
| top500 | pred_a2 | 0.590 | 0.353 | 0.294 |
| top500 | pred_a3 | 0.384 | 0.339 | 0.206 |
| top500 | pred_a4 | 0.442 | 0.275 | 0.214 |
| top500 | pred_a6 | 0.600 | 0.353 | 0.293 |
| all | pred_full | 0.940 | 0.857 | 0.879 |
| all | pred_a1 | 0.593 | 0.467 | 0.404 |
| all | pred_a2 | 0.631 | 0.478 | 0.421 |
| all | pred_a3 | 0.449 | 0.428 | 0.327 |
| all | pred_a4 | 0.492 | 0.374 | 0.312 |
| all | pred_a6 | 0.635 | 0.474 | 0.417 |

## Key L2 Macro Results

| source_group | model | macro AUC | macro AP | macro F1@0.5 |
|---|---|---:|---:|---:|
| random500 | pred_full | 0.742953 | 0.362044 | 0.246981 |
| random500 | pred_a2 | 0.665823 | 0.296206 | 0.24294 |
| random500 | pred_a6 | 0.672697 | 0.301897 | 0.246909 |
| top500 | pred_full | 0.71875 | 0.324169 | 0.246989 |
| top500 | pred_a2 | 0.694784 | 0.315062 | 0.239451 |
| top500 | pred_a6 | 0.697221 | 0.314567 | 0.243677 |
| all | pred_full | 0.752301 | 0.332104 | 0.251821 |
| all | pred_a2 | 0.701184 | 0.277056 | 0.24399 |
| all | pred_a6 | 0.70581 | 0.279428 | 0.248128 |

## Paired L1 Accuracy Deltas

| source_group | comparison | full-minus-comparison accuracy delta | 95% CI |
|---|---|---:|---:|
| random500 | teacher_plurality | 0.070 | [0.048, 0.092] |
| random500 | teacher_uni2_h | 0.084 | [0.060, 0.112] |
| random500 | pred_a2 | 0.276 | [0.234, 0.320] |
| top500 | teacher_plurality | 0.262 | [0.218, 0.306] |
| top500 | teacher_uni2_h | 0.248 | [0.202, 0.294] |
| top500 | pred_a2 | 0.342 | [0.298, 0.392] |
| all | teacher_plurality | 0.166 | [0.140, 0.191] |
| all | teacher_uni2_h | 0.166 | [0.139, 0.193] |
| all | pred_a2 | 0.309 | [0.277, 0.343] |

## Matched Ablation L1 Deltas

| source_group | contrast | reference-minus-comparison accuracy delta | 95% CI |
|---|---|---:|---:|
| random500 | pred_a1 vs pred_a3 | 0.146 | [0.108, 0.184] |
| random500 | pred_a2 vs pred_a1 | 0.012 | [-0.006, 0.032] |
| random500 | pred_a4 vs pred_a3 | 0.028 | [0.002, 0.054] |
| random500 | pred_a6 vs pred_a1 | 0.010 | [-0.008, 0.030] |
| random500 | pred_a6 vs pred_a4 | 0.128 | [0.092, 0.164] |
| random500 | pred_a6 vs pred_a2 | -0.002 | [-0.012, 0.008] |
| random500 | pred_a0 vs pred_a5 | 0.062 | [0.034, 0.090] |
| top500 | pred_a1 vs pred_a3 | 0.142 | [0.098, 0.182] |
| top500 | pred_a2 vs pred_a1 | 0.064 | [0.032, 0.096] |
| top500 | pred_a4 vs pred_a3 | 0.058 | [0.032, 0.084] |
| top500 | pred_a6 vs pred_a1 | 0.074 | [0.040, 0.106] |
| top500 | pred_a6 vs pred_a4 | 0.158 | [0.116, 0.200] |
| top500 | pred_a6 vs pred_a2 | 0.010 | [-0.002, 0.022] |
| top500 | pred_a0 vs pred_a5 | 0.118 | [0.076, 0.158] |
| all | pred_a1 vs pred_a3 | 0.144 | [0.116, 0.174] |
| all | pred_a2 vs pred_a1 | 0.038 | [0.019, 0.057] |
| all | pred_a4 vs pred_a3 | 0.043 | [0.025, 0.061] |
| all | pred_a6 vs pred_a1 | 0.042 | [0.022, 0.062] |
| all | pred_a6 vs pred_a4 | 0.143 | [0.116, 0.170] |
| all | pred_a6 vs pred_a2 | 0.004 | [-0.004, 0.012] |
| all | pred_a0 vs pred_a5 | 0.090 | [0.064, 0.116] |

## Conflict Quartile Sensitivity

| conflict bin | disagreement range | top500 fraction | full accuracy | plurality accuracy |
|---|---:|---:|---:|---:|
| Q1 | 0.0902-0.2507 | 0.000 | 0.968 | 0.964 |
| Q2 | 0.2507-1.1168 | 0.000 | 0.928 | 0.792 |
| Q3 | 1.2787-1.3007 | 1.000 | 0.936 | 0.676 |
| Q4 | 1.3008-1.4448 | 1.000 | 0.928 | 0.664 |

## Interpretation

- Random queue full-model L1 accuracy: 0.948.
- High-conflict queue full-model L1 accuracy: 0.932.
- High-conflict A6 matched 10%-scale L1 accuracy: 0.6.
- The high-conflict queue is a teacher-defined stress set, not a population estimate; random500 and top500 should be reported separately.
- `queue_construction_provenance.csv` records the selection basis and leakage boundary.
- `l1_per_class_metrics.csv` and `conflict_quartile_l1_metrics.csv` provide class-level and conflict-graded checks for reviewer-facing tables.
- `high_conflict_topn_sensitivity.csv` reports whether the high-conflict result is stable across top100/top250/top500 subsets.
