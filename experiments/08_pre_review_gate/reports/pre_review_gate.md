# Pre-review Gate

Sampled cache rows: 200000.
Query count: 200. Gallery count: 50000. Top-k: 10.
Elapsed seconds: 373.3.

## Retrieval QC

| model | queries full top-k | mean same-slide skipped | mean top1-topk margin |
|---|---:|---:|---:|
| z_hcc | 200 | 50.58 | 0.0493 |
| gigapath | 200 | 65.39 | 0.0526 |
| h_optimus_1 | 200 | 61.66 | 0.0450 |
| uni2_h | 200 | 62.26 | 0.0441 |
| virchow2 | 200 | 34.88 | 0.0350 |

## Gate Decision

PASS for automatic pre-review benchmarking: all models produced full same-slide-filtered top-k retrieval tables.
Expert scoring is still not started; this gate only validates the frozen candidate-generation machinery.
