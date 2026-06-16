# Checkpoint Comparison

Protocol: fixed-seed local sampled MPS evaluation.

| split | metric | epoch61 | epoch100 | delta |
|---|---|---:|---:|---:|
| val | `gigapath_feature_cosine` | 0.482419 | 0.494267 | 0.011848 |
| val | `h_optimus_1_feature_cosine` | 0.902278 | 0.901700 | -0.000579 |
| val | `uni2_h_feature_cosine` | 0.657904 | 0.659694 | 0.001791 |
| val | `virchow2_feature_cosine` | 0.899293 | 0.900486 | 0.001193 |
| exval | `gigapath_feature_cosine` | 0.611393 | 0.619073 | 0.007680 |
| exval | `h_optimus_1_feature_cosine` | 0.637140 | 0.648619 | 0.011479 |
| exval | `uni2_h_feature_cosine` | 0.682387 | 0.693640 | 0.011253 |
| exval | `virchow2_feature_cosine` | 0.860849 | 0.866906 | 0.006057 |

Manuscript default under this sampled local protocol: `epoch100`.
