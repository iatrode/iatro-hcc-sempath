# Local Evaluation Summary

Protocol: fixed-seed local sampled evaluation on MPS using `max_eval_batches=4`, `batch_size=16`, and split tile fraction `0.001`.

| metric | val | exval |
|---|---:|---:|
| `teacher_alignment_score` |  |  |
| `scientific_score` |  |  |
| `gigapath_feature_cosine` | 0.494267 | 0.619073 |
| `h_optimus_1_feature_cosine` | 0.901700 | 0.648619 |
| `uni2_h_feature_cosine` | 0.659694 | 0.693640 |
| `virchow2_feature_cosine` | 0.900486 | 0.866906 |
| `gigapath_retrieval_overlap` | 0.725000 | 0.721875 |
| `h_optimus_1_retrieval_overlap` | 0.421875 | 0.690625 |
| `uni2_h_retrieval_overlap` | 0.573438 | 0.681250 |
| `virchow2_retrieval_overlap` | 0.495312 | 0.653125 |
| `prototype_bank_zhcc_level1_accuracy` |  |  |
| `prototype_bank_zhcc_level2_macro_auc` |  |  |
| `prototype_bank_zhcc_prototype_topk_precision` |  |  |
