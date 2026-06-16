# 09 Representation Audit

Purpose: support the non-copy claim for `z_hcc` by auditing whether its
retrieval neighborhood is distinct from individual teacher neighborhoods while
retaining multi-teacher/prototype structure.

Primary output:

- `tables/model_teacher_agreement_summary.csv`
- `tables/cross_model_overlap_summary.csv`
- `tables/query_failure_strata_summary.csv`
- `results/cross_model_overlap.csv`
- `results/pair_teacher_agreement.csv`
- `results/model_teacher_agreement_summary.csv`
- `results/query_failure_strata.csv`
- `results/prototype_agreement_summary.csv`
- `reports/representation_audit.md`

The `tables/` files are the public-facing summary tables copied from the
completed manuscript audit artifact. The `scripts/` entry is retained for local
reproduction when the private feature packages are available.
