# Teacher-Disagreement Expert Review

> Historical V1 asset. The reviewed annotations and teacher-disagreement
> construction are retained for provenance, but all HCC-SemPath predictions
> and L2 attribute metrics here come from the superseded V1 model. They are not
> V2 spatial-model evidence.

## Design

The reviewed set contains 1,000 external-validation tiles. Two pathologists
independently annotated every tile; all discordant Level-1 decisions and
Level-2 attributes were jointly reviewed and resolved into a consensus label:

- `random500`: 500 fully random external-validation tiles.
- `top500`: 500 non-degenerate external-validation tiles prioritized by high
  teacher disagreement.

The high-conflict queue is defined by poor coordination among teacher outputs:
higher disagreement receives higher priority. The ranking key is
`disagreement_score = vote_entropy + primary_pairwise_l1 + attribute_pairwise_l1`.
The queue construction does not use expert labels or HCC-SemPath predictions;
those are applied only after the two 500-item queues are fixed. The reviewed
file stores the consensus Level-1 state, Level-2 attributes,
per-teacher Level-1 outputs, confidence values, and disagreement diagnostics.

## Input

```text
annotations/reviews/teacher_disagreement/exval_1000/review.csv
```

## Provenance outputs

```text
experiments/10_teacher_disagreement_review/tables/
experiments/10_teacher_disagreement_review/reports/
```

`tables/queue_construction_provenance.csv` records the selection basis for each
queue and the leakage boundary.

Generated V1 score tables and figures are historical outputs, not maintained
documentation and not V2 evidence.
