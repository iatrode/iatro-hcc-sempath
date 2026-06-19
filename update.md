# HCC-SemPath Level-2 Upgrade — Final Plan

Status: final design for staged implementation. The current release model and published primary results remain unchanged until the corresponding decision gate is passed.

## 1. Research Question

Level 1 describes the dominant tile-level tissue state and remains a global mutually exclusive response. Level 2 describes coexisting local morphology presence. The upgrade asks:

> Can slide-cross-fitted, attribute-wise teacher adjudication and independently anchored local morphology evidence reshape `z_hcc` to encode reliable Level-2 semantics without degrading its global Level-1 organization, retrieval quality, or teacher-prior retention?

## 2. Core Hypothesis

Teacher reliability is attribute-specific. A teacher may be reliable for necrosis but unreliable for bile pigment on the same tile. Therefore:

- the existing global tile-teacher reliability continues to control feature/relation distillation;
- a new tile-teacher-attribute reliability controls only the Level-2 semantic target;
- local spatial supervision is introduced through a separate patch-token path and is transferred back into the deployable global embedding through directional local-to-global consistency.

The primary intervention is attribute-wise filtering. The spatial branch is a second-stage extension and proceeds only if the loss-only intervention passes its mechanism gate.

## 3. Data Organization and Leakage Control

### 3.1 Data roles

- **Training/prototype side**: prototype discovery, teacher-space prototype construction, student prototype construction, reliability estimation, model fitting, and model selection.
- **External/held-out side**: final Level-2 mechanism evaluation, blinded morphology retrieval, and held-out localization evaluation.
- All partitions are made by slide. Institution is additionally balanced or held out where the cohort permits.

### 3.2 Slide-level cross-fitting

The 3,000 expert-labelled prototype tiles are split into `K` folds by slide. For every prototype tile, teacher responses used to estimate label agreement are produced with prototypes that exclude the tile's entire slide. Aggregated out-of-fold predictions cover all eligible tiles while preventing in-sample prototype-label agreement.

Cross-fitting is applied to the Level-2 reliability path before the loss-only pilot. The existing Level-1 path is held fixed during the first L2 comparison to avoid changing two mechanisms simultaneously; its in-sample reliability is audited separately and may be cross-fitted in a dedicated follow-up experiment.

### 3.3 Teacher-attribute reliability prior

Out-of-fold expert labels estimate a global teacher-attribute prior:

```text
R[m, k] = cross-fitted reliability of teacher m for attribute k
```

`R[m,k]` is estimated with empirical-Bayes or hierarchical shrinkage toward the same teacher's global reliability or cross-attribute mean. Stability is judged using effective independent slide count and slide-clustered confidence intervals, not the positive count of an individual fold.

Fallback policy:

1. use shrunk `R[m,k]` when its stability criterion is met;
2. otherwise use the global teacher prior or disable attribute-wise filtering for that attribute;
3. never use pure cross-teacher consensus as a standalone reliability estimate.

## 4. Attribute-Wise Level-2 Adjudication

### 4.1 Teacher response

Each frozen teacher and its offline teacher-space prototype registry produce:

```text
s[m, k](x) in [0, 1]
```

for teacher `m`, attribute `k`, and tile `x`.

### 4.2 Local confidence modifier

Available tile-level evidence forms a local modifier:

```text
C[m, k](x) = weighted_available_mean(
    leave-one-teacher-out agreement,
    out-of-fold expert-label agreement when available,
    ROI-derived tile presence agreement when available,
    delayed stop-gradient student agreement when enabled
)
```

Weights are normalized only over signals available for that tile. Student agreement is disabled during early training and remains an ablation-controlled optional term.

The final Level-2 reliability is:

```text
r[m, k](x) = epsilon + (1 - epsilon) * clamp(R[m, k] * C[m, k](x), 0, 1)
```

`epsilon` is only for numerical stability. It is not the global `alpha_min`; incorrect teacher-attribute signals may be reduced close to zero. Reliability values are independently computed for each attribute. Normalization across teachers occurs only when constructing the target:

```text
q[k](x) = sum_m r[m, k](x) * s[m, k](x)
          / clamp_min(sum_m r[m, k](x), epsilon)
```

`q[k]` remains a continuous soft target.

### 4.3 Collective-uncertainty gate

Teacher conflict and collective uncertainty are treated separately. Let the reliability-weighted teacher mean be `s_bar[k]` and the normalized weighted teacher variance be `V[k]`. For tiles without an expert or ROI anchor:

```text
U[k] = BinaryEntropy(s_bar[k]) * (1 - V[k])
g[k] = 1 - eta_u * U[k]
```

`U[k]` is high only when teachers agree on an intermediate response. Teacher conflict is handled by `r[m,k]`, not penalized again by `g[k]`. For anchored tiles, `g[k] = 1`.

### 4.4 Asymmetric Level-2 loss

The global student produces Level-2 logits `ell_global[k]`. Training uses logits directly:

```text
positive_term = -q[k] * log_sigmoid(ell_global[k])
negative_term = -m_neg[k] * (1 - q[k]) * log_sigmoid(-ell_global[k])

L_l2 = sum_{x,k} g[k] * (positive_term + negative_term)
       / clamp_min(sum_{x,k} g[k] * (q[k] + m_neg[k] * (1 - q[k])), epsilon)
```

`m_neg[k]` approaches 1 only with strong negative teacher evidence or a complete expert/ROI negative. Unanchored low teacher responses are not automatically treated as reliable negatives. This contract is implemented with numerically stable `BCEWithLogits`-equivalent operations.

## 5. Local Morphology Path

### 5.1 Architecture

`StudentEncoder` exposes the ViT patch-token map before global pooling. At 224 px with a 16 px patch size, the expected map is `14 x 14`. A separate patch projection head maps tokens into a patch semantic space.

Tile-level dynamic student prototypes and patch-space morphology prototypes are distinct:

- global student prototypes remain tile-embedding means and serve global `z_hcc` readout;
- patch prototypes are EMA/momentum prototypes built only from positive ROI tokens, with a stop-gradient update branch;
- teacher-space prototypes remain offline teacher-feature prototypes and are not subject to student EMA refresh.

An unconstrained learnable patch prototype is not used. If prototype parameters are made learnable, they require an explicit expert/ROI anchoring loss.

### 5.2 Local response

```text
a[j, k] = cosine(project_patch(h_patch[j]), p_patch[k]) / tau_patch[k]
ell_local[k] = TopQPool_j(a[j, k])
```

Smooth Top-Q or LogSumExp pooling is used instead of global average or hard max. Probabilities are produced only for metrics and target construction; all training losses consume logits.

### 5.3 Patch-head warm-up

The spatial branch has two sub-stages.

**B0 — head-only alignment**

- initialize compatible patch-head LayerNorm/linear weights from the global projector where possible;
- detach patch tokens and the global target so that only the patch head is updated;
- align mean-pooled projected patch tokens with the stop-gradient global embedding only to initialize representation scale; B0 does not create semantic pseudo-targets and cannot update `z_hcc`;
- do not propagate a randomly initialized patch-head gradient into the backbone.

**B1 — joint spatial ramp**

- start only after patch-head output scale and calibration are stable;
- gradually unfreeze patch-to-backbone gradients;
- ramp ROI loss, patch-prototype EMA updates, and local-to-global consistency smoothly rather than switching them on at one step.

Warm-up addresses optimization stability only. It does not establish localization validity.

### 5.4 Masked ROI supervision

Let `y_roi[j,k]` be the binary or soft Gaussian ROI target and `valid_roi[j,k]` indicate `positive` or completely reviewed `negative` tokens. `ignore` tokens have `valid_roi=0`:

```text
L_roi = sum_{j,k} valid_roi[j,k] * BCEWithLogits(a[j,k], y_roi[j,k])
        / clamp_min(sum_{j,k} valid_roi[j,k], epsilon)
```

Attribute-specific positive weighting may be estimated from the ROI training split, capped, and frozen before evaluation. No loss is produced for an attribute/sample pair with no valid tokens.

### 5.5 Directional local-to-global consistency

Directional consistency is enabled only for tile-attribute pairs with valid ROI spatial supervision. Let `consistency_roi[k]` indicate that attribute `k` has either annotated positive tokens or a completely reviewed spatial negative on the current tile. The ROI-supervised local response then reshapes the deployable global readout:

```text
L_local_to_global = sum_k consistency_roi[k] * BCEWithLogits(
    ell_global[k],
    stopgrad(sigmoid(ell_local[k]))
) / clamp_min(sum_k consistency_roi[k], epsilon)
```

Model confidence, teacher consensus, or tile-level prototype labels cannot activate `consistency_roi`; none of them establishes spatial validity. Non-ROI local predictions therefore never act as targets for `z_hcc`. The local branch is supervised by ROI labels rather than trained to reproduce unverified global context as spatial truth. Held-out localization metrics are required to exclude all-zero, all-one, and full-map broadcast shortcuts. Extending consistency to non-ROI pseudo-spatial targets is outside the final plan and would require a separate validated ablation.

## 6. ROI Annotation Protocol

ROI annotation is performed on a slide-diverse subset of the 3,000 prototype tiles after the loss-only gate passes.

The token-level schema is tri-state for every attribute:

- `positive`: visible morphology is present in the token;
- `negative`: the token was completely reviewed and morphology is absent;
- `ignore`: unreviewed, ambiguous, outside the valid tissue area, or surrounding a point/partial brush annotation.

Point and incomplete brush annotations never convert all unmarked tokens into negatives. The annotation interface records slide ID, institution, tile ID, attribute, geometry, review completeness, and annotator identity/version.

Sampling prioritizes rare attributes while controlling slide, institution, L1 state, and morphology diversity. Feasibility is judged per attribute by independent positive slide count and held-out positive regions. Attributes failing the frozen feasibility threshold remain tile-level only; their localization metrics are development diagnostics rather than external claims.

## 7. Training Schedule

Exact steps are selected from teacher-prior plateau and pilot stability. The initial full-scale schedule is:

```text
0-10,000       teacher-prior warm-up
10,000+        existing prototype intervention
11,000+        existing global teacher filtering
12,000-15,000  attribute-wise L2 filtering ramp, no student evidence
15,000+        optional delayed student-agreement ramp after ablation approval
```

For spatial experiments only:

```text
10,000+        B0 patch-head-only alignment
15,000+        B1 joint spatial ramp after Gate B checks
```

The global L1, feature, relation, and release embedding paths remain unchanged during the loss-only pilot.

## 8. Validation Design

### 8.1 Controlled comparisons

1. current global L2 response;
2. attribute-wise L2 filtering only;
3. attribute-wise filtering plus patch branch without ROI supervision;
4. full attribute-wise filtering plus ROI-supervised patch branch.

All comparisons use the same training tiles, external query/gallery, optimization budget, and model-selection policy. Runs use multiple random seeds. Differences are reported with paired, slide-clustered bootstrap confidence intervals.

### 8.2 Evidence hierarchy

**Mechanism evidence**

- external Level-2 macro average precision;
- per-attribute average precision;
- macro AUC as secondary ranking evidence;
- reliability calibration and coverage;
- ablation of student agreement, uncertainty gate, negative mask, and shrinkage prior.

**Primary representation evidence**

- existing blinded morphology retrieval protocol;
- precision@k, mean relevance@k, NDCG@k, model win rate, and paired confidence intervals;
- comparison with all individual teachers and current HCC-SemPath.

**Spatial mechanism evidence**

- pointing accuracy on held-out ROI tiles;
- soft-IoU or token-level F1 on completely reviewed regions;
- activation-area distribution, local/global logit drift, and degenerate-map rate.

**Non-degradation checks**

- L1 accuracy;
- blinded retrieval quality;
- teacher-prior retention;
- embedding stability, throughput, memory, and release-interface compatibility.

External L2 labels must be slide-independent from prototype construction. If taxonomy design or annotators overlap, a genuinely independent test cohort is retained for the final claim.

## 9. Decision Gates

### Gate A — before the loss-only pilot

- slide-level cross-fitting is implemented and tested;
- teacher-attribute reliability shrinkage and fallback rules are frozen;
- `r/q/g/m_neg` tensor shapes and mathematical contracts are frozen;
- logits-based asymmetric loss and effective-weight normalization are tested;
- L2 reliability floor is `epsilon`, separate from global `alpha_min`.

Gate A passes when attribute-wise filtering improves external L2 macro AP or a prespecified set of key attribute APs without material degradation of L1, retrieval, or teacher-prior retention.

### Gate B — before the spatial branch

- ROI schema and annotation completeness rules are frozen;
- attribute-specific ROI feasibility audit passes;
- patch-space EMA prototypes are separated from tile-level dynamic prototypes;
- B0 head-only warm-up is stable;
- local-to-global consistency direction and gradient stops are frozen;
- held-out localization split is fixed by slide.

B1 starts only if patch-head variance, positive rate, activation area, and global/local drift exclude degenerate solutions. The spatial contribution is retained only if it improves held-out localization and external Level-2 evidence without degrading the global representation.

## 10. Implementation Plan

### Phase A — loss-only attribute filtering

- add cross-fitted reliability asset generation;
- add teacher-attribute priors, shrinkage metadata, and stability diagnostics;
- extend adjudication tensors from `[batch]` to `[batch, attribute]` for the L2 path only;
- add weighted Level-2 target construction, uncertainty gate, negative mask, and asymmetric logits loss;
- preserve the existing scalar global alpha for feature/relation losses;
- add unit tests for missing signals, fallback rules, zero-effective-weight batches, and teacher/attribute normalization.

### Phase B — ROI data audit and annotation

- freeze ROI manifest and tri-state token-mask schema;
- produce per-attribute slide/region coverage reports;
- create slide-separated train/validation/test ROI manifests;
- validate annotation rendering and coordinate-to-token conversion.

### Phase C — patch-token branch

- expose patch tokens without changing release `encode()` behavior;
- add patch projector, EMA patch prototypes, Top-Q pooling, B0/B1 schedules, and directional local-to-global consistency;
- add degenerate-map diagnostics and held-out localization evaluation;
- export the patch branch only if needed for interpretation; the reusable global `z_hcc` remains the primary release representation.

## 11. Expected Contribution

The final method is:

> Global tissue-state organization with cross-fitted attribute-wise teacher adjudication and independently anchored local morphology evidence.

Its contribution is not a separate detector. It extends PAMT-D from tile-teacher adjudication to morphology-attribute adjudication, prevents circular reliability estimation, and introduces spatial evidence only when that evidence is independently anchored and demonstrably transferred into the deployable `z_hcc` representation.

— Final synthesis by Codex (GPT-5), incorporating the Claude and Antigravity cross-review record in `update.md`.
