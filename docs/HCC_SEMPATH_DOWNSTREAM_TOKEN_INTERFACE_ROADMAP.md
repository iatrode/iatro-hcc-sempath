# HCC-SemPath Downstream Token Interface Roadmap

Status: proposed post-V2 roadmap. This document describes a public integration boundary, not a
committed V2 training requirement.

## 1. Objective

HCC-SemPath V2 is designed as an HCC-specific pathology vision tower. After V2 training and
evaluation are complete, the model will be frozen and exposed through a stable feature contract.
A lightweight downstream adapter may then convert those frozen pathology features into a fixed
number of continuous tokens for language, multimodal, or multi-expert systems.

The research question is:

> Can a lightweight adapter translate frozen, ROI-grounded HCC morphology features into compact
> continuous tokens that preserve tissue type, morphology composition, and spatial evidence for
> downstream language alignment?

This roadmap does not change the V2 scientific object, optimization target, or release criteria.

## 2. Core Hypothesis

The frozen V2 representation contains complementary levels of pathology evidence:

- `z_hcc` describes the global HCC morphology context;
- Level-1 readouts describe the dominant tissue state;
- Level-2 readouts describe non-exclusive morphology components;
- patch features and Level-2 spatial evidence maps describe where supported components occur.

A query-based adapter should preserve this structure more effectively than a single linear
projection of `z_hcc`. Level-1, Level-2, and spatial outputs can provide auxiliary semantic
constraints while the underlying V2 model remains frozen.

## 3. Component Boundary

The boundary is deliberately one-way:

```text
224 x 224 pathology tile or reviewed ROI
    -> frozen HCC-SemPath V2
    -> frozen pathology feature package
    -> downstream token adapter
    -> continuous pathology tokens
```

HCC-SemPath remains responsible for pathology representation and evidence localization. The
downstream consumer is responsible for token-space alignment, routing, memory, language
generation, and task-specific behavior. Downstream losses must not be required to reproduce the
released V2 representation.

## 4. Proposed Frozen Feature Contract

The post-V2 export should provide a versioned, model-agnostic feature package containing:

```text
embedding_norm       [B, D_global]
patch_features       [B, N_patch, D_patch]
patch_grid           [2]
l1_scores            [B, K_l1]
l2_scores            [B, K_l2]
l2_spatial_evidence  [B, K_l2, H_patch, W_patch]
input_validity       [B]
```

The export metadata should include:

- artifact format and semantic version;
- checkpoint and preprocessing checksums;
- input size, patch size, patch grid, and normalization;
- Level-1 and Level-2 names in fixed order;
- score semantics, calibration metadata, and supported thresholds;
- explicit distinction between spatial evidence, object detection, and segmentation.

Level-2 spatial outputs are attribute-specific evidence maps. They must not be represented as
bounding-box detection or pixel-accurate segmentation unless separately trained and validated for
that purpose.

## 5. Reference Token Adapter

The reference downstream adapter is a small learned-query module:

```text
learned pathology queries
    -> cross-attention over frozen patch features
    -> conditioning by z_hcc and semantic readouts
    -> projection to downstream hidden size
    -> pathology_tokens [B, K_token, D_target]
    -> pathology_mask   [B, K_token]
```

The adapter is not part of the HCC-SemPath encoder checkpoint. It may be implemented as a linear
or MLP baseline followed by a query-based cross-attention model. The query model is the primary
design because spatially sparse, coexisting morphologies should not be forced through one pooled
vector.

The adapter training objective is:

```text
L_adapter = L_language_alignment
          + lambda_l1 * L_l1_consistency
          + lambda_l2 * L_l2_consistency
          + lambda_spatial * L_spatial_grounding
```

- `L_language_alignment` aligns continuous pathology tokens with faithful local morphology text;
- `L_l1_consistency` preserves dominant tissue-state information;
- `L_l2_consistency` preserves multi-label morphology composition;
- `L_spatial_grounding` aligns adapter attention or token attribution with frozen V2 spatial
  evidence.

Only the adapter and its downstream consumer are updated. HCC-SemPath V2 remains frozen.

## 6. Language Supervision

Language targets should describe only the visible tile or reviewed ROI. A controlled description
may combine:

```text
Level-1 tissue state
+ supported Level-2 components
+ spatial distribution and extent
+ uncertainty wording
+ local-view boundary statement
```

The text-generation policy must distinguish positive evidence, negative complete review, and
unknown or unreviewed attributes. Low-confidence or unreviewed components must not be converted
into definitive negative statements.

Public releases may include schemas, controlled vocabulary, generation rules, and de-identified
example records. Private slides, patient identifiers, institution-specific paths, internal model
outputs, and non-redistributable annotations remain outside the repository.

## 7. Validation Design

The primary comparison is:

```text
pooled z_hcc linear adapter
versus
z_hcc + patch-feature query adapter with semantic constraints
```

Validation should cover:

1. Level-1 reconstruction accuracy from frozen adapter tokens;
2. Level-2 macro and per-attribute reconstruction performance;
3. agreement between token attribution and held-out spatial evidence;
4. morphology-description factuality, omission rate, and unsupported-attribute rate;
5. retrieval consistency between `z_hcc`, adapter tokens, and matched morphology text;
6. robustness to invalid, low-tissue, or out-of-domain inputs;
7. unchanged V2 outputs before and after downstream adapter training.

Evaluation splits must remain slide- or patient-independent as appropriate. Adapter model
selection must not consume the frozen external evidence reserved for V2 confirmatory evaluation.

## 8. Expected Contribution

The expected contribution is a stable bridge from an ROI-grounded HCC pathology representation
to continuous token interfaces without retraining the pathology tower. This enables HCC-SemPath
to operate as an independently validated pathology expert within larger language, multimodal, or
multi-expert systems while retaining a clear evidence boundary.

## 9. Implementation Gates

### Gate T0 — freeze V2

- complete the V2 annotation, training, and external validation gates;
- select and freeze one release checkpoint;
- prohibit downstream task loss from changing that checkpoint.

### Gate T1 — feature export

- implement the versioned feature package;
- verify numerical equality between direct inference and exported features;
- publish the schema and public-safe synthetic examples.

### Gate T2 — adapter pilot

- train the pooled baseline and query adapter from cached frozen features;
- verify Level-1, Level-2, spatial, and non-collapse constraints;
- select token count and target dimension before downstream evaluation.

### Gate T3 — downstream integration

- integrate only through the versioned token contract;
- report adapter-specific results separately from V2 representation results;
- retain explicit local-view and non-diagnostic claim boundaries.

## 10. Non-Goals

This roadmap does not define a patient-level pathology model, whole-slide diagnosis, staging,
prognosis, treatment recommendation, a general-purpose vision-language model, or the architecture
of any external consumer. Those capabilities require separate data, supervision, validation, and
governance.
