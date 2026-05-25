# HCC-SemPath Prototype Format / Prototype 格式

This document defines the public prototype package contract for HCC-SemPath.
Prototypes are runtime data, not hard-coded model classes.

本文档定义 HCC-SemPath 的公开 prototype package 约定。Prototype 是运行时数据，
不是写死在模型中的类别。

## Package

A prototype package is a `.pt` or `.pth` file loaded with `torch.load`.

```python
{
    "version": 1,
    "prototypes": Tensor[num_prototypes, dim],
    "names": list[str],
    "groups": list[str | None],          # optional
    "levels": list[int],
    "exclusive": list[bool],
    "thresholds": Tensor[num_prototypes], # optional
    "counts": list[int],                 # optional
    "source": dict,                      # optional public-safe metadata
}
```

Required fields:

- `version`: currently `1`.
- `prototypes`: 2D float tensor.
- `names`: unique prototype names.
- `levels`: prototype semantic level. Level `1` is the primary mutually
  exclusive state, such as tumor versus non-tumor / background tissue. Level `2`
  is a non-exclusive attribute, such as lymphocyte-rich, fibrotic stroma,
  necrosis, vascular context, or background liver change.
- `exclusive`: per-prototype exclusivity flag. Level-1 prototypes must be
  `True`; level-2 prototypes must be `False`.

Optional fields:

- `groups`: coarse prototype groups such as tumor morphology, background liver,
  stromal context, or immune microenvironment.
- `thresholds`: per-prototype confidence thresholds for weak-label or
  self-training workflows.
- `counts`: number of curated examples used to initialize each prototype.
- `source`: public-safe metadata such as builder, concept source, release name,
  or checksum. It should not contain patient identifiers or private WSI paths.

## Directory

A prototype directory can contain a manifest and a package:

```text
prototype_dir/
  prototype_manifest.yaml
  prototypes.pt
```

Example manifest:

```yaml
version: 1
prototype_file: prototypes.pt
source:
  release: hcc_semantic_v1
```

The manifest metadata is merged with the package payload. Package payload values
take precedence for tensor fields.

## Training Use

Prototype supervision is two-level:

- Level 1 is mutually exclusive. It should use softmax-style competition or
  cross-entropy over primary states.
- Level 2 is non-exclusive. A tile can match multiple level-2 prototypes at
  once, so it should use multi-label regression, BCE-style losses, or
  positive-unlabeled objectives.

Prototype response should also support teacher filtering: teacher signals that
conflict with HCC prototype responses can be down-weighted with bounded soft
reliability weights during distillation.
