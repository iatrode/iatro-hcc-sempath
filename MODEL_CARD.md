---
license: cc-by-nc-nd-4.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - histopathology
  - hepatocellular-carcinoma
  - computational-pathology
  - spatial-morphometry
  - knowledge-distillation
---

# HCC-SemPath

HCC-SemPath is a compact pathology model for hepatocellular carcinoma (HCC)
histomorphologic classification and spatial component measurement. It uses a
DINOv2-S/14 student trained by prototype-adjudicated distillation from four
frozen pathology encoders and sparse expert supervision.

The gated release is the full-population model. Training checkpoints and
teacher-specific projection heads are not part of the inference release.

## Inputs and outputs

The released CLI accepts a 224-pixel image, a
`<name>.tile.path.iac` tile package, or a supported whole-slide image. Whole
slides are tissue segmented and tiled at a standardized 20x-equivalent scale
before inference.

For each retained tile, the model returns:

- probabilities for seven mutually exclusive histomorphologic classes; and
- two stride-7 spatial response grids for eleven tissue components: an
  abundance/area response and an instance-centre response.

The IAC writer preserves slide/package identity, tile coordinates, level,
MPP, orientation, grid shape, and stride in `<name>.pred.path.iac`, so dense
responses can be mapped back to the source field.

### Classification classes

1. HCC tumour, well differentiated
2. HCC tumour, moderately differentiated
3. HCC tumour, poorly differentiated
4. background liver
5. inflammatory/stromal tissue
6. haemorrhage/necrosis
7. artefact/contamination

### Spatial components

1. hepatocellular parenchyma
2. necrosis
3. haemorrhage
4. bile pigment
5. inflammatory cells
6. fibroblasts
7. fibrous stroma
8. steatosis/vacuolation
9. small vessels
10. large vessels
11. ductular/portal structures

## Model design and training data

The student observes 224x224-pixel fields at approximately 0.5 micrometres per
pixel. Its native 14-pixel patch spans approximately one immune-cell diameter;
overlapping stride-7 local windows provide the dense observation grid. Final
Transformer context and local features are fused by the spatial head.

Population training used 13,964,919 tissue-retained tiles from 928 whole-slide
images collected at three institutions. Four pathology foundation encoders
(GigaPath, H-optimus-1, UNI2-h, and Virchow2) supplied frozen representation
targets. Expert supervision comprised a 2,800-tile balanced classification
prototype bank and a separate 493-tile spatial set. The release does not
contain source slides, annotations, teacher weights, or teacher features.

## Internal checkpoint readout

The full-population checkpoint was selected using the prespecified internal
validation procedure. On its 1,183-tile expert classification bank, accuracy,
balanced accuracy, and macro F1 were 0.877, 0.874, and 0.872. On the 325-tile
spatial checkpoint-selection bank, tile-component macro F1 was 0.862 and macro
one-vs-rest AUROC was 0.948; abundance and instance false-positive rates on
explicitly negative regions were 0.0079 and 0.0019.

These are internal checkpoint-selection readouts. The manuscript's locked
external classification and component-specific spatial tests are reported
separately and supersede this section when released.

## Usage

Install HCC-SemPath, obtain access to the gated model repository, and download
the release:

```bash
python -m pip install iatro-hcc-sempath
hcc-sempath download
```

Run inference on one image, an IAC tile package, or a whole slide:

```bash
hcc-sempath infer \
  --input /path/to/case.svs \
  --output /path/to/predictions
```

A local release directory can be supplied with `--model`. A complete release
contains the model contract `config.json` and inference weights
`model.safetensors`, together with this model card and the licence.

## Intended use

HCC-SemPath is intended for research use in HCC tissue representation,
tile-level histomorphologic analysis, spatial composition measurement, and
downstream cohort-level association studies. Outputs are quantitative model
responses, not diagnoses or treatment recommendations.

The model was developed on H&E liver pathology. Performance outside this
stain, organ, acquisition range, or tumour setting has not been established.
Clinical deployment requires an independently validated workflow appropriate
to the target laboratory and jurisdiction.

## Licence and third-party models

The released model is distributed under CC BY-NC-ND 4.0 through a gated model
repository. Third-party software and the four teacher models remain governed
by their own licences and access terms. Receiving HCC-SemPath does not grant
access to, or redistribute, any teacher model.

## Citation

Please cite the accompanying manuscript, *Prototype-Adjudicated Multi-Teacher
Distillation for HCC Tissue Classification and Spatial Morphometry*. Formal
bibliographic metadata will be added when available.
