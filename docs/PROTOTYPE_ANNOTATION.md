# Prototype Annotation UI

This tool collects physician review labels for the HCC-SemPath two-level
prototype taxonomy. L1 is a mutually exclusive primary tile type. L2 is a
non-mutually-exclusive attribute set that can cross L1 types.

本工具用于采集医师对 HCC-SemPath 双层 prototype 体系的 tile 级注释。L1 是互斥的主导
tile 类型，L2 是非互斥属性，同一个 tile 可以有多个 L2 标签，并且 L2 可以跨 L1 类型存在。

## Command

```bash
hcc-sempath annotate-prototypes \
  --input data/packages \
  --state annotations/hcc_prototype_review.json
```

`--input` accepts either a single image-tile `.iac` package or a directory. A
directory is scanned recursively. If packages are stored as
`<input>/<dataset_name>/<package>.iac`, the first directory level is shown as the
dataset name in the package list and written to the annotation output.

`--state` is the resumable JSON state file. The UI also writes a CSV file next
to it by replacing the suffix with `.csv`.

## Review Flow

The browser UI has three working areas:

- left: discovered IAC package list with dataset name and annotated/total count;
- center: current tile plus a downsampled thumbnail of the current IAC package;
- right: progress, L1 single-select buttons, L2 multi-select buttons, and save
  controls.

Clicking the thumbnail jumps to the nearest tile at that spatial location. After
`Save + next`, or when no tile is selected, the UI randomly selects another
unannotated tile from the same IAC package. Already annotated tiles are excluded
from random selection and are outlined on the thumbnail.

## Output Contract

The JSON state file stores:

```json
{
  "version": 1,
  "input_path": "/absolute/path/to/input",
  "l1_prototypes": ["HCC-tumor"],
  "l2_prototypes": ["necrosis-present"],
  "annotations": {
    "dataset/slide.iac::tile_id::123,456": {
      "dataset": "dataset",
      "iac": "dataset/slide.iac",
      "iac_path": "/absolute/path/to/dataset/slide.iac",
      "tile_id": "tile_id",
      "row": 0,
      "slide": "slide_id",
      "x": 123,
      "y": 456,
      "l1": "HCC-tumor",
      "l2": ["necrosis-present"]
    }
  }
}
```

The CSV export uses one row per annotated tile:

```text
dataset,iac,iac_path,tile_id,row,slide,x,y,l1,l2
```

`l2` is serialized as a semicolon-separated list. The trace key is intentionally
based on IAC relative path, tile id, and spatial coordinate so annotations remain
auditable against both the package and tile table.
