# IatroCache v1

IatroCache (`.iac`) is an internal engineering cache contract for HCC-SemPath.
It exists to make offline image-tile caches and teacher-feature caches stable
enough for HCC-specific student embedding training.

IatroCache is not the scientific contribution of HCC-SemPath. It is not a
general-purpose pathology file format, and it is not a replacement for DICOM,
WSI storage, a WSI viewer backend, a pyramid image format, a database, an
archive format, or an experiment tracker.

The scientific line of the repository remains HCC-specific representation
learning, multi-teacher distillation, weak supervision / semantic shaping, and
student embeddings for downstream tasks. IAC is only the local cache contract
used before training.

## Top-Level Layout

IAC v1 uses a fixed header plus three byte segments:

```text
[Fixed Header Region]
[Slide Table Segment]
[Record Table Segment]
[Data Segment]
```

The fixed header region is 65,536 bytes. It starts with the IAC magic/version
fields and contains a UTF-8 JSON header. The slide and record tables are Arrow
IPC streams. The data segment is concatenated record payload bytes.

All `offset` values in the record table are relative to the start of the data
segment.

## Common Header

Every v1 package must include these fields after writing:

```text
format
version
payload_type
header_bytes
slide_table_offset
slide_table_length
record_table_offset
record_table_length
data_offset
data_length
num_slides
num_records
tile_width
tile_height
stride_x
stride_y
coordinate_mode = tile_grid
origin = top_left
checksum = crc32
created_by = hcc-sempath
```

`payload_type` is one of:

```text
image_tiles
teacher_features
```

`tile_x` and `tile_y` are tile-grid coordinates, not raw pixel coordinates:

```text
pixel_x = tile_x * stride_x
pixel_y = tile_y * stride_y
```

## Slide Table

The slide table schema is shared by image-tile packages and teacher-feature
packages:

```text
slide_idx   uint8
slide_id    string
patient_id  string
```

`slide_idx` is the compact key referenced by every record. A single package
therefore supports at most 255 slides in v1.

`split` is not part of the IAC v1 core schema. Experimental train/val/test
splits belong in an external experiment manifest.

## Record Tables

The identity and coordinate columns are shared by `image_tiles` and
`teacher_features`:

```text
slide_idx   uint8
tile_x      uint16
tile_y      uint16
tile_id     string
flags       uint8
```

For `image_tiles`, the record table also stores the per-tile variable-length
image payload location and checksum:

```text
offset      uint64
length      uint32
crc32       uint32
```

Field meanings:

- `slide_idx` points to `slide_table`.
- `tile_x` and `tile_y` are tile-grid coordinates.
- `tile_id` must be unique within a package.
- `tile_id` is the primary join key between image-tile packages and
  teacher-feature packages generated from the same tile set.
- `flags` is a one-byte bit field, normally `0` for clean retained records.
- `offset` is relative to the data segment.
- `length` is the payload byte length.
- `crc32` is the payload CRC32.

For `teacher_features`, the record table does not store per-feature
`offset/length/crc32`. The feature data segment contains one compressed matrix
block, and the record table defines the row order of that matrix.

Existing production image-tile packages may contain an extra `split` column from
the earlier implementation. Readers may tolerate it, but it is not part of the
minimal v1 core schema.

## `image_tiles`

Image-tile packages store one encoded image tile per record.

Additional header fields:

```text
codec = jxl
codec_params.lossless
codec_params.distance
codec_params.effort
codec_params.tile_color_space
codec_params.input_dtype
```

Current HCC-SemPath image-tile packages use JPEG XL payloads. Validators check
the common layout, CRC32, slide references, tile-id uniqueness, sampled JXL
decode, and decoded image size against `tile_width` and `tile_height`.

## `teacher_features`

Teacher-feature packages store one package-level feature matrix. Feature
packages are generated from the same tile set as an image-tile package and keep
the same slide identity, tile-grid coordinates, and `tile_id` join key. The
record table order is the matrix row order:

```text
matrix.shape = (num_records, feature_dim)
```

Additional header fields:

```text
teacher
feature_dim
dtype
feature_layout = matrix
compression = none | zstd | zlib | lzma
compression_level
matrix_offset
matrix_length
matrix_crc32
matrix_uncompressed_length
matrix_shape
```

`teacher` is the required source-teacher label. `feature_dim` and `dtype`
determine how the decompressed matrix is parsed:

```text
matrix_uncompressed_length = num_records * feature_dim * numpy.dtype(dtype).itemsize
```

The data segment stores the compressed matrix block. `matrix_offset` is relative
to the data segment and is normally `0`. `matrix_length` is the compressed byte
length. `matrix_crc32` is the CRC32 of the compressed matrix block.

Supported feature matrix compression values are deliberately small:

```text
none
zstd
zlib
lzma
```

The default writer uses `zstd` because it gives strong general-purpose
lossless compression with fast decompression. `zlib` and `lzma` are retained as
simple alternatives for compatibility or space-focused experiments. JXL is not
used for teacher features because teacher embeddings are not image payloads.

Teacher-feature payloads are raw contiguous matrix bytes after decompression.
IAC v1 does not store model revision, model path, preprocessing provenance,
hashes, experiment splits, or training provenance in the package header. If
needed, those belong in an external experiment manifest.

Validators check the common layout, CRC32, slide references, tile-id uniqueness,
non-empty `teacher`, positive `feature_dim`, valid NumPy `dtype`, and feature
matrix byte length / shape. They do not attempt image decoding for
`teacher_features`.

## CLI

Validate either package type:

```bash
hcc-sempath validate-package --package data/packages/tiles.iac
hcc-sempath validate-package --package data/packages/h_optimus_1.features.iac
```

Typical success output:

```text
package_valid type=image_tiles records=...
package_valid type=teacher_features teacher=... dim=... dtype=...
```
