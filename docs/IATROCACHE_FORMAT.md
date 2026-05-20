# IatroCache file format

IatroCache is a lightweight indexed payload container for offline medical image and feature cache construction.

This format is designed for project-controlled training and feature-cache pipelines. It is not a clinical image exchange format, not a DICOM replacement, not a WSI viewer backend, and not a multi-resolution pyramid format.

The first use case in this repository is transferring and decoding 20x / 224-pixel pathology tiles for H-optimus-1 teacher feature extraction. The format should remain general enough to support other disease domains, image modalities, tile payloads, teacher features, student features, and anchor-response payloads.

## Design goals

1. Store large numbers of variable-length payload records in a compact single file.
2. Avoid millions of small files.
3. Support efficient offline transfer to a server.
4. Support sequential and worker-parallel decoding for cache construction.
5. Keep the format simple and project-controlled.
6. Avoid unnecessary features such as pyramids, ROI viewing, transactional writes, or filesystem-like directory trees.
7. Keep compression assumptions explicit in the file header.

## Non-goals

IatroCache does not aim to provide:

- DICOM compatibility.
- Viewer random access.
- Multi-resolution WSI browsing.
- Clinical archive storage.
- A generic compressed archive format.
- A full embedded filesystem.
- General-purpose metadata querying.

## File extension

Recommended extension:

```text
.iac
```

Example files:

```text
train_tiles_000.iac
tcga_tiles_000.iac
teacher_features_000.iac
anchor_scores_000.iac
```

## Top-level layout

IatroCache v1 uses a thin indexed payload layout:

```text
[Fixed Header Region]
[Slide Table Segment]
[Record Table Segment]
[Data Segment]
```

The data segment contains raw concatenated payload bytes. The record table maps each logical record to its byte offset and length within the data segment.

## Fixed header region

The first 64 KiB of the file are reserved for the fixed header region.

Recommended structure:

```text
magic          8 bytes    ASCII: IATROC\0\1
header_len     uint32     length of valid JSON header bytes
version        uint32     format version, currently 1
header_json    remaining fixed header bytes, UTF-8 JSON, zero padded
```

The JSON header contains global format, payload, table, coordinate, and compression conventions.

Example header:

```json
{
  "format": "IatroCache",
  "version": 1,
  "payload_type": "image_tiles",
  "header_bytes": 65536,
  "slide_table_offset": 65536,
  "slide_table_length": 4096,
  "record_table_offset": 69632,
  "record_table_length": 1048576,
  "data_offset": 1118208,
  "num_slides": 42,
  "num_records": 2500000,
  "codec": "jxl",
  "codec_params": {
    "mode": "lossy_high_quality",
    "distance": null,
    "quality": null,
    "effort": null
  },
  "tile_width": 224,
  "tile_height": 224,
  "stride_x": 224,
  "stride_y": 224,
  "coordinate_mode": "tile_grid",
  "origin": "top_left",
  "slide_idx_dtype": "uint8",
  "tile_xy_dtype": "uint16",
  "offset_dtype": "uint64",
  "length_dtype": "uint32",
  "flags_dtype": "uint8",
  "checksum": "crc32",
  "max_slides_per_pack": 255,
  "created_by": "hcc-sempath",
  "created_at": null
}
```

### Compression convention

The compression convention is stored in the header because all image records in one pack should use the same codec and codec parameters.

For the first implementation, the preferred image payload is independently encoded JPEG XL tile bytes.

Recommended initial codec:

```json
{
  "codec": "jxl",
  "codec_params": {
    "mode": "lossy_high_quality_or_lossless_after_benchmark",
    "tile_color_space": "RGB",
    "input_dtype": "uint8"
  }
}
```

JPEG2000 or other codecs may be used later by changing the header convention, while preserving the same container layout.

## Slide table segment

The slide table maps compact per-record slide indices to slide-level metadata.

For v1, `slide_idx` should use `uint8`. Therefore a single IatroCache pack must contain no more than 255 slides.

Minimal slide table columns:

```text
slide_idx   uint8
slide_id    string
```

Optional slide table columns:

```text
patient_id  string
source      string
split       uint8 or string
```

The split field is optional. Experimental train/validation/test splits can also be stored outside the low-level cache file in a separate experiment manifest.

## Record table segment

The record table is intentionally compact. It stores only the information required to locate payload bytes and reconstruct the tile position within a slide.

Recommended image-tile record schema:

```text
slide_idx   uint8
tile_x      uint16
tile_y      uint16
offset      uint64
length      uint32
crc32       uint32
flags       uint8
```

### Field definitions

`slide_idx`: integer key into the slide table. A single pack supports up to 255 slides.

`tile_x`, `tile_y`: tile-grid coordinates, not raw pixel coordinates. Raw pixel coordinates are reconstructed as:

```text
pixel_x = tile_x * stride_x
pixel_y = tile_y * stride_y
```

`offset`: byte offset relative to the start of the data segment, not relative to the start of the file.

`length`: byte length of the payload record.

`crc32`: CRC32 checksum of the payload bytes. This is recommended for large transfer verification and damaged-record diagnosis.

`flags`: one-byte bit field. Suggested initial convention:

```text
0x00 normal
0x01 reserved_invalid
0x02 reserved_decode_failed
0x04 reserved_low_tissue
0x08 reserved_background
```

For a clean image-tile cache containing only retained valid tiles, `flags` should normally be zero.

## Data segment

The data segment is a raw concatenation of payload bytes:

```text
[payload_0][payload_1][payload_2]...[payload_N]
```

For image-tile packs, each payload is an encoded tile image, initially expected to be JPEG XL bytes.

For feature packs, each payload may be a fixed-size raw feature vector or another agreed binary representation. The same top-level layout can be reused with a different `payload_type`.

## Payload types

### image_tiles

Used for compressed RGB tile images.

Header requirements:

```text
payload_type = image_tiles
codec = jxl or another image codec
tile_width
tile_height
stride_x
stride_y
coordinate_mode = tile_grid
```

Record table requirements:

```text
slide_idx, tile_x, tile_y, offset, length, crc32, flags
```

### teacher_features

Used for teacher feature cache records.

Example header additions:

```json
{
  "payload_type": "teacher_features",
  "teacher": "H-optimus-1",
  "feature_dim": 1536,
  "dtype": "float16"
}
```

The record table may reuse the same coordinate fields when each feature corresponds to one tile.

### anchor_scores

Used for anchor-response vectors.

Example header additions:

```json
{
  "payload_type": "anchor_scores",
  "num_anchors": 8,
  "dtype": "float16"
}
```

The record table may reuse the same coordinate fields when each score vector corresponds to one tile.

## Table encoding

The table encoding should be simple and implementation-friendly.

Recommended v1 implementation:

```text
Arrow IPC table segment
```

Reason:

- It supports compact typed columns.
- It avoids writing a custom binary table parser at the beginning.
- It is easy to inspect and convert in Python.
- It is adequate for a project-controlled cache format.

If pyarrow becomes an unwanted dependency later, the table can be replaced by a fixed-width binary table while preserving the same logical schema.

## Pack sizing

Do not put all slides into one huge file.

Recommended pack policy:

- Keep each pack under 255 slides because `slide_idx` is `uint8`.
- Prefer pack sizes that are convenient for transfer and parallel processing.
- A practical target is tens of GB per pack, adjusted by storage and transfer conditions.

Dataset-level organization can use a simple manifest file:

```json
{
  "format": "IatroCacheDataset",
  "version": 1,
  "packs": [
    "train_tiles_000.iac",
    "train_tiles_001.iac"
  ]
}
```

## Read strategy

The primary read strategy is offline batch streaming:

1. Open an `.iac` pack.
2. Read and validate the fixed header.
3. Load the slide table and record table.
4. Iterate records in data-offset order.
5. Read payload bytes from `data_offset + offset` with `length` bytes.
6. Verify CRC32 if requested.
7. Decode image payloads or parse feature payloads.
8. Push decoded tiles or feature vectors into the cache-building pipeline.

This supports both sequential reading and worker-parallel reading by assigning different packs or slide groups to different workers.

## Write strategy

Recommended write procedure:

1. Reserve the fixed 64 KiB header region.
2. Write or buffer payload bytes into the data segment while collecting record metadata.
3. Build the slide table and record table.
4. Assemble final file as header + slide table + record table + data segment.
5. Fill the JSON header with final offsets, lengths, counts, codec parameters, and coordinate conventions.
6. Rewrite the fixed header region.
7. Run verification on the final pack.

A temporary build layout may be used internally:

```text
payloads.tmp
slide_table.tmp
record_table.tmp
final.iac
```

## Verification

A minimal verifier should check:

1. Magic string and version.
2. Header JSON validity.
3. Table offsets and lengths are within file bounds.
4. Record offsets and lengths are within the data segment.
5. `num_records` matches the record table.
6. `slide_idx` values are valid in the slide table.
7. CRC32 matches payload bytes for sampled or all records.
8. Image payloads can be decoded for sampled or all records.

## Minimal commands to implement

Initial tooling should stay small:

```text
iatrocache build-images
iatrocache inspect
iatrocache verify
iatrocache decode-benchmark
```

Later commands may include:

```text
iatrocache build-features
iatrocache export-manifest
iatrocache split-pack
```

## Current recommended default for this repository

For HCC-SemPath image tile transfer and teacher-cache construction:

```text
Format: IatroCache v1
Extension: .iac
Payload type: image_tiles
Codec: JXL after benchmarked parameter selection
Header: fixed 64 KiB with codec convention
Coordinates: tile-grid uint16 coordinates
Slide index: uint8 per pack
Data segment: concatenated compressed tile bytes
Record table: slide_idx, tile_x, tile_y, offset, length, crc32, flags
```

This keeps the format compact, project-specific, and easy to implement while avoiding unnecessary compatibility or viewer-oriented design.
