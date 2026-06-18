"""Compatibility shim — moved to the shared ``iatrocache`` package.

The WSI tile / JXL pipeline now lives in ``iatrocache.adapters.tiles`` and is
re-exported from ``iatrocache``. This module keeps existing
``hcc_sempath.io.tile_package`` imports working. New code should import from
``iatrocache`` directly.
"""

from __future__ import annotations

from iatrocache.adapters.tiles import (  # noqa: F401
    TilePackageReader,
    build_tile_package,
    build_tile_package_from_records,
    decode_jxl,
    decode_jxl_array,
    encode_jxl,
    encode_jxl_array,
    iter_package_tiles,
    read_package_manifest,
    read_package_metadata,
    _build_record_table,
    _build_slide_table,
    _ensure_unique_column,
    _ensure_unique_tile_ids,
    _infer_coordinate_stride,
    _require_image_tile_header,
    _slide_map,
    _to_tile_record,
)
