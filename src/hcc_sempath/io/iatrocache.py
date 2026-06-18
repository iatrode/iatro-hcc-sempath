"""Compatibility shim.

The IatroCache container format has been extracted to the standalone
``iatrocache`` package (../../iatrocache), so it can be shared across the
HCC-CAMoE projects (SemPath image/feature caches, Course clinical-text caches).

This module re-exports the format layer unchanged. Project adapters
(``tile_package``, ``feature_cache``) keep importing from here, so no call
sites changed. New code may import from ``iatrocache`` directly.
"""

from __future__ import annotations

from iatrocache import (  # noqa: F401
    FORMAT_VERSION,
    HEADER_BYTES,
    MAGIC,
    PackReader,
    build_pack,
    build_pack_data_segment,
    build_pack_data_segment_from_file,
    build_pack_streaming,
    iter_payloads,
    read_header,
    read_payload,
    read_tables,
)
