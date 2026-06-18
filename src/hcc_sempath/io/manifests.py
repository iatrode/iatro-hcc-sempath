"""Compatibility shim — moved to the shared ``iatrocache`` package.

``TileRecord`` and the tile-manifest helpers now live in ``iatrocache``; this
re-exports them so existing ``hcc_sempath.io.manifests`` imports keep working.
New code should import from ``iatrocache`` directly.
"""

from __future__ import annotations

from iatrocache.manifests import (  # noqa: F401
    REQUIRED_TILE_COLUMNS,
    TILE_COLUMNS,
    TileRecord,
    read_tile_manifest,
    write_tile_manifest,
)
