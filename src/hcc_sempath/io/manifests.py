"""Compatibility shim — moved to the shared ``iatro.iac`` package.

``TileRecord`` and the tile-manifest helpers now live in ``iatro.iac``; this
re-exports them so existing ``hcc_sempath.io.manifests`` imports keep working.
New code should import from ``iatro.iac`` directly.
"""

from __future__ import annotations

from iatro.iac.manifests import (  # noqa: F401
    REQUIRED_TILE_COLUMNS,
    TILE_COLUMNS,
    TileRecord,
    read_tile_manifest,
    write_tile_manifest,
)
