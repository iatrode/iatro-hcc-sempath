"""Compatibility shim — moved to the shared ``iatrocache`` package.

The teacher-feature cache now lives in ``iatrocache.adapters.features`` and is
re-exported from ``iatrocache``. This module keeps existing
``hcc_sempath.io.feature_cache`` imports working. New code should import from
``iatrocache`` directly.
"""

from __future__ import annotations

from iatrocache.adapters.features import (  # noqa: F401
    FeatureCacheReader,
    build_teacher_feature_package,
    build_teacher_feature_package_from_feature_map,
    build_teacher_feature_package_from_tile_package,
    read_feature_package_records,
)
