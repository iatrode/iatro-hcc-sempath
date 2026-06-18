"""Compatibility shim — moved to the shared ``iatro_iac`` package.

The teacher-feature cache now lives in ``iatro_iac.adapters.features`` and is
re-exported from ``iatro_iac``. This module keeps existing
``hcc_sempath.io.feature_cache`` imports working. New code should import from
``iatro_iac`` directly.
"""

from __future__ import annotations

from iatro_iac.adapters.features import (  # noqa: F401
    FeatureCacheReader,
    build_teacher_feature_package,
    build_teacher_feature_package_from_feature_map,
    build_teacher_feature_package_from_tile_package,
    read_feature_package_records,
)
