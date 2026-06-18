"""Compatibility shim — moved to the shared ``iatrocache`` package.

Package validation now lives in ``iatrocache.validate``. This keeps existing
``hcc_sempath.io.validate_package`` imports (and the ``validate-package`` CLI
entry) working. New code should import from ``iatrocache.validate`` directly.
"""

from __future__ import annotations

from iatrocache.validate import (  # noqa: F401
    main,
    validate_package,
    _discover_packages,
    _format_valid_message,
    _require_columns,
    _sample_rows,
    _validate_common,
    _validate_image_tiles,
    _validate_record_payload_spans,
    _validate_teacher_features,
    _validate_unique_tile_ids,
)


if __name__ == "__main__":
    main()
