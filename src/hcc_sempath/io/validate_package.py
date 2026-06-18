"""Compatibility shim — moved to the shared ``iatro_iac`` package.

Package validation now lives in ``iatro_iac.validate``. This keeps existing
``hcc_sempath.io.validate_package`` imports (and the ``validate-package`` CLI
entry) working. New code should import from ``iatro_iac.validate`` directly.
"""

from __future__ import annotations

from iatro_iac.validate import (  # noqa: F401
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
