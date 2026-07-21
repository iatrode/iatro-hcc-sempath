from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from iatro.iac import build_pack, read_tables


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "upgrade_iac_v1_to_v2.py"
SPEC = importlib.util.spec_from_file_location("upgrade_iac_v1_to_v2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
upgrade = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = upgrade
SPEC.loader.exec_module(upgrade)


def _make_v1_tile_pack(path: Path) -> tuple[bytes, bytes]:
    slide_table = pa.table(
        {
            "slide_idx": pa.array([0], type=pa.uint8()),
            "slide_id": ["slide"],
            "patient_id": ["patient"],
        }
    )
    index_table = pa.table(
        {
            "slide_idx": pa.array([0, 0], type=pa.uint8()),
            "tile_x": pa.array([0, 1], type=pa.uint16()),
            "tile_y": pa.array([0, 0], type=pa.uint16()),
            "tile_id": ["tile-0", "tile-1"],
            "split": ["train", "train"],
            "flags": pa.array([0, 0], type=pa.uint8()),
        }
    )
    payloads = [b"first-jxl-payload", b"second-jxl-payload"]
    build_pack(
        path,
        {
            "payload_type": "image_tiles",
            "codec": "jxl",
            "codec_params": {"mode": "lossy", "lossless": False, "distance": 1.0, "effort": 7},
        },
        slide_table,
        index_table,
        payloads,
        overwrite=True,
    )
    raw = bytearray(path.read_bytes())
    header_length = struct.unpack_from("<I", raw, 8)[0]
    header = json.loads(raw[16 : 16 + header_length])
    data_before = bytes(raw[header["data_offset"] :])
    header["record_table_offset"] = header.pop("index_table_offset")
    header["record_table_length"] = header.pop("index_table_length")
    header["version"] = 1
    encoded = json.dumps(header, indent=2, sort_keys=True).encode("utf-8")
    fixed = bytearray(upgrade.HEADER_BYTES)
    fixed[:8] = upgrade.V1_MAGIC
    struct.pack_into("<I", fixed, 8, len(encoded))
    struct.pack_into("<I", fixed, 12, 1)
    fixed[16 : 16 + len(encoded)] = encoded
    raw[: upgrade.HEADER_BYTES] = fixed
    path.write_bytes(raw)
    return data_before, b"".join(payloads)


def test_header_only_upgrade_preserves_payload_and_is_idempotent(tmp_path: Path) -> None:
    package = tmp_path / "tiles.iac"
    data_before, expected_payloads = _make_v1_tile_pack(package)

    inspection = upgrade.inspect_package(package)
    assert inspection.state == "v1"
    assert inspection.payload_type == "image_tiles"
    upgrade.upgrade_one(inspection, transaction="in-place")

    raw = package.read_bytes()
    assert raw[:8] == upgrade.V2_MAGIC
    assert struct.unpack_from("<I", raw, 12)[0] == 2
    header, _, index = read_tables(package)
    assert header["version"] == 2
    assert "index_table_offset" in header
    assert "record_table_offset" not in header
    assert raw[header["data_offset"] :] == data_before
    assert b"".join(
        raw[header["data_offset"] + int(index["offset"][row].as_py()) :
            header["data_offset"] + int(index["offset"][row].as_py()) + int(index["length"][row].as_py())]
        for row in range(len(index))
    ) == expected_payloads

    second = upgrade.inspect_package(package)
    assert second.state == "v2"
    upgrade.upgrade_one(second, transaction="in-place")
    assert package.read_bytes() == raw


@pytest.mark.skipif(sys.platform != "darwin", reason="clonefile transaction requires macOS/APFS")
def test_clone_transaction_replaces_validated_package(tmp_path: Path) -> None:
    package = tmp_path / "tiles.iac"
    data_before, _ = _make_v1_tile_pack(package)
    old_inode = package.stat().st_ino

    upgrade.upgrade_one(upgrade.inspect_package(package), transaction="clone")

    header, _, _ = read_tables(package)
    assert header["version"] == 2
    assert package.stat().st_ino != old_inode
    assert package.read_bytes()[header["data_offset"] :] == data_before
    assert not list(tmp_path.glob(".*.iac-v2-upgrade-*"))


def test_rejects_compressed_matrix_feature_subformat(tmp_path: Path) -> None:
    package = tmp_path / "legacy.features.iac"
    _make_v1_tile_pack(package)
    fixed, _, _, _, header = upgrade._read_fixed_header(package)
    header["payload_type"] = "teacher_features"
    header["feature_layout"] = "matrix"
    encoded = json.dumps(header, indent=2, sort_keys=True).encode("utf-8")
    replacement = bytearray(upgrade.HEADER_BYTES)
    replacement[:8] = upgrade.V1_MAGIC
    struct.pack_into("<I", replacement, 8, len(encoded))
    struct.pack_into("<I", replacement, 12, 1)
    replacement[16 : 16 + len(encoded)] = encoded
    raw = bytearray(package.read_bytes())
    raw[: upgrade.HEADER_BYTES] = replacement
    package.write_bytes(raw)

    try:
        upgrade.inspect_package(package)
    except upgrade.UpgradeError as exc:
        assert "feature_layout=matrix" in str(exc)
    else:
        raise AssertionError("compressed matrix feature package was not rejected")
