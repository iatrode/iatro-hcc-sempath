from __future__ import annotations

import tempfile
import zlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import pyarrow as pa
from PIL import Image

from hcc_sempath.io.feature_cache import (
    FeatureCacheReader,
    build_teacher_feature_package,
    build_teacher_feature_package_from_feature_map,
    build_teacher_feature_package_from_tile_package,
    read_feature_package_records,
)
from hcc_sempath.io.iatro_iac import read_tables
from hcc_sempath.io.iatro_iac import build_pack_data_segment
from hcc_sempath.io.manifests import read_tile_manifest, write_tile_manifest
from hcc_sempath.io.tile_package import build_tile_package
from hcc_sempath.io.validate_package import _validate_common, _validate_teacher_features


_CONVERT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_legacy_feature_iac.py"
_CONVERT_SPEC = importlib.util.spec_from_file_location("convert_legacy_feature_iac", _CONVERT_SCRIPT)
assert _CONVERT_SPEC is not None and _CONVERT_SPEC.loader is not None
_CONVERT_MODULE = importlib.util.module_from_spec(_CONVERT_SPEC)
_CONVERT_SPEC.loader.exec_module(_CONVERT_MODULE)
convert_legacy_feature_iac = _CONVERT_MODULE.convert_legacy_feature_iac
convert_legacy_feature_iac_tree = _CONVERT_MODULE.convert_legacy_feature_iac_tree


def test_teacher_feature_package_round_trip_and_dataset_read() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tile_dir = root / "tiles"
        tile_dir.mkdir()
        rows = []
        feature_by_tile_id = {}
        for idx in range(3):
            tile_id = f"s1_{idx:07d}"
            tile_path = tile_dir / f"{tile_id}.png"
            Image.new("RGB", (224, 224), (idx * 40, 30, 120)).save(tile_path)
            feature_by_tile_id[tile_id] = np.arange(4, dtype=np.float32) + idx
            rows.append(
                {
                    "tile_id": tile_id,
                    "patient_id": "p1",
                    "slide_id": "s1",
                    "tile_path": str(tile_path),
                    "x": idx * 224,
                    "y": 0,
                    "split": "train",
                }
            )
        manifest_path = root / "manifest.csv"
        package_path = root / "teacher_features.iac"
        write_tile_manifest(manifest_path, rows)
        records = read_tile_manifest(manifest_path)
        build_teacher_feature_package_from_feature_map(
            records,
            feature_by_tile_id,
            package_path,
            teacher_name="toy",
        )

        header, slide_table, table = read_tables(package_path)
        assert header["payload_type"] == "teacher_features"
        assert header["teacher"] == "toy"
        assert header["feature_dim"] == 4
        assert header["dtype"] == "float32"
        assert header["feature_record_bytes"] == 4 * 4
        assert header["data_length"] == 3 * 4 * 4
        assert "feature_layout" not in header
        assert "compression" not in header
        assert "matrix_length" not in header
        assert header["stride_x"] == 224
        assert header["stride_y"] == 1
        assert len(slide_table) == 1
        assert len(table) == 3
        assert table.column_names == ["slide_idx", "tile_x", "tile_y", "tile_id", "flags"]
        assert table.column("slide_idx").to_pylist() == [0, 0, 0]
        assert table.column("tile_x").to_pylist() == [0, 1, 2]
        assert table.column("tile_y").to_pylist() == [0, 0, 0]

        reader = FeatureCacheReader(package_path)
        try:
            np.testing.assert_allclose(reader.read_feature("s1_0000002"), np.array([2, 3, 4, 5], dtype=np.float32))
        finally:
            reader.close()

        restored = read_feature_package_records(package_path)
        assert [(r.tile_id, r.patient_id, r.slide_id, r.x, r.y, r.split) for r in restored] == [
            ("s1_0000000", "p1", "s1", 0, 0, ""),
            ("s1_0000001", "p1", "s1", 224, 0, ""),
            ("s1_0000002", "p1", "s1", 448, 0, ""),
        ]
        _validate_common(header, slide_table, table)
        rows_checked = _validate_teacher_features(str(package_path), header, table, max_payload=8)
        assert rows_checked == 3


def test_teacher_feature_package_from_tile_package_uses_tile_grid_header() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tile_dir = root / "tiles"
        tile_dir.mkdir()
        rows = []
        for idx, (x, y) in enumerate(((0, 0), (448, 0), (0, 448))):
            tile_id = f"s1_{idx:07d}"
            tile_path = tile_dir / f"{tile_id}.png"
            Image.new("RGB", (224, 224), (idx * 40, 30, 120)).save(tile_path)
            rows.append(
                {
                    "tile_id": tile_id,
                    "patient_id": "p1",
                    "slide_id": "s1",
                    "tile_path": str(tile_path),
                    "x": x,
                    "y": y,
                    "split": "train",
                }
            )
        manifest_path = root / "manifest.csv"
        tile_package_path = root / "tiles.iac"
        feature_package_path = root / "features.iac"
        write_tile_manifest(manifest_path, rows)
        build_tile_package(manifest_path, tile_package_path, stride_x=448, stride_y=448)

        features = [np.full((2,), idx, dtype=np.float32) for idx in range(3)]
        build_teacher_feature_package_from_tile_package(
            tile_package_path,
            features,
            feature_package_path,
            teacher_name="toy",
        )

        header, slide_table, table = read_tables(feature_package_path)
        assert header["tile_width"] == 224
        assert header["tile_height"] == 224
        assert header["stride_x"] == 448
        assert header["stride_y"] == 448
        assert header["coordinate_mode"] == "tile_grid"
        assert header["origin"] == "top_left"
        assert header["feature_record_bytes"] == 2 * 4
        assert len(slide_table) == 1
        assert table.column("tile_x").to_pylist() == [0, 1, 0]
        assert table.column("tile_y").to_pylist() == [0, 0, 1]

        restored = read_feature_package_records(feature_package_path)
        assert [(r.x, r.y) for r in restored] == [(0, 0), (448, 0), (0, 448)]


def test_teacher_feature_package_writes_fixed_length_records(tmp_path: Path) -> None:
    records = [read_tile_manifest_record("a", 0, 0), read_tile_manifest_record("b", 224, 0)]
    package_path = tmp_path / "features.iac"
    build_teacher_feature_package(
        records,
        [np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) + 10],
        package_path,
        teacher_name="toy",
    )

    header, slide_table, table = read_tables(package_path)
    assert header["feature_record_bytes"] == 16
    assert header["data_length"] == 32
    _validate_common(header, slide_table, table)
    assert _validate_teacher_features(str(package_path), header, table, max_payload=0) == 2
    reader = FeatureCacheReader(package_path)
    try:
        np.testing.assert_allclose(reader.read_feature("b"), np.array([10, 11, 12, 13], dtype=np.float32))
    finally:
        reader.close()


def test_teacher_feature_package_rejects_duplicate_tile_id() -> None:
    records = [
        read_tile_manifest_record("dup", 0, 0),
        read_tile_manifest_record("dup", 224, 0),
    ]
    with pytest.raises(ValueError, match="duplicate tile_id"):
        build_teacher_feature_package(
            records,
            [np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)],
            "unused.iac",
            teacher_name="toy",
        )


def test_teacher_feature_package_rejects_feature_count_and_dim_mismatch(tmp_path: Path) -> None:
    records = [read_tile_manifest_record("a", 0, 0), read_tile_manifest_record("b", 224, 0)]
    with pytest.raises(ValueError, match="feature count mismatch"):
        build_teacher_feature_package(
            records,
            [np.zeros(2, dtype=np.float32)],
            tmp_path / "count.iac",
            teacher_name="toy",
        )
    with pytest.raises(ValueError, match="inconsistent feature dim"):
        build_teacher_feature_package(
            records,
            [np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32)],
            tmp_path / "dim.iac",
            teacher_name="toy",
        )


def test_convert_legacy_compressed_matrix_package_to_fixed_records(tmp_path: Path) -> None:
    records = [read_tile_manifest_record("a", 0, 0), read_tile_manifest_record("b", 224, 0)]
    slide_table = pa.table(
        {
            "slide_idx": pa.array(np.array([0], dtype=np.uint8), type=pa.uint8()),
            "slide_id": ["s1"],
            "patient_id": ["p1"],
        }
    )
    record_table = pa.table(
        {
            "slide_idx": pa.array(np.array([0, 0], dtype=np.uint8), type=pa.uint8()),
            "tile_x": pa.array(np.array([0, 1], dtype=np.uint16), type=pa.uint16()),
            "tile_y": pa.array(np.array([0, 0], dtype=np.uint16), type=pa.uint16()),
            "tile_id": [record.tile_id for record in records],
            "flags": pa.array(np.zeros(2, dtype=np.uint8), type=pa.uint8()),
        }
    )
    matrix = np.stack([np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) + 10])
    raw = matrix.tobytes(order="C")
    legacy_path = tmp_path / "legacy.features.iac"
    converted_path = tmp_path / "converted.features.iac"
    build_pack_data_segment(
        legacy_path,
        {
            "payload_type": "teacher_features",
            "teacher": "toy",
            "feature_dim": 4,
            "dtype": "float32",
            "feature_layout": "matrix",
            "compression": "none",
            "compression_level": None,
            "matrix_offset": 0,
            "matrix_length": len(raw),
            "matrix_crc32": zlib.crc32(raw) & 0xFFFFFFFF,
            "matrix_uncompressed_length": len(raw),
            "matrix_shape": [2, 4],
            "tile_width": 224,
            "tile_height": 224,
            "stride_x": 224,
            "stride_y": 224,
            "coordinate_mode": "tile_grid",
            "origin": "top_left",
            "slide_idx_dtype": "uint8",
            "tile_xy_dtype": "uint16",
            "flags_dtype": "uint8",
            "checksum": "crc32",
            "created_by": "hcc-sempath",
        },
        slide_table,
        record_table,
        raw,
    )

    convert_legacy_feature_iac(legacy_path, converted_path)

    header, _, _ = read_tables(converted_path)
    assert header["feature_record_bytes"] == 16
    assert header["data_length"] == 32
    assert "feature_layout" not in header
    assert "compression" not in header
    reader = FeatureCacheReader(converted_path)
    try:
        np.testing.assert_allclose(reader.read_feature("b"), np.array([10, 11, 12, 13], dtype=np.float32))
    finally:
        reader.close()


def test_convert_legacy_feature_iac_tree_preserves_relative_paths(tmp_path: Path) -> None:
    input_root = tmp_path / "legacy"
    output_root = tmp_path / "fixed"
    nested = input_root / "teacher"
    nested.mkdir(parents=True)
    legacy_path = nested / "slide_a.toy.features.iac"
    converted_path = output_root / "teacher" / "slide_a.toy.features.iac"
    build_teacher_feature_package(
        [read_tile_manifest_record("a", 0, 0)],
        [np.arange(4, dtype=np.float32)],
        tmp_path / "new_format.features.iac",
        teacher_name="toy",
    )

    slide_table = pa.table(
        {
            "slide_idx": pa.array(np.array([0], dtype=np.uint8), type=pa.uint8()),
            "slide_id": ["s1"],
            "patient_id": ["p1"],
        }
    )
    record_table = pa.table(
        {
            "slide_idx": pa.array(np.array([0], dtype=np.uint8), type=pa.uint8()),
            "tile_x": pa.array(np.array([0], dtype=np.uint16), type=pa.uint16()),
            "tile_y": pa.array(np.array([0], dtype=np.uint16), type=pa.uint16()),
            "tile_id": ["a"],
            "flags": pa.array(np.zeros(1, dtype=np.uint8), type=pa.uint8()),
        }
    )
    raw = np.arange(4, dtype=np.float32).reshape(1, 4).tobytes(order="C")
    build_pack_data_segment(
        legacy_path,
        {
            "payload_type": "teacher_features",
            "teacher": "toy",
            "feature_dim": 4,
            "dtype": "float32",
            "feature_layout": "matrix",
            "compression": "none",
            "compression_level": None,
            "matrix_offset": 0,
            "matrix_length": len(raw),
            "matrix_crc32": zlib.crc32(raw) & 0xFFFFFFFF,
            "matrix_uncompressed_length": len(raw),
            "matrix_shape": [1, 4],
            "tile_width": 224,
            "tile_height": 224,
            "stride_x": 224,
            "stride_y": 224,
            "coordinate_mode": "tile_grid",
            "origin": "top_left",
            "slide_idx_dtype": "uint8",
            "tile_xy_dtype": "uint16",
            "flags_dtype": "uint8",
            "checksum": "crc32",
            "created_by": "hcc-sempath",
        },
        slide_table,
        record_table,
        raw,
    )

    converted = convert_legacy_feature_iac_tree(input_root, output_root)

    assert converted == [converted_path]
    header, _, _ = read_tables(converted_path)
    assert header["feature_record_bytes"] == 16
    assert "feature_layout" not in header


def read_tile_manifest_record(tile_id: str, x: int, y: int):
    from hcc_sempath.io.manifests import TileRecord

    return TileRecord(
        tile_id=tile_id,
        patient_id="p1",
        slide_id="s1",
        tile_path=Path(f"tiles/{tile_id}.png"),
        x=x,
        y=y,
        split="train",
    )
