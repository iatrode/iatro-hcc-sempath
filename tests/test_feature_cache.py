from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iatro.iac.adapters.features import (
    FeatureCacheReader,
    build_teacher_feature_package,
    build_teacher_feature_package_from_feature_map,
    build_teacher_feature_package_from_tile_package,
    read_feature_package_records,
)
from iatro.iac import read_tables
from iatro.iac.adapters.manifests import read_tile_manifest, write_tile_manifest
from iatro.iac.adapters.tiles import build_tile_package
from iatro.iac.adapters.validate import validate_package



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
        validation = validate_package(package_path, max_decode=8)
        rows_checked = validation["rows_checked"]
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
    validation = validate_package(package_path, max_decode=0)
    assert validation["rows_checked"] == 2
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


def read_tile_manifest_record(tile_id: str, x: int, y: int):
    from iatro.iac.adapters.manifests import TileRecord

    return TileRecord(
        tile_id=tile_id,
        patient_id="p1",
        slide_id="s1",
        tile_path=Path(f"tiles/{tile_id}.png"),
        x=x,
        y=y,
        split="train",
    )
