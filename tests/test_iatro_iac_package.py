from __future__ import annotations

import tempfile
import zlib
from pathlib import Path

from PIL import Image

from iatro.iac import PackReader, read_tables
from iatro.iac.adapters.manifests import write_tile_manifest
from iatro.iac.adapters.tiles import build_tile_package, read_package_manifest
from iatro.iac.adapters.validate import validate_package


def _write_manifest_with_tiles(root: Path, coords: tuple[tuple[int, int], ...]) -> tuple[Path, list[dict]]:
    tile_dir = root / "tiles"
    tile_dir.mkdir()
    rows = []
    for idx, (x, y) in enumerate(coords):
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
    write_tile_manifest(manifest_path, rows)
    return manifest_path, rows


def test_package_defaults_to_tile_size_stride_and_writes_crc() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_path, rows = _write_manifest_with_tiles(root, ((0, 0), (224, 0), (0, 224), (224, 224)))
        package_path = root / "tiles.iac"
        build_tile_package(manifest_path, package_path)

        header, _, record_table = read_tables(package_path)
        assert header["tile_width"] == 224
        assert header["tile_height"] == 224
        assert header["stride_x"] == 224
        assert header["stride_y"] == 224
        assert header["coordinate_mode"] == "tile_grid"
        assert record_table.column("tile_x").to_pylist() == [0, 1, 0, 1]
        assert record_table.column("tile_y").to_pylist() == [0, 0, 1, 1]

        reader = PackReader(package_path)
        try:
            for row in range(len(record_table)):
                expected_crc = record_table.column("crc32")[row].as_py()
                payload = reader.read_payload(row)
                assert zlib.crc32(payload) & 0xFFFFFFFF == expected_crc
        finally:
            reader.close()

        restored = read_package_manifest(package_path)
        assert [(r.tile_id, r.x, r.y) for r in restored] == [(r["tile_id"], r["x"], r["y"]) for r in rows]

        validation = validate_package(package_path, max_decode=8, max_crc=0)
        crc_checked = validation["crc_checked"]
        decoded = validation["decoded"]
        assert crc_checked == 4
        assert decoded == 4


def test_package_accepts_explicit_non_tile_stride() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_path, rows = _write_manifest_with_tiles(root, ((0, 0), (448, 0), (0, 448), (448, 448)))
        package_path = root / "tiles.iac"
        build_tile_package(manifest_path, package_path, stride_x=448, stride_y=448)

        header, _, record_table = read_tables(package_path)
        assert header["tile_width"] == 224
        assert header["tile_height"] == 224
        assert header["stride_x"] == 448
        assert header["stride_y"] == 448
        assert record_table.column("tile_x").to_pylist() == [0, 1, 0, 1]
        assert record_table.column("tile_y").to_pylist() == [0, 0, 1, 1]

        restored = read_package_manifest(package_path)
        assert [(r.tile_id, r.x, r.y) for r in restored] == [(r["tile_id"], r["x"], r["y"]) for r in rows]


def test_package_infers_non_tile_stride_from_manifest_coordinates() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_path, rows = _write_manifest_with_tiles(root, ((0, 0), (448, 0), (0, 448), (448, 448)))
        package_path = root / "tiles.iac"
        build_tile_package(manifest_path, package_path)

        header, _, record_table = read_tables(package_path)
        assert header["stride_x"] == 448
        assert header["stride_y"] == 448
        assert record_table.column("tile_x").to_pylist() == [0, 1, 0, 1]
        assert record_table.column("tile_y").to_pylist() == [0, 0, 1, 1]

        restored = read_package_manifest(package_path)
        assert [(r.tile_id, r.x, r.y) for r in restored] == [(r["tile_id"], r["x"], r["y"]) for r in rows]
