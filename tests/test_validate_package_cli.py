from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from hcc_sempath.io.feature_cache import build_teacher_feature_package_from_tile_package
from hcc_sempath.io.manifests import write_tile_manifest
from hcc_sempath.io.tile_package import build_tile_package
from hcc_sempath.io.validate_package import main, validate_package


def _write_tile_and_feature_packages(root):
    tile_dir = root / "tiles"
    tile_dir.mkdir()
    rows = []
    for idx in range(2):
        tile_id = f"slide_a_{idx:07d}"
        tile_path = tile_dir / f"{tile_id}.png"
        Image.new("RGB", (16, 16), (idx * 40, 50, 80)).save(tile_path)
        rows.append(
            {
                "tile_id": tile_id,
                "patient_id": "p1",
                "slide_id": "slide_a",
                "tile_path": str(tile_path),
                "x": idx * 16,
                "y": 0,
                "split": "train",
            }
        )
    manifest_path = root / "slide_a.csv"
    tile_package = root / "slide_a.tiles.iac"
    feature_package = root / "slide_a.toy.features.iac"
    write_tile_manifest(manifest_path, rows)
    build_tile_package(manifest_path, tile_package, stride_x=16, stride_y=16)
    build_teacher_feature_package_from_tile_package(
        tile_package,
        [np.full((4,), idx, dtype=np.float32) for idx in range(2)],
        feature_package,
        teacher_name="toy",
    )
    return tile_package, feature_package


def test_validate_package_accepts_image_tiles_and_teacher_features(tmp_path):
    tile_package, feature_package = _write_tile_and_feature_packages(tmp_path)

    tile_result = validate_package(tile_package, max_decode=1, max_crc=1)
    feature_result = validate_package(feature_package, max_decode=1)

    assert tile_result["type"] == "image_tiles"
    assert tile_result["records"] == 2
    assert feature_result["type"] == "teacher_features"
    assert feature_result["teacher"] == "toy"
    assert feature_result["dim"] == 4


def test_validate_package_directory_scans_tiles_and_features_with_progress(monkeypatch, capsys, tmp_path):
    _write_tile_and_feature_packages(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hcc-sempath validate-package",
            "--input",
            str(tmp_path),
            "--max-decode",
            "1",
            "--max-crc",
            "1",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "type=image_tiles" in captured.out
    assert "type=teacher_features" in captured.out
    assert "validation_summary total=2 ok=2 failed=0" in captured.out


def test_validate_package_directory_reports_failures_after_scan(monkeypatch, capsys, tmp_path):
    _write_tile_and_feature_packages(tmp_path)
    (tmp_path / "broken.iac").write_text("bad package", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["hcc-sempath validate-package", "--input", str(tmp_path), "--max-decode", "1"])

    with pytest.raises(SystemExit, match="validation_failed count=1"):
        main()

    captured = capsys.readouterr()
    assert "package_invalid" in captured.out
    assert "validation_summary total=3 ok=2 failed=1" in captured.out

