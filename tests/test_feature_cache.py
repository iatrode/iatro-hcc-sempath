from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from hcc_sempath.datasets import DistillationTileDataset, validate_teacher_cache
from hcc_sempath.feature_cache import FeatureCacheReader, build_teacher_feature_package_from_manifest
from hcc_sempath.iatrocache import read_tables
from hcc_sempath.manifests import read_tile_manifest, write_tile_manifest


def test_teacher_feature_package_round_trip_and_dataset_read() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tile_dir = root / "tiles"
        feature_dir = root / "features"
        tile_dir.mkdir()
        feature_dir.mkdir()
        rows = []
        for idx in range(3):
            tile_id = f"s1_{idx:07d}"
            tile_path = tile_dir / f"{tile_id}.png"
            Image.new("RGB", (224, 224), (idx * 40, 30, 120)).save(tile_path)
            np.save(feature_dir / f"{tile_id}.npy", np.arange(4, dtype=np.float32) + idx)
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
        build_teacher_feature_package_from_manifest(
            manifest_path,
            feature_dir,
            package_path,
            teacher_name="toy",
        )

        header, _, table = read_tables(package_path)
        assert header["payload_type"] == "teacher_features"
        assert header["teacher"] == "toy"
        assert header["feature_dim"] == 4
        assert header["dtype"] == "float32"
        assert len(table) == 3

        reader = FeatureCacheReader(package_path)
        try:
            np.testing.assert_allclose(reader.read_feature("s1_0000002"), np.array([2, 3, 4, 5], dtype=np.float32))
        finally:
            reader.close()

        records = read_tile_manifest(manifest_path)
        validate_teacher_cache(records, feature_dir, expected_dim=4, teacher_cache_package_path=package_path)
        dataset = DistillationTileDataset(
            records,
            teacher_cache_dir=feature_dir,
            image_size=224,
            teacher_cache_package_path=package_path,
        )
        assert dataset[1]["teacher_feature"].tolist() == [1.0, 2.0, 3.0, 4.0]
