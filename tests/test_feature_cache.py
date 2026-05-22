from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from hcc_sempath.io.feature_cache import FeatureCacheReader, build_teacher_feature_package_from_feature_map
from hcc_sempath.io.iatrocache import read_tables
from hcc_sempath.io.manifests import read_tile_manifest, write_tile_manifest
from hcc_sempath.training.datasets import DistillationTileDataset, validate_teacher_cache


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

        validate_teacher_cache(records, None, expected_dim=4, teacher_cache_package_path=package_path)
        dataset = DistillationTileDataset(
            records,
            teacher_cache_dir=None,
            image_size=224,
            teacher_cache_package_path=package_path,
        )
        assert dataset[1]["teacher_feature"].tolist() == [1.0, 2.0, 3.0, 4.0]
