from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hcc_sempath.cli.view_iac import IacViewerData
from hcc_sempath.io.feature_cache import build_teacher_feature_package
from hcc_sempath.io.manifests import TileRecord
from hcc_sempath.io.tile_package import build_tile_package_from_records, encode_jxl_array


def _records() -> list[TileRecord]:
    return [
        TileRecord(
            tile_id=f"s1_{idx:07d}",
            patient_id="p1",
            slide_id="s1",
            tile_path=Path(f"tiles/s1_{idx:07d}.jxl"),
            x=x * 16,
            y=y * 16,
            split="train",
        )
        for idx, (x, y) in enumerate((x, y) for y in range(3) for x in range(3))
    ]


def test_iac_viewer_reads_image_map_and_5x5_window() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiles.iac"
        records = _records()
        payloads = []
        for idx, _record in enumerate(records):
            arr = np.full((16, 16, 3), idx * 20, dtype=np.uint8)
            payloads.append(encode_jxl_array(arr, lossless=True, distance=None, effort=1))
        build_tile_package_from_records(
            records,
            payloads,
            path,
            tile_width=16,
            tile_height=16,
            stride_x=16,
            stride_y=16,
            lossless=True,
            effort=1,
            overwrite=True,
        )

        data = IacViewerData(path)
        try:
            assert data.summary()["payload_type"] == "image_tiles"
            map_payload = data.map_payload("0")
            assert map_payload["count"] == 9
            nearest = data.nearest("0", 16.2, 16.1)["record"]
            assert nearest["grid_x"] == 1
            assert nearest["grid_y"] == 1
            window = data.image_window_at("0", 16.2, 16.1)
            assert len(window["cells"]) == 25
            assert sum(1 for cell in window["cells"] if cell["record"] is not None) == 9
            assert window["center"] == {"slide": "0", "x": 16, "y": 16, "grid_x": 1, "grid_y": 1}
            assert data.read_tile_png(nearest["row"]).startswith(b"\x89PNG")
        finally:
            data.close()


def test_iac_viewer_reads_feature_cache_as_heatmap() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "features.iac"
        records = _records()
        features = [np.arange(4, dtype=np.float32) + idx for idx in range(len(records))]
        build_teacher_feature_package(records, features, path, teacher_name="smoke", overwrite=True)

        data = IacViewerData(path)
        try:
            assert data.summary()["payload_type"] == "teacher_features"
            map_payload = data.map_payload("s1")
            assert map_payload["count"] == 9
            nearest = data.nearest("s1", 16.2, 16.1)["record"]
            assert nearest["x"] == 16
            assert nearest["y"] == 16
        finally:
            data.close()
