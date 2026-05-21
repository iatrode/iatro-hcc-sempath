from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from hcc_sempath.manifests import write_tile_manifest
from hcc_sempath.qc import render_tile_package_qc
from hcc_sempath.tile_package import build_tile_package


def test_render_tile_package_qc_outputs_nonblank_contact_sheet() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tile_dir = root / "tiles"
        tile_dir.mkdir()
        rows = []
        for idx, color in enumerate(((180, 20, 20), (20, 180, 20), (20, 20, 180), (150, 150, 20))):
            tile_id = f"s1_{idx:07d}"
            tile_path = tile_dir / f"{tile_id}.png"
            Image.new("RGB", (224, 224), color).save(tile_path)
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
        package_path = root / "tiles.iac"
        qc_path = root / "qc.png"
        write_tile_manifest(manifest_path, rows)
        build_tile_package(manifest_path, package_path)
        render_tile_package_qc(package_path, qc_path, max_tiles=4, thumb_size=96)

        assert qc_path.exists()
        with Image.open(qc_path) as image:
            assert image.size[0] > 96
            assert image.size[1] > 96
            assert np.asarray(image.convert("RGB")).std() > 0
