from __future__ import annotations

import numpy as np
from PIL import Image

from hcc_sempath.modeling.roi_plan import (
    _rank_nucleus_points,
    _saliency_regions,
    detect_hematoxylin_centers,
)


def test_hematoxylin_detector_finds_dark_nucleus_like_centers() -> None:
    image = np.full((96, 96, 3), 235, dtype=np.uint8)
    image[20:25, 30:35] = [45, 30, 75]
    image[22, 32] = [20, 10, 45]
    image[65:70, 70:75] = [55, 35, 85]
    image[67, 72] = [25, 15, 50]

    centers = detect_hematoxylin_centers(Image.fromarray(image))

    assert len(centers) >= 2
    assert any(abs(item["x"] - 32 / 96) < 0.05 and abs(item["y"] - 22 / 96) < 0.05 for item in centers)
    assert any(abs(item["x"] - 72 / 96) < 0.05 and abs(item["y"] - 67 / 96) < 0.05 for item in centers)


def test_feature_saliency_ranks_nuclei_and_spatial_regions() -> None:
    saliency = np.zeros((16, 16), dtype=np.float32)
    saliency[4, 5] = 1.0
    saliency[12, 13] = 0.8
    candidates = [
        {"x": 5.5 / 16, "y": 4.5 / 16, "stain": 0.8},
        {"x": 1.5 / 16, "y": 1.5 / 16, "stain": 0.9},
    ]

    points = _rank_nucleus_points(candidates, saliency, limit=1, stain_weight=0.25)
    regions = _saliency_regions(saliency, limit=2)

    assert points[0]["x"] == candidates[0]["x"]
    assert len(regions) == 2
    assert regions[0]["confidence"] == 1.0
