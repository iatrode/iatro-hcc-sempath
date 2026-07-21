from __future__ import annotations

import numpy as np
from PIL import Image

from hcc_sempath.modeling.roi_plan import (
    _estimate_seed_spacing,
    _local_color_descriptor,
    _rank_similar_centers,
    _relative_scores,
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


def test_local_similarity_ranks_centers_and_excludes_seed_neighbors() -> None:
    centers = [(0.10, 0.10), (0.50, 0.50), (0.80, 0.80), (0.82, 0.82)]
    scores = _relative_scores(np.asarray([0.2, 0.9, 0.7, 0.6], dtype=np.float32))

    ranked = _rank_similar_centers(centers, scores, [(0.49, 0.49)], limit=3)

    assert ranked[0][:2] == (0.80, 0.80)
    assert all(abs(x - 0.50) > 0.02 or abs(y - 0.50) > 0.02 for x, y, _ in ranked)
    assert ranked[0][2] == 1.0


def test_seed_spacing_tracks_selected_class_target_size() -> None:
    image = np.full((96, 96, 3), 235, dtype=np.uint8)
    yy, xx = np.ogrid[:96, :96]
    image[(xx - 24) ** 2 + (yy - 48) ** 2 <= 3**2] = [35, 20, 65]
    image[(xx - 70) ** 2 + (yy - 48) ** 2 <= 7**2] = [35, 20, 65]
    pil_image = Image.fromarray(image)

    small_normalized, small_pixels = _estimate_seed_spacing(
        pil_image, [(24 / 96, 48 / 96)]
    )
    large_normalized, large_pixels = _estimate_seed_spacing(
        pil_image, [(70 / 96, 48 / 96)]
    )

    assert 5 <= small_pixels < large_pixels <= 18
    assert small_normalized == small_pixels / 96
    assert large_normalized == large_pixels / 96


def test_local_similarity_uses_requested_class_spacing() -> None:
    centers = [(0.40, 0.40), (0.43, 0.40), (0.60, 0.60)]
    scores = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)

    ranked = _rank_similar_centers(
        centers, scores, [], limit=3, min_distance=0.05
    )

    assert [item[:2] for item in ranked] == [(0.40, 0.40), (0.60, 0.60)]


def test_local_color_descriptor_changes_with_patch_stain() -> None:
    image = np.full((64, 64, 3), 230, dtype=np.float32)
    image[8:25, 8:25] = [45, 30, 80]

    dark = _local_color_descriptor(image, 16 / 64, 16 / 64)
    pale = _local_color_descriptor(image, 48 / 64, 48 / 64)

    assert dark.shape == pale.shape == (14,)
    assert not np.allclose(dark, pale)
