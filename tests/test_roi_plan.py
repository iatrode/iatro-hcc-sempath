from __future__ import annotations

import numpy as np
from PIL import Image

from hcc_sempath.modeling.roi_plan import (
    RoiPlanGenerator,
    _detect_nucleus_centers,
    _estimate_seed_exclusion_distance,
    _estimate_seed_spacing,
    _image_matching_channels,
    _normalized_template_response,
)


def _paint_nucleus(image: np.ndarray, x: int, y: int, *, rotated: bool = False) -> None:
    pattern = np.asarray(
        [
            [[220, 190, 220], [90, 45, 115], [55, 25, 90], [150, 105, 165], [225, 195, 225]],
            [[150, 105, 170], [45, 20, 80], [25, 10, 60], [65, 30, 100], [175, 135, 185]],
            [[185, 145, 195], [70, 35, 105], [35, 15, 70], [100, 55, 125], [210, 175, 215]],
        ],
        dtype=np.uint8,
    )
    if rotated:
        pattern = np.rot90(pattern)
    height, width = pattern.shape[:2]
    image[y - height // 2 : y - height // 2 + height, x - width // 2 : x - width // 2 + width] = pattern


def test_pixel_template_response_finds_rotated_local_pattern() -> None:
    image = np.full((64, 64, 3), [235, 210, 225], dtype=np.uint8)
    _paint_nucleus(image, 18, 22)
    _paint_nucleus(image, 46, 42, rotated=True)
    channels = _image_matching_channels(image)
    template = channels[18:27, 14:23]

    response = np.maximum.reduce(
        [
            _normalized_template_response(channels, np.rot90(template, turns, axes=(0, 1)))
            for turns in range(4)
        ]
    )

    target = response[38:47, 42:51]
    assert float(target.max()) > 0.95
    assert float(target.max()) > float(response[4:14, 4:14].max()) + 0.5


def test_nucleus_detector_keeps_one_center_per_nucleus() -> None:
    image = np.full((64, 64, 3), [235, 210, 225], dtype=np.uint8)
    _paint_nucleus(image, 20, 24)
    _paint_nucleus(image, 48, 44, rotated=True)
    hematoxylin = _image_matching_channels(image)[..., 0]

    centers = _detect_nucleus_centers(
        hematoxylin,
        nucleus_spacing_pixels=7,
        border=5,
    )

    near_first = [(x, y) for x, y in centers if (x - 20) ** 2 + (y - 24) ** 2 < 5**2]
    assert len(near_first) == 1
    assert any((x - 48) ** 2 + (y - 44) ** 2 < 5**2 for x, y in centers)


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


def test_seed_exclusion_caps_sparse_user_spacing_at_nucleus_scale() -> None:
    image = Image.new("RGB", (224, 224))
    normalized, pixels = _estimate_seed_exclusion_distance(
        image,
        [(0.25, 0.50), (0.35, 0.50), (0.45, 0.50)],
        fallback_pixels=7,
    )

    assert np.isclose(pixels, 8.75)
    assert np.isclose(normalized, pixels / 224)


def test_seed_exclusion_falls_back_to_morphology_for_one_mark() -> None:
    normalized, pixels = _estimate_seed_exclusion_distance(
        Image.new("RGB", (224, 224)),
        [(0.5, 0.5)],
        fallback_pixels=8.5,
    )

    assert pixels == 8.5
    assert normalized == pixels / 224


def test_generator_uses_independent_image_templates_and_real_similarity() -> None:
    image = np.full((96, 96, 3), [235, 210, 225], dtype=np.uint8)
    _paint_nucleus(image, 24, 30)
    _paint_nucleus(image, 70, 64, rotated=True)

    result = RoiPlanGenerator().generate_similar(
        Image.fromarray(image),
        attribute="inflammatory-cell-present",
        seeds=[(24 / 96, 30 / 96)],
        occupied=[(24 / 96, 30 / 96)],
    )

    assert result["method"] == "same-tile-nucleus-image-match-v4"
    assert "device" not in result
    target_matches = [
        item
        for item in result["suggestions"]
        if (item["geometry"]["point"][0] * 96 - 70) ** 2
        + (item["geometry"]["point"][1] * 96 - 64) ** 2
        < 5**2
        and item["confidence"] >= 0.70
    ]
    assert len(target_matches) == 1
    assert target_matches[0]["confidence"] > 0.95
    assert result["suggestions"][0] in target_matches
    assert result["summary"]["template_radius_pixels"] >= 4
    assert result["summary"]["peak_spacing_pixels"] >= 5
    assert result["summary"]["consensus_seed_count"] == 1
    assert result["summary"]["preview_target"] == 4


def test_generator_rejects_non_nuclear_attributes() -> None:
    with np.testing.assert_raises_regex(ValueError, "nucleus matching currently supports"):
        RoiPlanGenerator().generate_similar(
            Image.new("RGB", (96, 96), (220, 190, 210)),
            attribute="bile-pigment-present",
            seeds=[(0.5, 0.5)],
        )
