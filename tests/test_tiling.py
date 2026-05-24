from __future__ import annotations

import numpy as np

from hcc_sempath.io.tiling import infer_native_mpp_from_properties, select_read_level, tissue_fraction


def test_select_read_level_matches_target_native_downsample() -> None:
    assert select_read_level([1.0, 2.0, 4.0, 8.0], native_mpp=0.25, target_mpp=0.5) == 1
    assert select_read_level([1.0, 2.0, 4.0, 8.0], native_mpp=0.25, target_mpp=0.75) == 1
    assert select_read_level([1.0, 2.0, 4.0, 8.0], native_mpp=0.25, target_mpp=1.0) == 2
    assert select_read_level([1.0, 4.0, 8.0], native_mpp=0.25, target_mpp=0.75) == 1


def test_select_read_level_falls_back_to_level_zero_when_target_is_finer() -> None:
    assert select_read_level([1.0, 2.0, 4.0], native_mpp=0.5, target_mpp=0.25) == 0


def test_tissue_fraction_excludes_black_and_white_background() -> None:
    tile = np.zeros((2, 2, 3), dtype=np.uint8)
    tile[0, 0] = [0, 0, 0]
    tile[0, 1] = [255, 255, 255]
    tile[1, 0] = [120, 80, 110]
    tile[1, 1] = [30, 30, 30]

    assert tissue_fraction(tile, white_threshold=220, black_threshold=8) == 0.5


def test_infer_native_mpp_prefers_explicit_mpp() -> None:
    props = {"openslide.mpp-x": "0.2529", "openslide.mpp-y": "0.2531", "aperio.AppMag": "40"}

    assert infer_native_mpp_from_properties(props) == (0.2529, 0.2531, "metadata")


def test_infer_native_mpp_uses_supported_objective_power_when_mpp_is_missing() -> None:
    props = {"aperio.AppMag": "20"}

    assert infer_native_mpp_from_properties(props) == (0.5, 0.5, "objective_power")


def test_infer_native_mpp_rejects_unknown_objective_power() -> None:
    props = {"aperio.AppMag": "37"}

    assert infer_native_mpp_from_properties(props) == (None, None, None)
