from __future__ import annotations

from hcc_sempath.io.tiling import select_read_level


def test_select_read_level_matches_target_native_downsample() -> None:
    assert select_read_level([1.0, 2.0, 4.0, 8.0], native_mpp=0.25, target_mpp=0.5) == 1
    assert select_read_level([1.0, 2.0, 4.0, 8.0], native_mpp=0.25, target_mpp=0.75) == 1
    assert select_read_level([1.0, 2.0, 4.0, 8.0], native_mpp=0.25, target_mpp=1.0) == 2
    assert select_read_level([1.0, 4.0, 8.0], native_mpp=0.25, target_mpp=0.75) == 1


def test_select_read_level_falls_back_to_level_zero_when_target_is_finer() -> None:
    assert select_read_level([1.0, 2.0, 4.0], native_mpp=0.5, target_mpp=0.25) == 0
