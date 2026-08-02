from pathlib import Path

import pytest
from PIL import Image

from iatro.iac import read_tables

from hcc_sempath.inference.inputs import (
    MaterializationOptions,
    materialize_input,
    plan_inputs,
)


def _options(**overrides) -> MaterializationOptions:
    values = {
        "split": "inference",
        "target_mpp": 0.5,
        "native_mpp": None,
        "native_mpp_y": None,
        "min_tissue_fraction": 0.1,
        "prefilter_tissue_fraction": 0.05,
        "white_threshold": 220,
        "black_threshold": 8,
        "mask_max_pixels": 1_000_000,
        "max_tiles": None,
        "workers": 1,
        "lossless": True,
        "distance": 1.0,
        "effort": 1,
        "overwrite": False,
        "show_progress": False,
    }
    values.update(overrides)
    return MaterializationOptions(**values)


def test_224_raster_is_not_rescaled_or_cropped(tmp_path: Path) -> None:
    source = tmp_path / "case.jpg"
    Image.new("RGB", (224, 224), (80, 20, 120)).save(source)
    plan = plan_inputs([str(source)], tmp_path / "out")[0]
    package = materialize_input(plan, _options())
    header, _, index = read_tables(package)
    assert header["tiling"]["crop_box"] == [0, 0, 224, 224]
    assert index.column("tile_x")[0].as_py() == 0


def test_other_raster_dimensions_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "case.png"
    Image.new("RGB", (244, 244)).save(source)
    plan = plan_inputs([str(source)], tmp_path / "out")[0]
    with pytest.raises(ValueError, match="224x224"):
        materialize_input(plan, _options())


def test_wsi_plan_uses_canonical_intermediate_and_prediction_names(tmp_path: Path) -> None:
    source = tmp_path / "case.mrxs"
    source.write_bytes(b"placeholder")
    plan = plan_inputs([str(source)], tmp_path / "out")[0]
    assert plan.kind == "wsi"
    assert plan.tile_package.name == "case.tile.path.iac"
    assert plan.prediction_package.name == "case.pred.path.iac"
