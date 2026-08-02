from pathlib import Path

import pytest

from hcc_sempath.iac_naming import (
    pathology_feature_stem,
    pathology_prediction_path,
    pathology_tile_path,
    pathology_tile_stem,
)


def test_pathology_names_use_four_letter_role_and_domain() -> None:
    assert pathology_tile_path("out", "case") == Path("out/case.tile.path.iac")
    assert pathology_prediction_path("out", "case") == Path("out/case.pred.path.iac")
    assert pathology_tile_stem("case.tile.path.iac") == "case"
    assert pathology_feature_stem("case.feat.path.iac") == "case"


def test_legacy_tile_and_feature_names_remain_read_compatible() -> None:
    assert pathology_tile_stem("case.tiles.iac") == "case"
    assert pathology_feature_stem("case.teacher.features.iac") == "case.teacher"
    with pytest.raises(ValueError, match="must end"):
        pathology_tile_stem("case.pred.path.iac")
