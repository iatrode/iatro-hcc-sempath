from pathlib import Path

from PIL import Image

from hcc_sempath.build.teacher_features import TEACHERS, _parser, _plans
from hcc_sempath.inference.inputs import MaterializationOptions, materialize_input, plan_inputs


def _tile_package(tmp_path: Path) -> Path:
    image = tmp_path / "case.png"
    Image.new("RGB", (224, 224), (90, 40, 110)).save(image)
    plan = plan_inputs([str(image)], tmp_path / "tiles")[0]
    options = MaterializationOptions(
        split="train",
        target_mpp=0.5,
        native_mpp=None,
        native_mpp_y=None,
        min_tissue_fraction=0.1,
        prefilter_tissue_fraction=0.05,
        white_threshold=220,
        black_threshold=8,
        mask_max_pixels=1_000_000,
        max_tiles=None,
        workers=1,
        lossless=True,
        distance=1.0,
        effort=1,
        overwrite=False,
        show_progress=False,
    )
    return materialize_input(plan, options)


def test_public_teacher_feature_builder_has_fixed_four_teacher_contract() -> None:
    help_text = _parser().format_help()
    assert TEACHERS == ("gigapath", "h_optimus_1", "uni2_h", "virchow2")
    assert "merged four-teacher" in help_text
    assert "--teacher " not in help_text


def test_feature_plan_writes_one_merged_canonical_package(tmp_path: Path) -> None:
    tile_package = _tile_package(tmp_path)
    plans = _plans(str(tile_package), str(tmp_path / "features"))
    assert len(plans) == 1
    assert plans[0].output_path == (tmp_path / "features" / "case.feat.path.iac").resolve()
