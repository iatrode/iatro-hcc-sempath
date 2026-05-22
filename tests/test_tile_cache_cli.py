from pathlib import Path
from tempfile import TemporaryDirectory

from hcc_sempath.cli.tile_cache import _discover_wsi


def test_discover_wsi_scans_only_directory_top_level_for_mrxs_layout() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        top_level_slide = root / "case.mrxs"
        top_level_slide.write_text("mrxs header", encoding="utf-8")
        mrxs_data_dir = root / "case"
        mrxs_data_dir.mkdir()
        nested_sidecar_like_file = mrxs_data_dir / "nested.svs"
        nested_sidecar_like_file.write_text("sidecar payload", encoding="utf-8")

        assert _discover_wsi(root) == [top_level_slide]
