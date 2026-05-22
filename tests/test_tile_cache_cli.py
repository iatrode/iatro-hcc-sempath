from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from hcc_sempath.cli.tile_cache import _discover_wsi, _plan_slide_jobs, _safe_id, _write_batch_progress


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


def test_safe_id_preserves_unicode_slide_names() -> None:
    assert _safe_id("高勇_HE_Ca") == "高勇_HE_Ca"
    assert _safe_id("高晓斌_HE_Ca") == "高晓斌_HE_Ca"


def test_plan_slide_jobs_rejects_output_path_collisions() -> None:
    slides = [Path("case+a.svs"), Path("case a.svs")]
    with pytest.raises(ValueError, match="same output IAC path"):
        _plan_slide_jobs(
            slides,
            Path("/tmp/out"),
            patient_id=None,
            slide_id=None,
            tcga_patient_id=False,
            qc=False,
        )


def test_write_batch_progress_counts_skipped_rows() -> None:
    with TemporaryDirectory() as tmp:
        output = Path(tmp)
        rows = [{"status": "ok"}, {"status": "skipped"}]
        _write_batch_progress(output, Path("input"), rows, [], total=3, started=0.0)

        payload = (output / "batch_progress.json").read_text(encoding="utf-8")
        assert '"processed": 2' in payload
        assert '"total": 3' in payload
