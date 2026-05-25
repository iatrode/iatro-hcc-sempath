from __future__ import annotations

from pathlib import Path

import numpy as np

from hcc_sempath.cli.annotate_prototypes import (
    AnnotationData,
    AnnotationState,
    L1_PROTOTYPES,
    L2_PROTOTYPES,
    discover_iac_packages,
)
from hcc_sempath.io.manifests import TileRecord
from hcc_sempath.io.tile_package import build_tile_package_from_records, encode_jxl_array


def _records() -> list[TileRecord]:
    return [
        TileRecord(
            tile_id=f"s1_{idx:07d}",
            patient_id="p1",
            slide_id="s1",
            tile_path=Path(f"tiles/s1_{idx:07d}.jxl"),
            x=x * 16,
            y=y * 16,
            split="train",
        )
        for idx, (x, y) in enumerate((x, y) for y in range(2) for x in range(2))
    ]


def _write_iac(path: Path) -> None:
    records = _records()
    payloads = []
    for idx, _record in enumerate(records):
        arr = np.full((16, 16, 3), 40 + idx * 20, dtype=np.uint8)
        payloads.append(encode_jxl_array(arr, lossless=True, distance=None, effort=1))
    build_tile_package_from_records(
        records,
        payloads,
        path,
        tile_width=16,
        tile_height=16,
        stride_x=16,
        stride_y=16,
        lossless=True,
        effort=1,
        overwrite=True,
    )


def test_discover_iac_packages_supports_direct_file_and_dataset_dirs(tmp_path: Path) -> None:
    direct = tmp_path / "direct.iac"
    dataset_path = tmp_path / "cohort_a" / "nested.iac"
    dataset_path.parent.mkdir()
    _write_iac(direct)
    _write_iac(dataset_path)

    single = discover_iac_packages(direct)
    assert single[0].rel_path == "direct.iac"
    assert single[0].dataset == ""

    packages = discover_iac_packages(tmp_path)
    by_name = {package.rel_path: package for package in packages}
    assert by_name["direct.iac"].dataset == ""
    assert by_name["cohort_a/nested.iac"].dataset == "cohort_a"


def test_annotation_state_writes_resume_json_and_csv(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        record = data.viewer(0).records[0]
        data.state.save_annotation(package, record, L1_PROTOTYPES[0], [L2_PROTOTYPES[0], L2_PROTOTYPES[3]])
    finally:
        data.close()

    reloaded = AnnotationState(state_path, iac_path)
    assert len(reloaded.annotations) == 1
    csv_text = state_path.with_suffix(".csv").read_text(encoding="utf-8")
    assert "HCC-trabecular" in csv_text
    assert "necrotic;inflammatory-rich" in csv_text


def test_random_record_skips_annotated_tiles_and_thumbnail_is_png(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        first = data.viewer(0).records[0]
        data.state.save_annotation(package, first, L1_PROTOTYPES[1], [])

        seen_rows = {data.random_record(0)["record"]["row"] for _ in range(30)}
        assert first.row not in seen_rows
        assert data.thumbnail_png(0).startswith(b"\x89PNG")
    finally:
        data.close()
