from __future__ import annotations

import io
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image

from hcc_sempath.cli.annotate_prototypes import (
    AnnotationData,
    AnnotationState,
    L1_PROTOTYPES,
    L2_PROTOTYPES,
    _auth_ok,
    _annotation_key,
    _load_or_create_auth_token,
    discover_iac_packages,
    make_handler,
)
from hcc_sempath.io.feature_cache import build_teacher_feature_package
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


def _write_iac_with_stride(path: Path, stride: int) -> None:
    records = [
        TileRecord(
            tile_id=f"s1_{idx:07d}",
            patient_id="p1",
            slide_id="s1",
            tile_path=Path(f"tiles/s1_{idx:07d}.jxl"),
            x=x * stride,
            y=y * stride,
            split="train",
        )
        for idx, (x, y) in enumerate((x, y) for y in range(2) for x in range(2))
    ]
    payloads = [
        encode_jxl_array(np.full((16, 16, 3), 60 + idx * 30, dtype=np.uint8), lossless=True, distance=None, effort=1)
        for idx, _record in enumerate(records)
    ]
    build_tile_package_from_records(
        records,
        payloads,
        path,
        tile_width=16,
        tile_height=16,
        stride_x=stride,
        stride_y=stride,
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
    assert by_name["cohort_a/nested.iac"].total == 4


def test_discover_iac_packages_follows_symlinked_dataset_dirs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "cohort_a"
    input_dir = tmp_path / "input"
    source_path = source_dir / "nested.iac"
    source_dir.mkdir(parents=True)
    input_dir.mkdir()
    _write_iac(source_path)
    (input_dir / "cohort_link").symlink_to(source_dir, target_is_directory=True)

    packages = discover_iac_packages(input_dir)

    assert [(package.rel_path, package.dataset, package.total) for package in packages] == [
        ("cohort_link/nested.iac", "cohort_link", 4)
    ]


def test_auth_token_requires_exact_nonempty_match() -> None:
    assert _auth_ok("secret", "secret")
    assert not _auth_ok("", "secret")
    assert not _auth_ok("secret ", "secret")
    assert not _auth_ok("other", "secret")


def test_auth_token_reuses_state_sidecar_and_creates_private_file(tmp_path: Path) -> None:
    state_path = tmp_path / "annotations.json"
    token_path = tmp_path / "annotations.auth-token"
    token_path.write_text("persisted-token\n", encoding="utf-8")

    assert _load_or_create_auth_token(state_path) == "persisted-token"
    assert _load_or_create_auth_token(state_path, "provided-token") == "provided-token"

    token_path.unlink()
    created = _load_or_create_auth_token(state_path)
    assert created
    assert token_path.read_text(encoding="utf-8").strip() == created
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_annotation_http_requires_auth_token(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    data = AnnotationData(iac_path, state_path)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(data, "secret"))
    except PermissionError:
        data.close()
        pytest.skip("local socket bind is blocked in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_address[1]}/"
    url = f"{root_url}api/packages"
    try:
        with urlopen(root_url, timeout=5) as response:
            assert response.status == 200
        try:
            urlopen(url, timeout=5)
            raise AssertionError("unauthenticated request unexpectedly succeeded")
        except HTTPError as exc:
            assert exc.code == 403
        with urlopen(f"{url}?token=secret", timeout=5) as response:
            assert response.status == 200
        with urlopen(f"{url}?auth_token=secret", timeout=5) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        data.close()


def test_discover_iac_packages_skips_teacher_feature_packages(tmp_path: Path) -> None:
    image_path = tmp_path / "tiles.iac"
    feature_path = tmp_path / "teacher.features.iac"
    records = _records()
    _write_iac(image_path)
    build_teacher_feature_package(
        records,
        [np.arange(4, dtype=np.float32) for _ in records],
        feature_path,
        teacher_name="smoke",
        overwrite=True,
    )

    packages = discover_iac_packages(tmp_path)
    assert [package.rel_path for package in packages] == ["tiles.iac"]


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
    assert reloaded.last_iac == "tiles.iac"
    csv_text = state_path.with_suffix(".csv").read_text(encoding="utf-8")
    assert "HCC-tumor" in csv_text
    assert "hepatocellular-parenchyma-present;bile-pigment-present" in csv_text


def test_annotation_state_uses_taxonomy_from_json_and_observed_labels(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    package = discover_iac_packages(iac_path)[0]
    viewer = AnnotationData(iac_path, state_path)
    try:
        record = viewer.viewer(0).records[0]
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "input_path": str(iac_path),
                    "l1_prototypes": ["HCC-tumor", "Background-liver"],
                    "l2_prototypes": ["hepatocellular-parenchyma-present"],
                    "annotations": {
                        _annotation_key(package, record): {
                            "dataset": package.dataset,
                            "iac": package.rel_path,
                            "iac_path": str(package.path),
                            "tile_id": record.tile_id,
                            "row": record.row,
                            "slide": record.slide_label,
                            "x": record.display_x,
                            "y": record.display_y,
                            "l1": "Inflammatory-stromal",
                            "l2": ["ductular-portal-present"],
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        viewer.close()

    data = AnnotationData(iac_path, state_path)
    try:
        assert data.state.l1_prototypes == ["HCC-tumor", "Background-liver", "Inflammatory-stromal"]
        assert data.state.l2_prototypes == ["hepatocellular-parenchyma-present", "ductular-portal-present"]
        second = data.viewer(0).records[1]
        data.state.save_annotation(package, second, "Inflammatory-stromal", ["ductular-portal-present"])
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert payload["l1_prototypes"] == data.state.l1_prototypes
        assert payload["l2_prototypes"] == data.state.l2_prototypes
    finally:
        data.close()


def test_csv_export_preserves_review_fields_without_crashing(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        record = data.viewer(0).records[0]
        data.state.save_annotation(package, record, L1_PROTOTYPES[0], [])
        key = _annotation_key(package, record)
        data.state.annotations[key]["reviewed"] = True
        data.state.annotations[key]["review_decision"] = "adjust"
        data.state.annotations[key]["review_suggested_l1"] = L1_PROTOTYPES[1]
        data.state.flush()
    finally:
        data.close()

    csv_text = state_path.with_suffix(".csv").read_text(encoding="utf-8")
    assert "review_decision" in csv_text
    assert "adjust" in csv_text


def test_scan_status_reports_last_iac_from_state(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "input_path": str(iac_path),
                "last_iac": "tiles.iac",
                "annotations": {},
            }
        ),
        encoding="utf-8",
    )

    data = AnnotationData(iac_path, state_path)
    try:
        assert data.scan_status()["last_iac"] == "tiles.iac"
    finally:
        data.close()


def test_progress_reports_overall_annotation_counts(tmp_path: Path) -> None:
    first_path = tmp_path / "a.iac"
    second_path = tmp_path / "cohort_b" / "b.iac"
    second_path.parent.mkdir()
    state_path = tmp_path / "annotations.json"
    _write_iac(first_path)
    _write_iac(second_path)

    data = AnnotationData(tmp_path, state_path)
    try:
        package = data.packages[0]
        record = data.viewer(0).records[0]
        data.state.save_annotation(package, record, L1_PROTOTYPES[0], [])

        progress = data.progress(0)
        assert progress["package"] == {"annotated": 1, "total": 4, "remaining": 3, "skipped": 0}
        assert progress["overall"] == {"annotated": 1, "total": 8, "remaining": 7, "skipped": 0}
    finally:
        data.close()


def test_progress_excludes_skipped_annotations(tmp_path: Path) -> None:
    first_path = tmp_path / "a.iac"
    second_path = tmp_path / "b.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(first_path)
    _write_iac(second_path)

    data = AnnotationData(tmp_path, state_path)
    try:
        first_package = data.packages[0]
        second_package = data.packages[1]
        first_records = data.viewer(0).records
        second_record = data.viewer(1).records[0]
        data.state.save_annotation(first_package, first_records[0], L1_PROTOTYPES[0], [])

        data.state.skipped.add(_annotation_key(first_package, first_records[1]))
        data.state.skipped.add(_annotation_key(second_package, second_record))

        progress = data.progress(0)
        assert progress["package"] == {"annotated": 1, "total": 4, "remaining": 2, "skipped": 1}
        assert progress["overall"] == {"annotated": 1, "total": 8, "remaining": 5, "skipped": 2}
        assert progress["l1"][L1_PROTOTYPES[0]] == 1
    finally:
        data.close()


def test_progress_label_counts_cover_all_packages(tmp_path: Path) -> None:
    first_path = tmp_path / "a.iac"
    second_path = tmp_path / "b.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(first_path)
    _write_iac(second_path)

    data = AnnotationData(tmp_path, state_path)
    try:
        first_package = data.packages[0]
        second_package = data.packages[1]
        first_record = data.viewer(0).records[0]
        second_record = data.viewer(1).records[0]
        data.state.save_annotation(first_package, first_record, L1_PROTOTYPES[0], [L2_PROTOTYPES[0]])
        data.state.save_annotation(second_package, second_record, L1_PROTOTYPES[1], [L2_PROTOTYPES[1]])

        progress = data.progress(0)
        assert progress["l1"][L1_PROTOTYPES[0]] == 1
        assert progress["l1"][L1_PROTOTYPES[1]] == 1
        assert progress["l2"][L2_PROTOTYPES[0]] == 1
        assert progress["l2"][L2_PROTOTYPES[1]] == 1
        assert progress["package_l1"][L1_PROTOTYPES[1]] == 0
    finally:
        data.close()


def test_top_level_skipped_records_are_preserved_and_not_sampled(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        skipped_record = data.viewer(0).records[0]
        skipped_key = _annotation_key(package, skipped_record)
    finally:
        data.close()

    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "input_path": str(iac_path.resolve()),
                "last_iac": package.rel_path,
                "l1_prototypes": L1_PROTOTYPES,
                "l2_prototypes": L2_PROTOTYPES,
                "annotations": {},
                "skipped": [skipped_key],
            }
        ),
        encoding="utf-8",
    )

    data = AnnotationData(iac_path, state_path)
    try:
        package = data.packages[0]
        skipped_record = data.viewer(0).records[0]
        assert data.state.is_annotated(package, skipped_record)

        progress = data.progress(0)
        assert progress["package"] == {"annotated": 0, "total": 4, "remaining": 3, "skipped": 1}

        seen_rows = {data.random_record(0)["record"]["row"] for _ in range(30)}
        assert skipped_record.row not in seen_rows

        data.state.save_annotation(package, skipped_record, L1_PROTOTYPES[0], [])
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert payload["last_iac"] == package.rel_path
        assert skipped_key not in payload["skipped"]
    finally:
        data.close()


def test_random_record_skips_annotated_tiles_and_overview_is_jpg(tmp_path: Path) -> None:
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
        assert data.thumbnail_jpg(0).startswith(b"\xff\xd8")
        assert data.overview_cache_path(package).exists()
    finally:
        data.close()


def test_context_jpg_renders_half_resolution_5x5_grid(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac(iac_path)

    data = AnnotationData(iac_path, state_path)
    try:
        image = Image.open(io.BytesIO(data.context_jpg(0, 0))).convert("RGB")
        assert image.size == (40, 40)
        assert image.getpixel((20, 20)) != (245, 247, 249)
        thumb = Image.open(io.BytesIO(data.thumbnail_jpg(0, selected_row=0))).convert("RGB")
        assert thumb.size == (8, 8)
    finally:
        data.close()


def test_overview_uses_fixed_cell_grid_for_spatial_layout(tmp_path: Path) -> None:
    iac_path = tmp_path / "tiles.iac"
    state_path = tmp_path / "annotations.json"
    _write_iac_with_stride(iac_path, stride=64)

    data = AnnotationData(iac_path, state_path)
    try:
        image = Image.open(io.BytesIO(data.thumbnail_jpg(0))).convert("RGB")
        assert image.size == (8, 8)
        assert image.getpixel((2, 2)) != (255, 255, 255)
    finally:
        data.close()
