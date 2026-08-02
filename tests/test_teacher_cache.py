from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from hcc_sempath.teacher import cache as teacher_cache
from iatro.iac.adapters.features import build_teacher_feature_package_from_tile_package
from iatro.iac.adapters.manifests import write_tile_manifest
from iatro.iac.adapters.tiles import build_tile_package
from hcc_sempath.teacher.cache import (
    _build_arg_parser,
    BoundedTeacherBatchIterator,
    _discover_tile_packages,
    _local_weight_path,
    _pool_virchow2_features,
    _resolve_feature_dtype,
    _resolve_model_spec,
    _teacher_name,
    cache_teacher_features_from_package,
    cache_teacher_features_from_packages,
)


class TinyTeacher(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


def test_teacher_cache_cli_uses_input_and_teacher_names() -> None:
    parser = _build_arg_parser()

    args = parser.parse_args(
        [
            "--input",
            "tiles",
            "--output",
            "features",
            "--teacher",
            "virchow2",
            "--precision",
            "fp16",
            "--compile",
            "--compile-mode",
            "default",
        ]
    )

    assert args.input == "tiles"
    assert args.output == "features"
    assert args.teacher == "virchow2"
    assert args.precision == "fp16"
    assert args.compile_model is True
    assert args.compile_mode == "default"
    assert "--teacher TEACHER" in parser.format_usage()


def test_teacher_cache_cli_defaults_for_development_throughput() -> None:
    parser = _build_arg_parser()

    args = parser.parse_args(["--input", "tiles", "--output", "features", "--teacher", "virchow2"])

    assert args.batch_size == 512
    assert args.precision == "bf16"
    assert args.feature_dtype == "auto"
    assert _resolve_feature_dtype(args.feature_dtype, args.precision, "cuda") == "float16"
    assert args.compile_model is True
    assert args.validate_output is False


def test_teacher_batch_iterator_can_warm_one_batch_before_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    built_batches: list[tuple[int, int]] = []

    def fake_build_batch(executor, package_path, tile_ids, transform, indices, num_workers):
        built_batches.append((indices[0], indices[-1]))
        return {
            "tile_id": [tile_ids[index] for index in indices],
            "image": torch.zeros((len(indices), 3, 4, 4)),
        }

    monkeypatch.setattr(teacher_cache, "_build_teacher_batch", fake_build_batch)
    iterator = BoundedTeacherBatchIterator(
        Path("tiles.iac"),
        [f"tile_{index}" for index in range(8)],
        lambda image: image,
        batch_size=2,
        num_workers=2,
        prefetch_factor=3,
        initial_prefetch=1,
    )
    try:
        deadline = time.time() + 1
        while len(built_batches) < 1 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)

        assert built_batches == [(0, 1)]

        iterator.promote()
        deadline = time.time() + 1
        while len(built_batches) < 3 and time.time() < deadline:
            time.sleep(0.01)

        assert built_batches[:3] == [(0, 1), (2, 3), (4, 5)]
    finally:
        iterator.close()


def test_supported_teacher_presets_resolve_expected_names() -> None:
    uni2_spec = _resolve_model_spec("uni2_h")
    assert uni2_spec["model_name"] == "hf-hub:MahmoodLab/UNI2-h"
    assert uni2_spec["feature_mode"] == "default"
    assert uni2_spec["model_kwargs"]["embed_dim"] == 1536
    assert _teacher_name("uni2_h", "") == "uni2_h"

    h_optimus_spec = _resolve_model_spec("h_optimus_1")
    assert h_optimus_spec["model_name"] == "hf-hub:bioptimus/H-optimus-1"
    assert h_optimus_spec["feature_mode"] == "default"
    assert _teacher_name("h_optimus_1", "") == "h_optimus_1"

    virchow2_spec = _resolve_model_spec("virchow2")
    assert virchow2_spec["model_name"] == "hf-hub:paige-ai/Virchow2"
    assert virchow2_spec["feature_mode"] == "virchow2"
    assert _teacher_name("virchow2", "") == "virchow2"


def test_pool_virchow2_features_concatenates_class_and_patch_mean() -> None:
    output = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    features = _pool_virchow2_features(output)

    expected = torch.cat([output[:, 0], output[:, 5:].mean(1)], dim=-1)
    assert features.shape == (2, 6)
    torch.testing.assert_close(features, expected)


def test_local_teacher_directory_uses_matching_preset(tmp_path) -> None:
    local_dir = tmp_path / "virchow2"
    local_dir.mkdir()

    spec = _resolve_model_spec(str(local_dir))
    assert spec["model_name"] == str(local_dir)
    assert spec["feature_mode"] == "virchow2"
    assert spec["model_kwargs"]["mlp_layer"] is not None


def test_local_weight_path_accepts_safetensors(tmp_path) -> None:
    model_dir = tmp_path / "teacher"
    model_dir.mkdir()
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"stub")

    assert _local_weight_path(model_dir, {}) == weight_path


def _write_tile_package(tmp_path: Path, stem: str = "slide_a") -> Path:
    from PIL import Image

    tile_dir = tmp_path / f"{stem}_tiles"
    tile_dir.mkdir()
    tile_path = tile_dir / f"{stem}_0000000.png"
    Image.new("RGB", (16, 16), (100, 40, 20)).save(tile_path)
    manifest_path = tmp_path / f"{stem}.csv"
    package_path = tmp_path / f"{stem}.tiles.iac"
    write_tile_manifest(
        manifest_path,
        [
            {
                "tile_id": f"{stem}_0000000",
                "patient_id": f"p_{stem}",
                "slide_id": stem,
                "tile_path": str(tile_path),
                "x": 0,
                "y": 0,
                "split": "train",
            }
        ],
    )
    build_tile_package(manifest_path, package_path)
    return package_path


def _write_many_tile_package(tmp_path: Path, stem: str, count: int) -> Path:
    from PIL import Image

    tile_dir = tmp_path / f"{stem}_tiles"
    tile_dir.mkdir()
    rows = []
    for idx in range(count):
        tile_id = f"{stem}_{idx:07d}"
        tile_path = tile_dir / f"{tile_id}.png"
        Image.new("RGB", (16, 16), ((idx * 17) % 255, 40, 120)).save(tile_path)
        rows.append(
            {
                "tile_id": tile_id,
                "patient_id": f"p_{stem}",
                "slide_id": stem,
                "tile_path": str(tile_path),
                "x": idx * 16,
                "y": 0,
                "split": "train",
            }
        )
    manifest_path = tmp_path / f"{stem}.csv"
    package_path = tmp_path / f"{stem}.tiles.iac"
    write_tile_manifest(manifest_path, rows)
    build_tile_package(manifest_path, package_path)
    return package_path


def test_teacher_cache_batch_skips_existing_valid_output_and_writes_progress(tmp_path: Path) -> None:
    tile_package = _write_tile_package(tmp_path)
    output_dir = tmp_path / "features"
    output_dir.mkdir()
    feature_package = output_dir / "slide_a.toy.feat.path.iac"
    build_teacher_feature_package_from_tile_package(
        tile_package,
        [np.arange(4, dtype=np.float32)],
        feature_package,
        teacher_name="toy",
    )

    cache_teacher_features_from_packages(
        model=torch.nn.Identity(),
        package_paths=[tile_package],
        output=output_dir,
        batch_size=1,
        device="cpu",
        teacher_name="toy",
        num_workers=0,
    )

    progress = (output_dir / "teacher_cache_progress.csv").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "teacher_cache_progress.json").read_text(encoding="utf-8"))
    assert "skipped" in progress
    assert summary["ok"] == 1
    assert summary["failed"] == 0


def test_teacher_cache_batch_records_invalid_existing_output_with_continue(tmp_path: Path) -> None:
    tile_package = _write_tile_package(tmp_path)
    output_dir = tmp_path / "features"
    output_dir.mkdir()
    (output_dir / "slide_a.toy.feat.path.iac").write_text("not a package", encoding="utf-8")

    cache_teacher_features_from_packages(
        model=torch.nn.Identity(),
        package_paths=[tile_package],
        output=output_dir,
        batch_size=1,
        device="cpu",
        teacher_name="toy",
        num_workers=0,
        continue_on_error=True,
    )

    progress = (output_dir / "teacher_cache_progress.csv").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "teacher_cache_progress.json").read_text(encoding="utf-8"))
    assert "failed" in progress
    assert summary["ok"] == 0
    assert summary["failed"] == 1


@pytest.mark.skipif(
    os.environ.get("HCC_SEMPATH_RUN_WORKER_TESTS") != "1",
    reason="requires local multiprocessing shared-memory permissions",
)
def test_teacher_cache_package_releases_worker_processes_across_iacs(tmp_path: Path) -> None:
    packages = [_write_many_tile_package(tmp_path, f"slide_{idx}", 16) for idx in range(3)]
    model = TinyTeacher()

    for idx, tile_package in enumerate(packages):
        cache_teacher_features_from_package(
            model=model,
            package_path=tile_package,
            output_path=tmp_path / f"slide_{idx}.toy.feat.path.iac",
            tile_size=None,
            batch_size=4,
            device="cpu",
            teacher_name="toy",
            num_workers=2,
            prefetch_factor=2,
            overwrite=True,
            precision="fp32",
            feature_dtype="float32",
            data_config={
                "input_size": (3, 16, 16),
                "interpolation": "bicubic",
                "mean": (0.0, 0.0, 0.0),
                "std": (1.0, 1.0, 1.0),
                "crop_pct": 1.0,
            },
        )
        gc.collect()
        assert mp.active_children() == []


def test_discover_tile_packages_rejects_invalid_iac_in_input_directory(tmp_path: Path) -> None:
    _write_tile_package(tmp_path)
    (tmp_path / "broken.iac").write_text("not an iac package", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid \\.iac package"):
        _discover_tile_packages(tmp_path)


def test_discover_tile_packages_rejects_non_tile_file_input(tmp_path: Path) -> None:
    tile_package = _write_tile_package(tmp_path)
    feature_package = tmp_path / "slide_a.toy.feat.path.iac"
    build_teacher_feature_package_from_tile_package(
        tile_package,
        [np.arange(4, dtype=np.float32)],
        feature_package,
        teacher_name="toy",
    )

    with pytest.raises(ValueError, match="not an image tile"):
        _discover_tile_packages(feature_package)
