from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hcc_sempath.io.feature_cache import build_teacher_feature_package_from_tile_package
from hcc_sempath.io.manifests import write_tile_manifest
from hcc_sempath.io.tile_package import build_tile_package
from hcc_sempath.teacher.cache import (
    _build_arg_parser,
    _discover_tile_packages,
    _local_weight_path,
    _pool_virchow2_features,
    _resolve_feature_dtype,
    _resolve_model_spec,
    _teacher_name,
    cache_teacher_features_from_packages,
)


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
    assert args.prefetch_packages is True


def test_teacher_cache_cli_keeps_legacy_aliases_hidden_but_parseable() -> None:
    parser = _build_arg_parser()

    args = parser.parse_args(["--tile-package", "tiles", "--output", "features", "--model", "uni2_h"])

    assert args.input == "tiles"
    assert args.teacher == "uni2_h"


def test_supported_teacher_presets_resolve_expected_names() -> None:
    uni2_spec = _resolve_model_spec("uni2_h")
    assert uni2_spec["model_name"] == "hf-hub:MahmoodLab/UNI2-h"
    assert uni2_spec["feature_mode"] == "default"
    assert uni2_spec["model_kwargs"]["embed_dim"] == 1536
    assert _teacher_name("uni2_h", "") == "uni2_h"

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


def test_teacher_cache_batch_skips_existing_valid_output_and_writes_progress(tmp_path: Path) -> None:
    tile_package = _write_tile_package(tmp_path)
    output_dir = tmp_path / "features"
    output_dir.mkdir()
    feature_package = output_dir / "slide_a.toy.features.iac"
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
    (output_dir / "slide_a.toy.features.iac").write_text("not a package", encoding="utf-8")

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


def test_discover_tile_packages_rejects_invalid_iac_in_input_directory(tmp_path: Path) -> None:
    _write_tile_package(tmp_path)
    (tmp_path / "broken.iac").write_text("not an iac package", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid \\.iac package"):
        _discover_tile_packages(tmp_path)


def test_discover_tile_packages_rejects_non_tile_file_input(tmp_path: Path) -> None:
    tile_package = _write_tile_package(tmp_path)
    feature_package = tmp_path / "slide_a.toy.features.iac"
    build_teacher_feature_package_from_tile_package(
        tile_package,
        [np.arange(4, dtype=np.float32)],
        feature_package,
        teacher_name="toy",
    )

    with pytest.raises(ValueError, match="not an image tile"):
        _discover_tile_packages(feature_package)
