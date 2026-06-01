from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

from hcc_sempath.io.feature_cache import build_teacher_feature_package_from_feature_map
from hcc_sempath.io.manifests import write_tile_manifest
from hcc_sempath.io.tile_package import build_tile_package, read_package_metadata
from hcc_sempath.training.config import manifest_data_paths
from hcc_sempath.training.datasets import (
    DistillationTileDataset,
    PackageSampledDistillationDataset,
    apply_split_overrides,
    collate_distillation,
    read_packaged_tile_records,
    validate_teacher_feature_package_pairs,
    validate_teacher_cache,
)
from hcc_sempath.training.manifest import build_training_manifest
from hcc_sempath.training.manifest import validate_manifest_artifacts
from hcc_sempath.training.train import _PackageShuffleBatchLoader
from hcc_sempath.training.prototype_labels import PrototypeLabel


def _write_package(root: Path, slide_id: str, value: int, count: int = 1) -> tuple[Path, Path]:
    tile_dir = root / f"{slide_id}_tiles"
    tile_dir.mkdir()
    rows = []
    feature_by_tile_id = {}
    for idx in range(count):
        tile_id = f"{slide_id}_{idx:07d}"
        tile_path = tile_dir / f"{tile_id}.png"
        Image.new("RGB", (32, 32), ((value + idx) % 255, 30, 120)).save(tile_path)
        rows.append(
            {
                "tile_id": tile_id,
                "patient_id": f"p_{slide_id}",
                "slide_id": slide_id,
                "tile_path": str(tile_path),
                "x": idx * 32,
                "y": 0,
                "split": "train",
            }
        )
        feature_by_tile_id[tile_id] = np.full((4,), value + idx, dtype=np.float32)
    manifest_path = root / f"{slide_id}.csv"
    tile_package_path = root / f"{slide_id}.tiles.iac"
    feature_package_path = root / f"{slide_id}.toy.features.iac"
    write_tile_manifest(manifest_path, rows)
    build_tile_package(manifest_path, tile_package_path)
    metadata = read_package_metadata(tile_package_path)
    records = read_packaged_tile_records([tile_package_path])
    build_teacher_feature_package_from_feature_map(
        [item.record for item in records],
        feature_by_tile_id,
        feature_package_path,
        teacher_name="toy",
        tile_width=int(metadata["tile_width"]),
        tile_height=int(metadata["tile_height"]),
        stride_x=int(metadata["stride_x"]),
        stride_y=int(metadata["stride_y"]),
    )
    return tile_package_path, feature_package_path


def test_multi_package_dataset_uses_external_slide_split(tmp_path: Path) -> None:
    tile_a, feature_a = _write_package(tmp_path, "slide_a", 10)
    tile_b, feature_b = _write_package(tmp_path, "slide_b", 20)
    split_path = tmp_path / "splits.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "split"])
        writer.writeheader()
        writer.writerow({"slide_id": "slide_a", "split": "train"})
        writer.writerow({"slide_id": "slide_b", "split": "val"})

    records = read_packaged_tile_records([tile_a, tile_b])
    records = apply_split_overrides(records, split_path, split_key="slide_id")
    validate_teacher_cache(records, expected_dim={"toy": 4}, teacher_cache_package_paths={"toy": [feature_a, feature_b]})

    train_records = [item for item in records if item.record.split == "train"]
    val_records = [item for item in records if item.record.split == "val"]
    assert [item.record.slide_id for item in train_records] == ["slide_a"]
    assert [item.record.slide_id for item in val_records] == ["slide_b"]

    dataset = DistillationTileDataset(
        records,
        teacher_cache_dir=None,
        image_size=(32, 32),
        teacher_cache_package_paths={"toy": [feature_a, feature_b]},
    )
    batch = collate_distillation([dataset[0], dataset[1]])

    assert batch["images"].shape == (2, 3, 32, 32)
    assert batch["prototype_mask"].tolist() == [False, False]
    np.testing.assert_allclose(batch["teacher_features"]["toy"][0].numpy(), np.full((4,), 10, dtype=np.float32))
    np.testing.assert_allclose(batch["teacher_features"]["toy"][1].numpy(), np.full((4,), 20, dtype=np.float32))


def test_dataset_adds_dynamic_prototype_supervision_fields(tmp_path: Path) -> None:
    tile_a, feature_a = _write_package(tmp_path, "slide_a", 10, count=2)
    records = read_packaged_tile_records([tile_a])
    labels = {
        "slide_a_0000000": PrototypeLabel(
            tile_id="slide_a_0000000",
            level1=1,
            level2=torch.tensor([1.0, 0.0]),
            source_split="train",
        )
    }

    dataset = DistillationTileDataset(
        records,
        teacher_cache_dir=None,
        image_size=(32, 32),
        teacher_cache_package_paths={"toy": [feature_a]},
        prototype_labels=labels,
    )
    batch = collate_distillation([dataset[0], dataset[1]])

    assert batch["prototype_mask"].tolist() == [True, False]
    assert batch["prototype_level1"].tolist() == [1, -1]
    assert batch["prototype_level2"].shape == (2, 2)


def test_build_training_manifest_splits_public_heldout_by_count(tmp_path: Path) -> None:
    dev_root = tmp_path / "dev"
    public_root = tmp_path / "public"
    dev_root.mkdir()
    public_root.mkdir()
    for idx in range(4):
        _write_package(dev_root, f"dev_{idx}", idx)
    for idx in range(5):
        _write_package(public_root, f"tcga_{idx}", idx)

    manifest = build_training_manifest(
        dev_sources={"internal": dev_root},
        public_source=("tcga", public_root),
        public_exval_n=2,
        val_frac=0.25,
        split_key="patient_id",
        seed=13,
    )

    public_train = set(manifest["splits"]["train"]["tcga"])
    public_exval = set(manifest["splits"]["exval"]["tcga_heldout"]["stems"])
    assert len(public_exval) == 2
    assert len(public_train) == 3
    assert public_train.isdisjoint(public_exval)
    assert set(manifest["splits"]["train"]["internal"]).isdisjoint(manifest["splits"]["val"]["internal"])
    assert manifest["summary"]["datasets"]["internal"]["package_count"] == 4
    assert manifest["summary"]["datasets"]["tcga"]["package_count"] == 5
    assert manifest["summary"]["splits"]["exval"]["tcga_heldout"]["package_count"] == 2


def test_manifest_data_paths_resolve_teacher_features_by_convention(tmp_path: Path) -> None:
    tile_root = tmp_path / "tiles"
    feature_root = tmp_path / "features"
    tile_root.mkdir()
    (feature_root / "toy").mkdir(parents=True)
    tile_path, feature_path = _write_package(tile_root, "slide_a", 10)
    expected_feature_path = feature_root / "toy" / "slide_a.toy.features.iac"
    feature_path.replace(expected_feature_path)
    manifest = {
        "version": 1,
        "tile_suffix": ".tiles.iac",
        "datasets": {"internal": {"role": "development", "tile_root": str(tile_root)}},
        "splits": {"train": {"internal": ["slide_a"]}, "val": {}, "exval": {}},
    }
    cfg = {
        "data": {"feature_root": str(feature_root), "teachers": ["toy"]},
        "model": {"teacher_dims": {"toy": 4}},
    }

    tile_packages, feature_packages = manifest_data_paths(cfg, manifest, "train")

    assert tile_packages == [str(tile_path)]
    assert feature_packages == {"toy": [str(expected_feature_path)]}


def test_validate_manifest_artifacts_checks_teacher_feature_packages(tmp_path: Path) -> None:
    tile_root = tmp_path / "tiles"
    feature_root = tmp_path / "features"
    tile_root.mkdir()
    (feature_root / "toy").mkdir(parents=True)
    tile_path, feature_path = _write_package(tile_root, "slide_a", 10)
    expected_feature_path = feature_root / "toy" / "slide_a.toy.features.iac"
    feature_path.replace(expected_feature_path)
    manifest = {
        "version": 1,
        "tile_suffix": ".tiles.iac",
        "datasets": {"internal": {"role": "development", "tile_root": str(tile_root)}},
        "splits": {"train": {"internal": ["slide_a"]}, "val": {}, "exval": {}},
    }

    result = validate_manifest_artifacts(
        manifest=manifest,
        splits=["train"],
        teachers=["toy"],
        feature_root=feature_root,
    )

    assert result["missing_tile_packages"] == []
    assert result["missing_feature_packages"] == []
    assert tile_path.exists()


def test_package_pair_validation_rejects_wrong_feature_stem(tmp_path: Path) -> None:
    tile_a, _ = _write_package(tmp_path, "slide_a", 10, count=3)
    _, feature_b = _write_package(tmp_path, "slide_b", 20, count=3)

    with pytest.raises(ValueError, match="stem mismatch"):
        validate_teacher_feature_package_pairs([tile_a], {"toy": [feature_b]}, expected_dims={"toy": 4})


def test_package_sampled_dataset_uses_every_selected_package_with_spread_rows(tmp_path: Path) -> None:
    tile_paths = []
    feature_paths = []
    for idx in range(4):
        tile_path, feature_path = _write_package(tmp_path, f"slide_{idx}", idx * 20, count=8)
        tile_paths.append(tile_path)
        feature_paths.append(feature_path)

    dataset = PackageSampledDistillationDataset(
        tile_paths,
        {"toy": feature_paths},
        image_size=(32, 32),
        max_records=8,
        seed=13,
        expected_dims={"toy": 4},
    )

    assert set(dataset.package_order) == {0, 1, 2, 3}
    for rows in dataset.package_sample_rows:
        assert rows is not None
        assert len(rows) == 2
        assert int(rows[1] - rows[0]) >= 2


def test_package_shuffle_loader_mixes_packages_single_thread(tmp_path: Path) -> None:
    tile_paths = []
    feature_paths = []
    for idx in range(6):
        tile_path, feature_path = _write_package(tmp_path, f"slide_{idx}", idx * 20, count=6)
        tile_paths.append(tile_path)
        feature_paths.append(feature_path)
    dataset = PackageSampledDistillationDataset(
        tile_paths,
        {"toy": feature_paths},
        image_size=(32, 32),
        max_records=24,
        seed=13,
        expected_dims={"toy": 4},
    )
    loader = _PackageShuffleBatchLoader(
        dataset,
        batch_size=8,
        num_workers=0,
        prefetch_batches=2,
        collate_fn=lambda batch: batch,
        seed=13,
    )

    first_batch = next(iter(loader))

    assert len({sample["_package_idx"] for sample in first_batch}) > 1
