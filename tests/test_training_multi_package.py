from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

from iatro.iac.adapters.features import build_teacher_feature_package_from_feature_map
from iatro.iac import read_header
from iatro.iac.adapters.manifests import write_tile_manifest
from iatro.iac.adapters.tiles import build_tile_package, read_package_metadata
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
from hcc_sempath.training.feature_pack_merge import (
    MergedTeacherFeatureCacheReader,
    _build_merged_package,
    maybe_prepare_merged_teacher_feature_packages,
)
from hcc_sempath.training.feature_pack_shuffle import maybe_prepare_shuffled_iac_packages
from hcc_sempath.training.manifest import build_training_manifest
from hcc_sempath.training.manifest import validate_manifest_artifacts
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.train import (
    BatchSlot,
    _PackageShuffleBatchLoader,
    _assert_disjoint_package_cohorts,
    _freeze_optimizer_visible_contract,
    _verify_frozen_supervision_assets,
    _alloc_batch_buffer,
    _target_rows_by_package,
    _validation_package_keep_indices,
)
from hcc_sempath.training.prototype_labels import PrototypeLabel
from hcc_sempath.training.roi import SpatialRoiTarget


def _write_package(
    root: Path,
    slide_id: str,
    value: int,
    count: int = 1,
    patient_id: str | None = None,
) -> tuple[Path, Path]:
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
                "patient_id": patient_id or f"p_{slide_id}",
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


def _write_other_feature_package(root: Path, tile_path: Path, teacher_name: str = "other", offset: int = 100) -> Path:
    records = read_packaged_tile_records([tile_path])
    metadata = read_package_metadata(tile_path)
    slide_id = records[0].record.slide_id
    feature_path = root / f"{slide_id}.{teacher_name}.features.iac"
    build_teacher_feature_package_from_feature_map(
        [item.record for item in records],
        {
            item.record.tile_id: np.full((4,), offset + idx, dtype=np.float32)
            for idx, item in enumerate(records)
        },
        feature_path,
        teacher_name=teacher_name,
        tile_width=int(metadata["tile_width"]),
        tile_height=int(metadata["tile_height"]),
        stride_x=int(metadata["stride_x"]),
        stride_y=int(metadata["stride_y"]),
    )
    return feature_path


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


def test_package_splits_reject_same_patient_across_slides(
    tmp_path: Path,
) -> None:
    train_tile, _ = _write_package(
        tmp_path,
        "slide_train",
        10,
        patient_id="shared-patient",
    )
    val_tile, _ = _write_package(
        tmp_path,
        "slide_val",
        20,
        patient_id="shared-patient",
    )

    with pytest.raises(ValueError, match="cohort leakage"):
        _assert_disjoint_package_cohorts(
            [str(train_tile)],
            [str(val_tile)],
        )


def test_optimizer_visible_contract_freezes_packages_and_assets(
    tmp_path: Path,
) -> None:
    population, _ = _write_package(
        tmp_path,
        "population",
        10,
    )
    replay, _ = _write_package(
        tmp_path,
        "replay",
        20,
    )
    spatial = tmp_path / "spatial.json"
    spatial.write_text('{"version": 1}', encoding="utf-8")
    cfg = {
        "data": {
            "spatial_manifest_path": str(spatial),
        }
    }

    _freeze_optimizer_visible_contract(
        cfg,
        population_packages=[str(population)],
        expert_packages=[str(replay)],
        expert_replay_enabled=True,
    )

    assert cfg["data"]["optimizer_visible_tile_packages"] == sorted(
        [str(population.resolve()), str(replay.resolve())]
    )
    assert (
        len(cfg["data"]["optimizer_visible_contract_sha256"])
        == 64
    )
    first_asset_digest = cfg["data"]["supervision_asset_sha256"][
        "data.spatial_manifest_path"
    ]
    _verify_frozen_supervision_assets(cfg)
    spatial.write_text('{"version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        _verify_frozen_supervision_assets(cfg)
    _freeze_optimizer_visible_contract(
        cfg,
        population_packages=[str(population)],
        expert_packages=[str(replay)],
        expert_replay_enabled=True,
    )
    assert (
        cfg["data"]["supervision_asset_sha256"][
            "data.spatial_manifest_path"
        ]
        != first_asset_digest
    )


def test_dataset_adds_dynamic_prototype_supervision_fields(tmp_path: Path) -> None:
    tile_a, feature_a = _write_package(tmp_path, "slide_a", 10, count=2)
    records = read_packaged_tile_records([tile_a])
    labels = {
        "slide_a_0000000": PrototypeLabel(
            tile_id="slide_a_0000000",
            classification=1,
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
    assert batch["prototype_classification"].tolist() == [1, -1]


def test_expert_row_resolution_reads_only_requested_tile_ids(
    tmp_path: Path,
) -> None:
    tile_a, _ = _write_package(tmp_path, "slide_a", 10, count=3)
    tile_b, _ = _write_package(tmp_path, "slide_b", 20, count=2)

    rows = _target_rows_by_package(
        [str(tile_a), str(tile_b)],
        {"slide_a_0000001", "slide_b_0000000"},
    )

    assert rows[0].tolist() == [1]
    assert rows[1].tolist() == [0]


def test_expert_row_resolution_rejects_tile_outside_train_packages(
    tmp_path: Path,
) -> None:
    tile_a, _ = _write_package(tmp_path, "slide_a", 10, count=2)

    with pytest.raises(ValueError, match="outside the training split"):
        _target_rows_by_package(
            [str(tile_a)],
            {"slide_a_0000000", "missing_tile"},
        )


def test_expert_row_resolution_can_find_subset_without_missing_error(
    tmp_path: Path,
) -> None:
    tile_a, _ = _write_package(tmp_path, "slide_a", 10, count=2)

    rows = _target_rows_by_package(
        [str(tile_a)],
        {"slide_a_0000000", "outside"},
        require_all=False,
    )

    assert rows[0].tolist() == [0]


def test_validation_exclusion_covers_same_patient_across_packages(
    tmp_path: Path,
) -> None:
    expert, _ = _write_package(
        tmp_path,
        "slide_expert",
        10,
        patient_id="shared",
    )
    related, _ = _write_package(
        tmp_path,
        "slide_related",
        20,
        patient_id="shared",
    )
    independent, _ = _write_package(
        tmp_path,
        "slide_independent",
        30,
        patient_id="other",
    )

    keep = _validation_package_keep_indices(
        [str(related), str(independent)],
        [str(expert)],
    )

    assert keep == [1]


def test_dataset_collates_spatial_targets_and_preserves_ignore_mask(tmp_path: Path) -> None:
    tile_a, feature_a = _write_package(tmp_path, "slide_a", 10, count=2)
    records = read_packaged_tile_records([tile_a])
    target = torch.zeros((2, 2, 2), dtype=torch.float32)
    valid = torch.zeros((2, 2, 2), dtype=torch.bool)
    target[0, 0, 0] = 1
    valid[0, 1, 1] = True
    dataset = DistillationTileDataset(
        records,
        teacher_cache_dir=None,
        image_size=(32, 32),
        teacher_cache_package_paths={"toy": [feature_a]},
        spatial_targets={
            "slide_a_0000000": SpatialRoiTarget(
                point_centers=target,
                brush_bag_ids=torch.zeros_like(target, dtype=torch.long),
                area_positive=torch.zeros_like(valid),
                explicit_negative=torch.zeros_like(valid),
                implicit_negative=valid,
            )
        },
        spatial_component_count=2,
        spatial_grid_size=(2, 2),
    )
    batch = collate_distillation([dataset[0], dataset[1]])

    assert batch["spatial_point_centers"].shape == (2, 2, 2, 2)
    assert batch["spatial_implicit_negative"][0].sum().item() == 1
    assert batch["spatial_implicit_negative"][1].sum().item() == 0


def test_package_scatter_loader_copies_spatial_tensors(tmp_path: Path) -> None:
    tile_path, feature_path = _write_package(tmp_path, "slide_a", 10, count=2)
    target = torch.zeros((1, 2, 2), dtype=torch.float32)
    valid = torch.zeros((1, 2, 2), dtype=torch.bool)
    target[0, 1, 1] = 1
    valid[0, 0, 0] = True
    dataset = PackageSampledDistillationDataset(
        [tile_path],
        {"toy": [feature_path]},
        image_size=(32, 32),
        expected_dims={"toy": 4},
        spatial_targets={
            "slide_a_0000000": SpatialRoiTarget(
                point_centers=target,
                brush_bag_ids=torch.zeros_like(target, dtype=torch.long),
                area_positive=torch.zeros_like(valid),
                explicit_negative=torch.zeros_like(valid),
                implicit_negative=valid,
            )
        },
        spatial_component_count=1,
        spatial_grid_size=(2, 2),
    )
    buffer = _alloc_batch_buffer(2, dataset.batch_buffer_spec(), pin=False)
    dataset.scatter_package_rows(
        0,
        np.asarray([0, 1], dtype=np.int64),
        [BatchSlot(0, 0), BatchSlot(0, 1)],
        [buffer],
    )

    batch = buffer.view(2)
    assert batch["spatial_point_centers"][0, 0, 1, 1].item() == 1
    assert batch["spatial_implicit_negative"][0].sum().item() == 1
    assert batch["spatial_implicit_negative"][1].sum().item() == 0


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


def test_build_training_manifest_keeps_public_patient_group_intact(
    tmp_path: Path,
) -> None:
    dev_root = tmp_path / "dev"
    public_root = tmp_path / "public"
    dev_root.mkdir()
    public_root.mkdir()
    for idx in range(2):
        _write_package(dev_root, f"dev_{idx}", idx)
    _write_package(
        public_root,
        "shared_slide_a",
        10,
        patient_id="shared-patient",
    )
    _write_package(
        public_root,
        "shared_slide_b",
        20,
        patient_id="shared-patient",
    )
    _write_package(
        public_root,
        "independent_slide",
        30,
        patient_id="independent-patient",
    )

    manifest = build_training_manifest(
        dev_sources={"internal": dev_root},
        public_source=("tcga", public_root),
        public_exval_n=1,
        val_frac=0.5,
        split_key="patient_id",
        seed=1,
    )

    public_train = set(manifest["splits"]["train"]["tcga"])
    public_exval = set(
        manifest["splits"]["exval"]["tcga_heldout"]["stems"]
    )
    shared_stems = {"shared_slide_a", "shared_slide_b"}
    assert shared_stems <= public_exval
    assert public_train.isdisjoint(public_exval)
    assert len(public_exval) == 2
    assert (
        manifest["summary"]["splits"]["exval"]["tcga_heldout"][
            "package_count"
        ]
        == len(public_exval)
    )


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


def test_manifest_data_paths_prefers_existing_merged_after_source_delete(tmp_path: Path) -> None:
    tile_root = tmp_path / "tiles" / "301"
    feature_root = tmp_path / "features"
    tile_root.mkdir(parents=True)
    for subdir in ("toy", "other"):
        (feature_root / subdir / "301").mkdir(parents=True)
    tile_path, feature_a = _write_package(tile_root, "slide_a", 10, count=4)
    feature_b = _write_other_feature_package(tile_root, tile_path)
    expected_a = feature_root / "toy" / "301" / feature_a.name
    expected_b = feature_root / "other" / "301" / feature_b.name
    feature_a.replace(expected_a)
    feature_b.replace(expected_b)
    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg={
            "data": {
                "auto_merge_teacher_features": True,
                "auto_merge_delete_source_features": True,
            }
        },
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(expected_a)], "other": [str(expected_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    merged_path = Path(merged["toy"][0])
    assert merged_path.exists()
    assert not expected_a.exists()
    assert not expected_b.exists()
    manifest = {
        "version": 1,
        "tile_suffix": ".tiles.iac",
        "datasets": {"internal": {"role": "development", "tile_root": str(tile_root)}},
        "feature_roots": {
            "toy": str(feature_root / "toy"),
            "other": str(feature_root / "other"),
        },
        "splits": {"train": {"internal": ["slide_a"]}, "val": {}, "exval": {}},
    }
    cfg = {
        "data": {
            "teachers": ["toy", "other"],
            "prefer_merged_teacher_features": True,
        },
        "model": {"teacher_dims": {"toy": 4, "other": 4}},
        "runtime": {"seed": 13},
    }

    tile_packages, feature_packages = manifest_data_paths(cfg, manifest, "train")

    assert tile_packages == [str(tile_path)]
    assert feature_packages == {"toy": [str(merged_path)], "other": [str(merged_path)]}


def test_manifest_data_paths_prefers_merged_feature_root_directory(tmp_path: Path) -> None:
    tile_root = tmp_path / "tiles" / "301"
    feature_root = tmp_path / "features"
    tile_root.mkdir(parents=True)
    for subdir in ("toy", "other", "merged"):
        (feature_root / subdir / "301").mkdir(parents=True)
    tile_path, feature_a = _write_package(tile_root, "slide_a", 10, count=4)
    feature_b = _write_other_feature_package(tile_root, tile_path)
    expected_a = feature_root / "toy" / "301" / feature_a.name
    expected_b = feature_root / "other" / "301" / feature_b.name
    feature_a.replace(expected_a)
    feature_b.replace(expected_b)
    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg={"data": {"auto_merge_teacher_features": True}},
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(expected_a)], "other": [str(expected_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    merged_path = Path(merged["toy"][0])
    canonical_merged = feature_root / "merged" / "301" / merged_path.name
    merged_path.replace(canonical_merged)
    expected_a.unlink()
    expected_b.unlink()
    manifest = {
        "version": 1,
        "tile_suffix": ".tiles.iac",
        "datasets": {"internal": {"role": "development", "tile_root": str(tile_root)}},
        "splits": {"train": {"internal": ["slide_a"]}, "val": {}, "exval": {}},
    }
    cfg = {
        "data": {
            "feature_root": str(feature_root),
            "teachers": ["toy", "other"],
            "prefer_merged_teacher_features": True,
        },
        "model": {"teacher_dims": {"toy": 4, "other": 4}},
        "runtime": {"seed": 13},
    }

    tile_packages, feature_packages = manifest_data_paths(cfg, manifest, "train")

    assert tile_packages == [str(tile_path)]
    assert feature_packages == {"toy": [str(canonical_merged)], "other": [str(canonical_merged)]}


def test_manifest_data_paths_prefers_merged_directory_with_manifest_feature_roots(tmp_path: Path) -> None:
    tile_root = tmp_path / "tiles" / "301"
    feature_root = tmp_path / "features"
    tile_root.mkdir(parents=True)
    for subdir in ("toy", "other", "merged"):
        (feature_root / subdir / "301").mkdir(parents=True)
    tile_path, feature_a = _write_package(tile_root, "slide_a", 10, count=4)
    feature_b = _write_other_feature_package(tile_root, tile_path)
    expected_a = feature_root / "toy" / "301" / feature_a.name
    expected_b = feature_root / "other" / "301" / feature_b.name
    feature_a.replace(expected_a)
    feature_b.replace(expected_b)
    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg={"data": {"auto_merge_teacher_features": True}},
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(expected_a)], "other": [str(expected_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    canonical_merged = feature_root / "merged" / "301" / Path(merged["toy"][0]).name
    Path(merged["toy"][0]).replace(canonical_merged)
    expected_a.unlink()
    expected_b.unlink()
    manifest = {
        "version": 1,
        "tile_suffix": ".tiles.iac",
        "datasets": {"internal": {"role": "development", "tile_root": str(tile_root)}},
        "feature_roots": {
            "toy": str(feature_root / "toy"),
            "other": str(feature_root / "other"),
        },
        "splits": {"train": {"internal": ["slide_a"]}, "val": {}, "exval": {}},
    }
    cfg = {
        "data": {
            "teachers": ["toy", "other"],
            "prefer_merged_teacher_features": True,
        },
        "model": {"teacher_dims": {"toy": 4, "other": 4}},
        "runtime": {"seed": 13},
    }

    tile_packages, feature_packages = manifest_data_paths(cfg, manifest, "train")

    assert tile_packages == [str(tile_path)]
    assert feature_packages == {"toy": [str(canonical_merged)], "other": [str(canonical_merged)]}


def test_build_merged_package_cleans_data_temp_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tile_path, feature_a = _write_package(tmp_path, "slide_a", 10, count=4)
    feature_b = _write_other_feature_package(tmp_path, tile_path)
    merged_path = tmp_path / "merged" / "slide_a.merged.features.iac"

    def fail_read_feature_at(self, row: int):
        raise OSError("No space left on device")

    monkeypatch.setattr(
        "hcc_sempath.training.feature_pack_merge._SourceFeatureReader.read_feature_at",
        fail_read_feature_at,
    )

    with pytest.raises(OSError, match="No space left on device"):
        _build_merged_package(
            tile_path=tile_path,
            source_paths={"toy": feature_a, "other": feature_b},
            merged_path=merged_path,
            expected_dims={"toy": 4, "other": 4},
            dtype="float32",
        )

    assert not merged_path.exists()
    assert list(merged_path.parent.glob("tmp*")) == []


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


def test_package_pair_validation_accepts_feature_source_alias_suffix(tmp_path: Path) -> None:
    tile_a, feature_a = _write_package(tmp_path, "slide_a", 10, count=3)
    alias_feature = tmp_path / "slide_a.teacher-source-alias.features.iac"
    feature_a.replace(alias_feature)

    counts = validate_teacher_feature_package_pairs([tile_a], {"toy": [alias_feature]}, expected_dims={"toy": 4})

    assert counts == [3]


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


def test_package_shuffle_loader_mixes_packages_multi_thread(tmp_path: Path) -> None:
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
        num_workers=2,
        prefetch_batches=2,
        collate_fn=lambda batch: dataset.collate(batch),
        seed=13,
    )

    first_batch = next(iter(loader))
    assert len(first_batch["tile_id"]) == 8


def test_package_sampled_tensor_collate_defers_image_preprocess_to_device(tmp_path: Path) -> None:
    tile_path, feature_path = _write_package(tmp_path, "slide_a", 10, count=4)
    dataset = PackageSampledDistillationDataset(
        [tile_path],
        {"toy": [feature_path]},
        image_size=(32, 32),
        max_records=4,
        seed=13,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        expected_dims={"toy": 4},
        tensor_collate=True,
    )
    batch = dataset.collate([dataset[index] for index in range(4)])

    assert batch["images"].dtype == torch.uint8
    assert batch["images"].shape == (4, 3, 32, 32)
    assert batch["images_uint8"] is True

    images = _prepare_images(
        batch,
        {
            "data": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        },
        torch.device("cpu"),
    )

    assert images.dtype == torch.float32
    assert images.shape == (4, 3, 32, 32)


def test_package_sampled_batch_reads_match_scalar_behavior(tmp_path: Path) -> None:
    tile_path, feature_path = _write_package(tmp_path, "slide_a", 10, count=4)
    dataset = PackageSampledDistillationDataset(
        [tile_path],
        {"toy": [feature_path]},
        image_size=(32, 32),
        max_records=4,
        seed=13,
        expected_dims={"toy": 4},
    )
    indices = [3, 0, 2]
    scalar = [dataset[index] for index in indices]
    batched = dataset.__getitems__(indices)

    assert [item["tile_id"] for item in batched] == [item["tile_id"] for item in scalar]
    for observed, expected in zip(batched, scalar):
        np.testing.assert_array_equal(observed["image"], expected["image"])
        np.testing.assert_array_equal(
            observed["teacher_features"]["toy"],
            expected["teacher_features"]["toy"],
        )


def test_auto_merge_teacher_feature_packages_replaces_sources(tmp_path: Path) -> None:
    tile_path, feature_a = _write_package(tmp_path, "slide_a", 10, count=4)
    records = read_packaged_tile_records([tile_path])
    metadata = read_package_metadata(tile_path)
    feature_b = tmp_path / "slide_a.other.features.iac"
    build_teacher_feature_package_from_feature_map(
        [item.record for item in records],
        {
            item.record.tile_id: np.full((4,), 100 + idx, dtype=np.float32)
            for idx, item in enumerate(records)
        },
        feature_b,
        teacher_name="other",
        tile_width=int(metadata["tile_width"]),
        tile_height=int(metadata["tile_height"]),
        stride_x=int(metadata["stride_x"]),
        stride_y=int(metadata["stride_y"]),
    )

    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg={
            "data": {
                "auto_merge_teacher_features": True,
                "auto_merge_delete_source_features": True,
            }
        },
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
        expected_dims={"toy": 4, "other": 4},
    )

    merged_path = Path(merged["toy"][0])
    assert merged["other"][0] == str(merged_path)
    assert merged_path.exists()
    assert not feature_a.exists()
    assert not feature_b.exists()

    validate_teacher_feature_package_pairs(
        [tile_path],
        merged,
        expected_dims={"toy": 4, "other": 4},
    )
    dataset = PackageSampledDistillationDataset(
        [tile_path],
        merged,
        image_size=(32, 32),
        max_records=4,
        seed=13,
        expected_dims={"toy": 4, "other": 4},
    )
    sample = dataset[0]
    np.testing.assert_allclose(sample["teacher_features"]["toy"], np.full((4,), 10, dtype=np.float32))
    np.testing.assert_allclose(sample["teacher_features"]["other"], np.full((4,), 100, dtype=np.float32))

    merged_reader = MergedTeacherFeatureCacheReader(merged["toy"][0])
    try:
        rows = [3, 0, 2]
        batch = merged_reader.read_features_many_at(rows, ["toy", "other"])
        for index, row in enumerate(rows):
            scalar = merged_reader.read_features_at(row, ["toy", "other"])
            np.testing.assert_array_equal(batch["toy"][index], scalar["toy"])
            np.testing.assert_array_equal(batch["other"][index], scalar["other"])
    finally:
        merged_reader.close()


def test_auto_merge_allows_source_teacher_header_aliases(tmp_path: Path) -> None:
    tile_path, _ = _write_package(tmp_path, "slide_a", 10, count=4)
    feature_a = _write_other_feature_package(
        tmp_path,
        tile_path,
        teacher_name="prov-gigapath-local",
        offset=10,
    )
    feature_b = _write_other_feature_package(
        tmp_path,
        tile_path,
        teacher_name="prov-uni-local",
        offset=100,
    )

    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg={
            "data": {
                "auto_merge_teacher_features": True,
                "auto_merge_delete_source_features": True,
            }
        },
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"gigapath": [str(feature_a)], "uni2_h": [str(feature_b)]},
        expected_dims={"gigapath": 4, "uni2_h": 4},
    )

    merged_path = Path(merged["gigapath"][0])
    header = read_header(merged_path)
    assert header["teachers"] == ["gigapath", "uni2_h"]
    assert header["source_teachers"] == {
        "gigapath": "prov-gigapath-local",
        "uni2_h": "prov-uni-local",
    }
    assert not feature_a.exists()
    assert not feature_b.exists()

    dataset = PackageSampledDistillationDataset(
        [tile_path],
        merged,
        image_size=(32, 32),
        max_records=4,
        seed=13,
        expected_dims={"gigapath": 4, "uni2_h": 4},
    )
    sample = dataset[0]
    np.testing.assert_allclose(sample["teacher_features"]["gigapath"], np.full((4,), 10, dtype=np.float32))
    np.testing.assert_allclose(sample["teacher_features"]["uni2_h"], np.full((4,), 100, dtype=np.float32))


def test_auto_merge_existing_bad_metadata_keeps_sources(tmp_path: Path) -> None:
    tile_path, feature_a = _write_package(tmp_path, "slide_a", 10, count=4)
    feature_b = _write_other_feature_package(tmp_path, tile_path)
    keep_sources_cfg = {
        "data": {
            "auto_merge_teacher_features": True,
            "auto_merge_delete_source_features": False,
        }
    }
    maybe_prepare_merged_teacher_feature_packages(
        cfg=keep_sources_cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
        expected_dims={"toy": 4, "other": 4},
    )

    with pytest.raises(ValueError, match="feature dim mismatch"):
        maybe_prepare_merged_teacher_feature_packages(
            cfg={
                "data": {
                    "auto_merge_teacher_features": True,
                    "auto_merge_delete_source_features": True,
                }
            },
            split="train",
            tile_packages=[str(tile_path)],
            teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
            expected_dims={"toy": 5, "other": 4},
        )

    assert feature_a.exists()
    assert feature_b.exists()


def test_auto_merge_existing_partial_sources_refuses_delete(tmp_path: Path) -> None:
    tile_path, feature_a = _write_package(tmp_path, "slide_a", 10, count=4)
    feature_b = _write_other_feature_package(tmp_path, tile_path)
    maybe_prepare_merged_teacher_feature_packages(
        cfg={
            "data": {
                "auto_merge_teacher_features": True,
                "auto_merge_delete_source_features": False,
            }
        },
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    feature_a.unlink()

    with pytest.raises(ValueError, match="refusing partial source deletion"):
        maybe_prepare_merged_teacher_feature_packages(
            cfg={
                "data": {
                    "auto_merge_teacher_features": True,
                    "auto_merge_delete_source_features": True,
                }
            },
            split="train",
            tile_packages=[str(tile_path)],
            teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
            expected_dims={"toy": 4, "other": 4},
        )

    assert feature_b.exists()


def test_auto_shuffle_iac_rows_reorders_tile_and_merged_feature_together(tmp_path: Path) -> None:
    tile_path, feature_a = _write_package(tmp_path, "slide_a", 10, count=8)
    records = read_packaged_tile_records([tile_path])
    metadata = read_package_metadata(tile_path)
    feature_b = tmp_path / "slide_a.other.features.iac"
    build_teacher_feature_package_from_feature_map(
        [item.record for item in records],
        {
            item.record.tile_id: np.full((4,), 100 + idx, dtype=np.float32)
            for idx, item in enumerate(records)
        },
        feature_b,
        teacher_name="other",
        tile_width=int(metadata["tile_width"]),
        tile_height=int(metadata["tile_height"]),
        stride_x=int(metadata["stride_x"]),
        stride_y=int(metadata["stride_y"]),
    )
    cfg = {
        "runtime": {"seed": 19},
        "data": {
            "auto_merge_teacher_features": True,
            "auto_merge_delete_source_features": True,
            "auto_shuffle_iac_rows": True,
            "iac_row_order_seed": 19,
        },
    }
    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    tile_paths, shuffled = maybe_prepare_shuffled_iac_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths=merged,
    )
    merged_path = Path(shuffled["toy"][0])

    assert tile_paths == [str(tile_path)]
    assert shuffled["other"][0] == str(merged_path)
    assert int(read_header(tile_path)["row_order_seed"]) == 19
    assert int(read_header(merged_path)["row_order_seed"]) == 19

    tile_ids = [item.record.tile_id for item in read_packaged_tile_records([tile_path])]
    assert tile_ids != [item.record.tile_id for item in records]
    validate_teacher_feature_package_pairs([tile_path], shuffled, expected_dims={"toy": 4, "other": 4})
    dataset = PackageSampledDistillationDataset(
        [tile_path],
        shuffled,
        image_size=(32, 32),
        max_records=8,
        seed=13,
        expected_dims={"toy": 4, "other": 4},
    )
    assert dataset.sequential_iac_rows is True
    sample = dataset[0]
    original_idx = int(sample["tile_id"].split("_")[-1])
    np.testing.assert_allclose(sample["teacher_features"]["toy"], np.full((4,), 10 + original_idx, dtype=np.float32))
    np.testing.assert_allclose(sample["teacher_features"]["other"], np.full((4,), 100 + original_idx, dtype=np.float32))


def test_auto_merge_and_shuffle_are_idempotent_after_source_delete(tmp_path: Path) -> None:
    tile_path, feature_a = _write_package(tmp_path, "slide_a", 10, count=8)
    feature_b = _write_other_feature_package(tmp_path, tile_path)
    cfg = {
        "runtime": {"seed": 19},
        "data": {
            "auto_merge_teacher_features": True,
            "prefer_merged_teacher_features": True,
            "auto_merge_delete_source_features": True,
            "auto_shuffle_iac_rows": True,
            "iac_row_order_seed": 19,
        },
    }
    first_merged = maybe_prepare_merged_teacher_feature_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    first_tile_paths, first_shuffled = maybe_prepare_shuffled_iac_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths=first_merged,
    )

    second_merged = maybe_prepare_merged_teacher_feature_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    second_tile_paths, second_shuffled = maybe_prepare_shuffled_iac_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths=second_merged,
    )

    assert not feature_a.exists()
    assert not feature_b.exists()
    assert first_tile_paths == second_tile_paths == [str(tile_path)]
    assert first_shuffled == second_shuffled
    assert int(read_header(tile_path)["row_order_seed"]) == 19
    assert int(read_header(first_shuffled["toy"][0])["row_order_seed"]) == 19


def test_auto_merge_and_shuffle_prepare_packages_in_parallel(tmp_path: Path) -> None:
    tile_paths = []
    feature_paths_a = []
    feature_paths_b = []
    for idx in range(4):
        tile_path, feature_a = _write_package(tmp_path, f"slide_{idx}", 10 + idx * 10, count=4)
        feature_b = _write_other_feature_package(tmp_path, tile_path, offset=100 + idx * 10)
        tile_paths.append(str(tile_path))
        feature_paths_a.append(str(feature_a))
        feature_paths_b.append(str(feature_b))
    cfg = {
        "runtime": {"seed": 19},
        "data": {
            "auto_merge_teacher_features": True,
            "auto_merge_delete_source_features": True,
            "auto_shuffle_iac_rows": True,
            "iac_row_order_seed": 19,
            "auto_iac_prepare_workers": 4,
        },
    }

    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg=cfg,
        split="train",
        tile_packages=tile_paths,
        teacher_package_paths={"toy": feature_paths_a, "other": feature_paths_b},
        expected_dims={"toy": 4, "other": 4},
    )
    shuffled_tile_paths, shuffled = maybe_prepare_shuffled_iac_packages(
        cfg=cfg,
        split="train",
        tile_packages=tile_paths,
        teacher_package_paths=merged,
    )

    assert shuffled_tile_paths == tile_paths
    for tile_path, feature_path in zip(shuffled_tile_paths, shuffled["toy"], strict=True):
        assert shuffled["other"][shuffled_tile_paths.index(tile_path)] == feature_path
        assert int(read_header(tile_path)["row_order_seed"]) == 19
        assert int(read_header(feature_path)["row_order_seed"]) == 19
    assert all(not Path(path).exists() for path in feature_paths_a)
    assert all(not Path(path).exists() for path in feature_paths_b)


def test_auto_shuffle_seed_conflict_keeps_existing_order(tmp_path: Path) -> None:
    tile_path, feature_a = _write_package(tmp_path, "slide_a", 10, count=8)
    feature_b = _write_other_feature_package(tmp_path, tile_path)
    cfg = {
        "runtime": {"seed": 19},
        "data": {
            "auto_merge_teacher_features": True,
            "auto_merge_delete_source_features": True,
            "auto_shuffle_iac_rows": True,
            "iac_row_order_seed": 19,
        },
    }
    merged = maybe_prepare_merged_teacher_feature_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths={"toy": [str(feature_a)], "other": [str(feature_b)]},
        expected_dims={"toy": 4, "other": 4},
    )
    _, shuffled = maybe_prepare_shuffled_iac_packages(
        cfg=cfg,
        split="train",
        tile_packages=[str(tile_path)],
        teacher_package_paths=merged,
    )
    merged_path = Path(shuffled["toy"][0])
    tile_ids_before = [item.record.tile_id for item in read_packaged_tile_records([tile_path])]

    with pytest.raises(ValueError, match="existing row_order_seed conflicts"):
        maybe_prepare_shuffled_iac_packages(
            cfg={
                "runtime": {"seed": 20},
                "data": {
                    "auto_shuffle_iac_rows": True,
                    "iac_row_order_seed": 20,
                },
            },
            split="train",
            tile_packages=[str(tile_path)],
            teacher_package_paths=shuffled,
        )

    assert int(read_header(tile_path)["row_order_seed"]) == 19
    assert int(read_header(merged_path)["row_order_seed"]) == 19
    assert [item.record.tile_id for item in read_packaged_tile_records([tile_path])] == tile_ids_before
