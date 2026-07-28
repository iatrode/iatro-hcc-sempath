from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.modeling.prototypes import load_prototype_registry, load_prototypes
from hcc_sempath.training.prototype_labels import load_prototype_labels


def test_load_prototype_registry_from_package(tmp_path: Path) -> None:
    package_path = tmp_path / "hcc_prototypes.pt"
    torch.save(
        {
            "version": 1,
            "prototypes": torch.randn(3, 4),
            "names": ["tumor_well", "tumor_moderate", "tumor_poor"],
            "thresholds": torch.tensor([0.6, 0.5, 0.4]),
            "source": {"curation": "synthetic"},
        },
        package_path,
    )

    registry = load_prototype_registry(package_path, expected_dim=4)

    assert registry.count == 3
    assert registry.dim == 4
    assert registry.names == ["tumor_well", "tumor_moderate", "tumor_poor"]
    assert registry.thresholds is not None
    assert registry.source == {"curation": "synthetic", "path": str(package_path)}
    assert load_prototypes(package_path, expected_dim=4).shape == (3, 4)


def test_load_prototype_registry_from_directory_manifest(tmp_path: Path) -> None:
    prototype_dir = tmp_path / "registry"
    prototype_dir.mkdir()
    torch.save(
        {
            "prototypes": torch.ones(3, 5),
            "names": ["tumor_well", "tumor_moderate", "tumor_poor"],
        },
        prototype_dir / "custom.pt",
    )
    with (prototype_dir / "prototype_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"version": 1, "prototype_file": "custom.pt", "source": {"release": "test"}}, handle)

    registry = load_prototype_registry(prototype_dir, expected_dim=5)

    assert registry.count == 3
    assert registry.names == ["tumor_well", "tumor_moderate", "tumor_poor"]
    assert registry.source == {"release": "test", "path": str(prototype_dir)}


def test_load_prototype_registry_requires_names(tmp_path: Path) -> None:
    package_path = tmp_path / "missing_names.pt"
    torch.save(
        {"prototypes": torch.zeros(3, 3)},
        package_path,
    )

    with pytest.raises(ValueError, match="names"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    package_path = tmp_path / "bad.pt"
    torch.save(
        {
            "prototypes": torch.zeros(3, 3),
            "names": ["dup", "dup", "attr"],
        },
        package_path,
    )

    with pytest.raises(ValueError, match="names must be unique"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_rejects_threshold_shape_mismatch(tmp_path: Path) -> None:
    package_path = tmp_path / "bad_thresholds.pt"
    torch.save(
        {
            "prototypes": torch.zeros(3, 3),
            "names": ["a", "b", "c"],
            "thresholds": torch.zeros(2),
        },
        package_path,
    )

    with pytest.raises(ValueError, match="thresholds"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_requires_at_least_two_classes(tmp_path: Path) -> None:
    package_path = tmp_path / "single_class.pt"
    torch.save(
        {
            "prototypes": torch.zeros(1, 3),
            "names": ["tumor"],
        },
        package_path,
    )

    with pytest.raises(ValueError, match="at least two classification prototypes"):
        load_prototype_registry(package_path)


def test_classification_supervision_manifest_is_resolved_by_registry_names(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prototype_supervision.csv"
    manifest_path.write_text(
        "tile_id,classification_label,source_split,adjudicated\n"
        "tile_1,tumor,train,true\n"
        "tile_2,non_tumor,val,true\n",
        encoding="utf-8",
    )
    registry = PrototypeRegistry(
        prototypes=torch.randn(2, 3),
        names=["tumor", "non_tumor"],
    )

    labels = load_prototype_labels(manifest_path, registry, allowed_source_splits={"train"})

    assert set(labels) == {"tile_1"}
    assert labels["tile_1"].classification == 0
