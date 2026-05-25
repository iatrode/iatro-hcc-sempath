from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from hcc_sempath.modeling.prototypes import load_prototype_registry, load_prototypes


def test_load_prototype_registry_from_package(tmp_path: Path) -> None:
    package_path = tmp_path / "hcc_prototypes.pt"
    torch.save(
        {
            "version": 1,
            "prototypes": torch.randn(2, 4),
            "names": ["primary_tumor", "lymphocyte_rich"],
            "groups": ["primary_state", "microenvironment"],
            "levels": [1, 2],
            "exclusive": [True, False],
            "thresholds": torch.tensor([0.6, 0.4]),
            "source": {"curation": "synthetic"},
        },
        package_path,
    )

    registry = load_prototype_registry(package_path, expected_dim=4)

    assert registry.count == 2
    assert registry.dim == 4
    assert registry.names == ["primary_tumor", "lymphocyte_rich"]
    assert registry.groups == ["primary_state", "microenvironment"]
    assert registry.levels == [1, 2]
    assert registry.exclusive == [True, False]
    assert registry.primary_indices == [0]
    assert registry.attribute_indices == [1]
    assert registry.thresholds is not None
    assert registry.source == {"curation": "synthetic", "path": str(package_path)}
    assert load_prototypes(package_path, expected_dim=4).shape == (2, 4)


def test_load_prototype_registry_from_directory_manifest(tmp_path: Path) -> None:
    prototype_dir = tmp_path / "registry"
    prototype_dir.mkdir()
    torch.save(
        {
            "prototypes": torch.ones(3, 5),
            "names": ["primary_tumor", "primary_non_tumor", "fibrotic_stroma"],
            "groups": ["primary_state", "primary_state", "stroma"],
            "levels": [1, 1, 2],
            "exclusive": [True, True, False],
        },
        prototype_dir / "custom.pt",
    )
    with (prototype_dir / "prototype_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"version": 1, "prototype_file": "custom.pt", "source": {"release": "test"}}, handle)

    registry = load_prototype_registry(prototype_dir, expected_dim=5)

    assert registry.count == 3
    assert registry.names == ["primary_tumor", "primary_non_tumor", "fibrotic_stroma"]
    assert registry.groups == ["primary_state", "primary_state", "stroma"]
    assert registry.source == {"release": "test", "path": str(prototype_dir)}


def test_load_prototype_registry_requires_names(tmp_path: Path) -> None:
    package_path = tmp_path / "missing_names.pt"
    torch.save(
        {"prototypes": torch.zeros(2, 3), "levels": [1, 2], "exclusive": [True, False]},
        package_path,
    )

    with pytest.raises(ValueError, match="names"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    package_path = tmp_path / "bad.pt"
    torch.save(
        {
            "prototypes": torch.zeros(2, 3),
            "names": ["dup", "dup"],
            "levels": [1, 2],
            "exclusive": [True, False],
        },
        package_path,
    )

    with pytest.raises(ValueError, match="names must be unique"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_rejects_threshold_shape_mismatch(tmp_path: Path) -> None:
    package_path = tmp_path / "bad_thresholds.pt"
    torch.save(
        {
            "prototypes": torch.zeros(2, 3),
            "names": ["a", "b"],
            "levels": [1, 2],
            "exclusive": [True, False],
            "thresholds": torch.zeros(3),
        },
        package_path,
    )

    with pytest.raises(ValueError, match="thresholds"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_rejects_nonexclusive_level_one(tmp_path: Path) -> None:
    package_path = tmp_path / "bad_level_one.pt"
    torch.save(
        {
            "prototypes": torch.zeros(2, 3),
            "names": ["primary_tumor", "lymphocyte_rich"],
            "levels": [1, 2],
            "exclusive": [False, False],
        },
        package_path,
    )

    with pytest.raises(ValueError, match="level-1 prototype must be exclusive"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_rejects_exclusive_level_two(tmp_path: Path) -> None:
    package_path = tmp_path / "bad_level_two.pt"
    torch.save(
        {
            "prototypes": torch.zeros(2, 3),
            "names": ["primary_tumor", "lymphocyte_rich"],
            "levels": [1, 2],
            "exclusive": [True, True],
        },
        package_path,
    )

    with pytest.raises(ValueError, match="level-2 prototype must be non-exclusive"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_requires_primary_level(tmp_path: Path) -> None:
    package_path = tmp_path / "missing_primary.pt"
    torch.save(
        {
            "prototypes": torch.zeros(2, 3),
            "names": ["lymphocyte_rich", "fibrotic_stroma"],
            "levels": [2, 2],
            "exclusive": [False, False],
        },
        package_path,
    )

    with pytest.raises(ValueError, match="level-1"):
        load_prototype_registry(package_path)
