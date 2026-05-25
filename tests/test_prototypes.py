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
            "names": ["tumor_trabecular", "lymphocyte_rich"],
            "groups": ["tumor", "microenvironment"],
            "thresholds": torch.tensor([0.6, 0.4]),
            "source": {"curation": "synthetic"},
        },
        package_path,
    )

    registry = load_prototype_registry(package_path, expected_dim=4)

    assert registry.count == 2
    assert registry.dim == 4
    assert registry.names == ["tumor_trabecular", "lymphocyte_rich"]
    assert registry.groups == ["tumor", "microenvironment"]
    assert registry.thresholds is not None
    assert registry.source == {"curation": "synthetic", "path": str(package_path)}
    assert load_prototypes(package_path, expected_dim=4).shape == (2, 4)


def test_load_prototype_registry_from_directory_manifest(tmp_path: Path) -> None:
    prototype_dir = tmp_path / "registry"
    prototype_dir.mkdir()
    torch.save(
        {
            "prototypes": torch.ones(3, 5),
            "names": ["a", "b", "c"],
            "groups": ["g1", "g1", "g2"],
        },
        prototype_dir / "custom.pt",
    )
    with (prototype_dir / "prototype_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"version": 1, "prototype_file": "custom.pt", "source": {"release": "test"}}, handle)

    registry = load_prototype_registry(prototype_dir, expected_dim=5)

    assert registry.count == 3
    assert registry.names == ["a", "b", "c"]
    assert registry.groups == ["g1", "g1", "g2"]
    assert registry.source == {"release": "test", "path": str(prototype_dir)}


def test_load_prototype_registry_assigns_default_names(tmp_path: Path) -> None:
    package_path = tmp_path / "unnamed.pt"
    torch.save(torch.zeros(2, 3), package_path)

    registry = load_prototype_registry(package_path)

    assert registry.names == ["prototype_000", "prototype_001"]
    assert registry.groups == [None, None]


def test_load_prototype_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    package_path = tmp_path / "bad.pt"
    torch.save({"prototypes": torch.zeros(2, 3), "names": ["dup", "dup"]}, package_path)

    with pytest.raises(ValueError, match="names must be unique"):
        load_prototype_registry(package_path)


def test_load_prototype_registry_rejects_threshold_shape_mismatch(tmp_path: Path) -> None:
    package_path = tmp_path / "bad_thresholds.pt"
    torch.save(
        {
            "prototypes": torch.zeros(2, 3),
            "names": ["a", "b"],
            "thresholds": torch.zeros(3),
        },
        package_path,
    )

    with pytest.raises(ValueError, match="thresholds"):
        load_prototype_registry(package_path)
