from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hcc_sempath.modeling.build_prototypes import main as build_prototypes_main
from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.modeling.prototypes import load_prototype_registry, load_prototypes
from hcc_sempath.training.prototype_labels import load_prototype_labels


def test_load_prototype_registry_from_package(tmp_path: Path) -> None:
    package_path = tmp_path / "hcc_prototypes.pt"
    torch.save(
        {
            "version": 1,
            "prototypes": torch.randn(3, 4),
            "names": ["primary_tumor", "primary_non_tumor", "lymphocyte_rich"],
            "groups": ["primary_state", "primary_state", "microenvironment"],
            "levels": [1, 1, 2],
            "exclusive": [True, True, False],
            "thresholds": torch.tensor([0.6, 0.5, 0.4]),
            "source": {"curation": "synthetic"},
        },
        package_path,
    )

    registry = load_prototype_registry(package_path, expected_dim=4)

    assert registry.count == 3
    assert registry.dim == 4
    assert registry.names == ["primary_tumor", "primary_non_tumor", "lymphocyte_rich"]
    assert registry.groups == ["primary_state", "primary_state", "microenvironment"]
    assert registry.levels == [1, 1, 2]
    assert registry.exclusive == [True, True, False]
    assert registry.primary_indices == [0, 1]
    assert registry.attribute_indices == [2]
    assert registry.thresholds is not None
    assert registry.source == {"curation": "synthetic", "path": str(package_path)}
    assert load_prototypes(package_path, expected_dim=4).shape == (3, 4)


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
        {"prototypes": torch.zeros(3, 3), "levels": [1, 1, 2], "exclusive": [True, True, False]},
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
            "levels": [1, 1, 2],
            "exclusive": [True, True, False],
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
            "levels": [1, 1, 2],
            "exclusive": [True, True, False],
            "thresholds": torch.zeros(2),
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


def test_build_prototypes_writes_two_level_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    primary_dir = tmp_path / "primary"
    attribute_dir = tmp_path / "attributes"
    for concept_dir in [
        primary_dir / "primary_tumor",
        primary_dir / "primary_non_tumor",
        attribute_dir / "lymphocyte_rich",
    ]:
        concept_dir.mkdir(parents=True)
        np.save(concept_dir / "example.npy", np.ones((4,), dtype=np.float32))
    output = tmp_path / "prototypes.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hcc-sempath build-prototypes",
            "--primary-dir",
            str(primary_dir),
            "--attribute-dir",
            str(attribute_dir),
            "--output",
            str(output),
        ],
    )

    build_prototypes_main()
    registry = load_prototype_registry(output, expected_dim=4)

    assert registry.names == ["primary_non_tumor", "primary_tumor", "lymphocyte_rich"]
    assert registry.levels == [1, 1, 2]
    assert registry.exclusive == [True, True, False]


def test_hcc_taxonomy_package_loads_with_expected_levels(tmp_path: Path) -> None:
    names = [
        "HCC-tumor",
        "Background-liver",
        "Inflammatory-stromal",
        "Degenerative-material",
        "hepatocellular-parenchyma-present",
        "necrosis-present",
        "hemorrhage-present",
        "bile-pigment-present",
        "inflammatory-cell-present",
        "fibrous-stroma-present",
        "steatosis-vacuolation-present",
        "hyaline-change-present",
        "vascular-structure-present",
        "ductular-portal-present",
    ]
    package_path = tmp_path / "taxonomy.pt"
    torch.save(
        {
            "version": 1,
            "prototypes": torch.randn(len(names), 8),
            "names": names,
            "groups": [
                "hcc_tumor",
                "background_liver",
                "inflammatory_stroma",
                "degenerative_material",
                "hepatocellular_parenchyma",
                "necrosis",
                "hemorrhage",
                "pigment",
                "inflammation",
                "fibrous_stroma",
                "steatosis_vacuolation",
                "hyaline_change",
                "vascular_structure",
                "ductular_portal",
            ],
            "levels": [1] * 4 + [2] * 10,
            "exclusive": [True] * 4 + [False] * 10,
        },
        package_path,
    )

    registry = load_prototype_registry(package_path, expected_dim=8)

    assert registry.primary_indices == list(range(4))
    assert registry.attribute_indices == list(range(4, 14))
    assert registry.names[0] == "HCC-tumor"
    assert registry.names[-1] == "ductular-portal-present"


def test_prototype_supervision_manifest_is_resolved_by_registry_names(tmp_path: Path) -> None:
    manifest_path = tmp_path / "prototype_supervision.csv"
    manifest_path.write_text(
        "tile_id,level1_label,level2_labels,source_split,expert_a,expert_b,adjudicated\n"
        "tile_1,primary_tumor,lymphocyte_rich;fibrotic_stroma,train,a,b,true\n"
        "tile_2,primary_non_tumor,,val,a,b,true\n",
        encoding="utf-8",
    )
    registry = PrototypeRegistry(
        prototypes=torch.randn(4, 3),
        names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich", "fibrotic_stroma"],
        groups=["primary", "primary", "immune", "stroma"],
        levels=[1, 1, 2, 2],
        exclusive=[True, True, False, False],
    )

    labels = load_prototype_labels(manifest_path, registry, allowed_source_splits={"train"})

    assert set(labels) == {"tile_1"}
    assert labels["tile_1"].level1 == 0
    assert labels["tile_1"].level2.tolist() == [1.0, 1.0]
