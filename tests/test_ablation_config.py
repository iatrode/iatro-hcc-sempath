from pathlib import Path

import yaml

from experiments.ablation.scripts.resolve_ablation_config import (
    resolve_ablation_config,
)


def _local_base(tmp_path: Path) -> Path:
    path = tmp_path / "train_full.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"output_dir": "local-output", "seed": 13},
                "data": {
                    "teachers": [
                        "gigapath",
                        "h_optimus_1",
                        "uni2_h",
                        "virchow2",
                    ],
                    "prototype_paths": {
                        "gigapath": "/local/gigapath.pt",
                        "h_optimus_1": "/local/h1.pt",
                        "uni2_h": "/local/uni2.pt",
                        "virchow2": "/local/virchow2.pt",
                    },
                    "prototype_supervision_manifest_path": "/local/l1.csv",
                },
                "model": {
                    "teacher_dims": {
                        "gigapath": 1536,
                        "h_optimus_1": 1536,
                        "uni2_h": 1536,
                        "virchow2": 2560,
                    }
                },
                "loss": {
                    "teacher_weights": {
                        "gigapath": 1.0,
                        "h_optimus_1": 1.0,
                        "uni2_h": 1.0,
                        "virchow2": 1.0,
                    }
                },
                "train": {"epochs": 100, "lr_warmup_steps": 1000},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_ablation_resolver_preserves_full_population_and_full_replay(
    tmp_path: Path,
) -> None:
    resolved = resolve_ablation_config(
        _local_base(tmp_path),
        "experiments/ablation/configs/a1_no_prototype.yaml",
    )

    assert resolved["data"]["train_tile_fraction"] == 1.0
    assert resolved["data"]["val_tile_fraction"] == 1.0
    assert resolved["train"]["epochs"] == 10
    assert resolved["train"]["lr_warmup_steps"] == 1000
    assert resolved["data"]["prototype_supervision_manifest_path"] is None
    assert (
        resolved["data"]["expert_replay_prototype_manifest_path"]
        == "/local/l1.csv"
    )


def test_single_teacher_ablation_uses_local_matching_prototype(
    tmp_path: Path,
) -> None:
    resolved = resolve_ablation_config(
        _local_base(tmp_path),
        "experiments/ablation/configs/a4_single_teacher_prototype.yaml",
    )

    assert resolved["data"]["teachers"] == ["virchow2"]
    assert resolved["data"]["prototype_paths"] == {
        "virchow2": "/local/virchow2.pt"
    }


def test_ablation_seed_is_frozen_into_output_identity(
    tmp_path: Path,
) -> None:
    resolved = resolve_ablation_config(
        _local_base(tmp_path),
        "experiments/ablation/configs/a0_full_pamtd.yaml",
        seed=37,
    )

    assert resolved["runtime"]["seed"] == 37
    assert resolved["runtime"]["output_dir"].endswith(
        "a0_full_pamtd/seed_37"
    )


def test_global_and_spatial_prototype_refresh_ablate_separately(
    tmp_path: Path,
) -> None:
    global_static = resolve_ablation_config(
        _local_base(tmp_path),
        "experiments/ablation/configs/a5_static_global_prototypes.yaml",
    )
    spatial_static = resolve_ablation_config(
        _local_base(tmp_path),
        "experiments/ablation/configs/a6_static_spatial_prototypes.yaml",
    )

    assert global_static["train"]["dynamic_prototype_refresh_steps"] == 0
    assert global_static["train"].get(
        "dynamic_spatial_prototype_refresh_steps",
        500,
    ) != 0
    assert spatial_static["train"]["dynamic_spatial_prototype_refresh_steps"] == 0
    assert spatial_static["train"].get(
        "dynamic_prototype_refresh_steps",
        500,
    ) != 0


def test_every_tracked_ablation_condition_resolves(
    tmp_path: Path,
) -> None:
    conditions = sorted(
        Path("experiments/ablation/configs").glob("a[0-8]_*.yaml")
    )
    assert len(conditions) == 9

    for condition in conditions:
        resolved = resolve_ablation_config(
            _local_base(tmp_path),
            condition,
            seed=71,
        )
        assert resolved["data"]["train_tile_fraction"] == 1.0
        assert resolved["train"]["epochs"] == 10
        assert resolved["runtime"]["seed"] == 71
