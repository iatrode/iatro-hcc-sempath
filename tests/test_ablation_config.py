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
                "train": {"epochs": 100},
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
