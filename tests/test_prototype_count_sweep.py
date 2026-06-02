from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import torch
import yaml


def _load_sweep_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prototype_count_sweep.py"
    spec = importlib.util.spec_from_file_location("prototype_count_sweep", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_prototype_package(path: Path) -> None:
    torch.save(
        {
            "version": 1,
            "prototypes": torch.randn(5, 4),
            "names": [
                "primary_tumor",
                "primary_non_tumor",
                "lymphocyte_rich",
                "fibrotic_stroma",
                "necrotic",
            ],
            "groups": ["primary", "primary", "immune", "stroma", "degeneration"],
            "levels": [1, 1, 2, 2, 2],
            "exclusive": [True, True, False, False, False],
        },
        path,
    )


def test_prototype_count_sweep_dry_run_writes_self_contained_experiment_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sweep = _load_sweep_module()
    zhcc_path = tmp_path / "zhcc.pt"
    teacher_path = tmp_path / "teacher.pt"
    _write_prototype_package(zhcc_path)
    _write_prototype_package(teacher_path)
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "tile_id,slide,l1,l2\n"
        "tile_1,slide_1,primary_tumor,lymphocyte_rich;fibrotic_stroma\n"
        "tile_2,slide_2,primary_non_tumor,necrotic\n"
        "tile_3,slide_3,primary_tumor,lymphocyte_rich\n",
        encoding="utf-8",
    )
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "cpu", "seed": 13, "output_dir": "unused"},
                "data": {
                    "teachers": ["toy"],
                    "prototype_paths": {"toy": str(teacher_path)},
                    "zhcc_prototype_path": str(zhcc_path),
                },
                "model": {"embedding_dim": 4, "teacher_dims": {"toy": 4}},
                "loss": {},
                "train": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "sweep"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prototype_count_sweep.py",
            "--base-config",
            str(base_config),
            "--input-path",
            str(annotation_path),
            "--output-root",
            str(output_root),
            "--prototype-counts",
            "4,5",
            "--seed",
            "17",
            "--dry-run",
        ],
    )

    sweep.main()

    experiment_dir = output_root / "count_004"
    assert not (output_root / "runs").exists()
    assert (experiment_dir / "config.yaml").exists()
    assert (experiment_dir / "prototype_supervision.csv").exists()
    assert (experiment_dir / "prototypes" / "zhcc_prototypes.pt").exists()
    assert (experiment_dir / "prototypes" / "toy_prototypes.pt").exists()
    assert not (experiment_dir / "reproduce.sh").exists()
    assert not (experiment_dir / "run_manifest.json").exists()
    cfg = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["runtime"]["seed"] == 17
    assert cfg["runtime"]["output_dir"] == str(experiment_dir / "training")
    assert cfg["data"]["zhcc_prototype_path"] == str(experiment_dir / "prototypes" / "zhcc_prototypes.pt")
    assert cfg["data"]["prototype_paths"]["toy"] == str(experiment_dir / "prototypes" / "toy_prototypes.pt")
    assert cfg["data"]["prototype_supervision_manifest_path"] == str(experiment_dir / "prototype_supervision.csv")
    with (output_root / "sweep_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["prototype_count_requested"] for row in rows] == ["4", "5"]
    assert {row["seed"] for row in rows} == {"17"}
    assert {row["status"] for row in rows} == {"dry_run"}
    assert {row["returncode"] for row in rows} == {"0"}
