from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from hcc_sempath.io.feature_cache import build_teacher_feature_package_from_tile_package
from hcc_sempath.io.tile_package import read_package_manifest


DEFAULT_TEACHER_DIMS = {
    "gigapath": 1536,
    "h_optimus_1": 1536,
    "uni2_h": 1536,
    "virchow2": 2560,
}


def _fake_features(record_count: int, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(record_count):
        yield rng.standard_normal(dim).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fake teacher packages and a prototype-free epoch-smoke config.")
    parser.add_argument("--tile-package", default="smoke_data/tiles.iac")
    parser.add_argument("--output-root", default="real_smoke/fake_teacher_epoch")
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    tile_package = Path(args.tile_package)
    output_root = Path(args.output_root)
    feature_root = output_root / "features"
    output_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    records = read_package_manifest(tile_package)
    needed = args.train_count + args.val_count
    if needed > len(records):
        raise ValueError(f"requested train+val records={needed}, package only has {len(records)}")

    for index, (teacher, dim) in enumerate(DEFAULT_TEACHER_DIMS.items()):
        teacher_dir = feature_root / teacher
        teacher_dir.mkdir(parents=True, exist_ok=True)
        feature_package = teacher_dir / f"{tile_package.stem}.{teacher}.features.iac"
        build_teacher_feature_package_from_tile_package(
            tile_package,
            _fake_features(len(records), dim, args.seed + index * 1000),
            feature_package,
            teacher_name=teacher,
            dtype="float16",
            feature_dim=dim,
            overwrite=True,
        )
        print(f"feature_package teacher={teacher} dim={dim} path={feature_package}")

    split_path = output_root / "tile_split.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile_id", "split"])
        writer.writeheader()
        for index, record in enumerate(records):
            if index < args.train_count:
                split = "train"
            elif index < needed:
                split = "val"
            else:
                split = "ignore"
            writer.writerow({"tile_id": record.tile_id, "split": split})

    config = {
        "runtime": {
            "device": "cpu",
            "seed": args.seed,
            "output_dir": str(output_root / "outputs"),
        },
        "data": {
            "image_tile_package_path": str(tile_package),
            "teacher_feature_package_paths": {
                teacher: [str(feature_root / teacher / f"{tile_package.stem}.{teacher}.features.iac")]
                for teacher in DEFAULT_TEACHER_DIMS
            },
            "split_manifest_path": str(split_path),
            "split_key": "tile_id",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_workers": 0,
        },
        "model": {
            "backbone_name": "vit_tiny_patch16_224",
            "embedding_dim": 384,
            "teacher_dims": DEFAULT_TEACHER_DIMS,
            "pretrained": False,
        },
        "loss": {
            "teacher_weights": {teacher: 1.0 for teacher in DEFAULT_TEACHER_DIMS},
            "relation_weight": 0.25,
            "semantic_weight": 0.0,
            "semantic_warmup_epochs": 0,
            "semantic_temperature": 1.0,
            "prototype_filter_weight": 0.0,
            "prototype_filter_warmup_epochs": 0,
            "prototype_filter_alpha_min": 0.25,
        },
        "train": {
            "batch_size": args.batch_size,
            "epochs": 1,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "amp": False,
            "topk": 2,
        },
    }
    config_path = output_root / "distill_fake_teacher_epoch.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"config path={config_path}")
    print(f"split path={split_path} train={args.train_count} val={args.val_count}")


if __name__ == "__main__":
    main()
