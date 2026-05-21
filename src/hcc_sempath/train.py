from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .anchors import load_anchors
from .config import load_config
from .datasets import DistillationTileDataset, collate_distillation, validate_teacher_cache
from .engine import fit
from .models import StudentEncoder
from .tile_package import read_package_manifest
from .utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HCC-SemPath distillation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["runtime"]["seed"]))
    device = torch.device(cfg["runtime"]["device"])
    image_tile_package_path = cfg["data"]["image_tile_package_path"]
    teacher_feature_package_path = cfg["data"]["teacher_feature_package_path"]
    records = read_package_manifest(image_tile_package_path)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    if not train_records or not val_records:
        raise ValueError("manifest must contain non-empty train and val splits")
    validate_teacher_cache(
        records,
        None,
        cfg["model"]["teacher_dim"],
        teacher_cache_package_path=teacher_feature_package_path,
    )
    dataset_kwargs = {
        "teacher_cache_dir": None,
        "image_size": cfg["data"]["image_size"],
        "mean": cfg["data"].get("mean"),
        "std": cfg["data"].get("std"),
        "tile_package_path": image_tile_package_path,
        "teacher_cache_package_path": teacher_feature_package_path,
    }
    train_ds = DistillationTileDataset(train_records, **dataset_kwargs)
    val_ds = DistillationTileDataset(val_records, **dataset_kwargs)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], collate_fn=collate_distillation)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=collate_distillation)
    anchors = load_anchors(cfg["data"]["anchors_path"], expected_dim=cfg["model"]["teacher_dim"]).to(device)
    model = StudentEncoder(cfg["model"]["backbone_name"], cfg["model"]["teacher_dim"], cfg["model"]["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
    metrics = fit(model, train_loader, val_loader, anchors, optimizer, device, cfg)
    print("train_ok " + " ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
