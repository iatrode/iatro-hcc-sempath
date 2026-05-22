from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..io.tile_package import read_package_manifest, read_package_metadata
from ..modeling.anchors import load_anchors
from ..modeling.models import StudentEncoder
from .config import load_config
from .datasets import DistillationTileDataset, collate_distillation, validate_teacher_cache
from .engine import collect_embeddings
from .metrics import evaluate_embeddings
from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained HCC-SemPath checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg["runtime"]["device"])
    image_tile_package_path = cfg["data"]["image_tile_package_path"]
    teacher_feature_package_path = cfg["data"]["teacher_feature_package_path"]
    tile_metadata = read_package_metadata(image_tile_package_path)
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    all_records = read_package_manifest(image_tile_package_path)
    records = [record for record in all_records if record.split == args.split]
    validate_teacher_cache(
        records,
        None,
        cfg["model"]["teacher_dim"],
        teacher_cache_package_path=teacher_feature_package_path,
    )
    dataset = DistillationTileDataset(
        records,
        None,
        image_size,
        mean=cfg["data"].get("mean"),
        std=cfg["data"].get("std"),
        tile_package_path=image_tile_package_path,
        teacher_cache_package_path=teacher_feature_package_path,
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=collate_distillation)
    model = StudentEncoder(cfg["model"]["backbone_name"], cfg["model"]["teacher_dim"], cfg["model"]["pretrained"]).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    student, teacher = collect_embeddings(model, loader, device)
    anchors = load_anchors(cfg["data"]["anchors_path"], expected_dim=cfg["model"]["teacher_dim"])
    metrics = evaluate_embeddings(student, teacher, anchors, int(cfg["train"]["topk"]))
    write_json(f"{cfg['runtime']['output_dir']}/eval_{args.split}.json", metrics)
    print("eval_ok " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
