from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iatro.iac.adapters.tiles import read_package_metadata
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.training.config import (
    embedding_dim,
    load_config,
    manifest_data_paths,
    teacher_dims,
    teacher_names,
    validate_training_config,
)
from hcc_sempath.training.datasets import (
    DistillationTileDataset,
    collate_distillation,
    read_packaged_tile_records,
    validate_teacher_cache,
)
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.manifest import load_training_manifest


def _load_model(cfg: dict, checkpoint: Path, device: torch.device) -> HCCSemPathModel:
    names = teacher_names(cfg)
    dims = teacher_dims(cfg, names)
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        grad_checkpointing=False,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _write_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "tile_id",
        "slide_id",
        "patient_id",
        "dataset",
        "split",
        "tile_package_path",
        "row_index",
        "x",
        "y",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_split(cfg: dict, checkpoint: Path, split: str, output_dir: Path) -> None:
    device = torch.device(cfg["runtime"]["device"])
    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, split)
    names = teacher_names(cfg)
    validate_training_config(cfg, names)
    dims = teacher_dims(cfg, names)
    tile_metadata = read_package_metadata(tile_packages[0])
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    records = read_packaged_tile_records(tile_packages)
    validate_teacher_cache(records, None, dims, teacher_cache_package_paths=teacher_packages)
    dataset = DistillationTileDataset(
        records,
        None,
        image_size,
        mean=cfg["data"].get("mean"),
        std=cfg["data"].get("std"),
        teacher_cache_package_paths=teacher_packages,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        collate_fn=collate_distillation,
    )
    model = _load_model(cfg, checkpoint, device)
    max_batches = int(cfg["train"].get("max_eval_batches", 0)) or None
    embeddings = []
    teachers = {name: [] for name in names}
    metadata_rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            outputs = model(_prepare_images(batch, cfg, device))
            embeddings.append(outputs["embedding_norm"].detach().cpu().numpy().astype("float32"))
            for name in names:
                teachers[name].append(batch["teacher_features"][name].cpu().numpy().astype("float32"))
            for tile_id in batch["tile_id"]:
                item = records[len(metadata_rows)]
                record = item.record
                package_path = str(item.tile_package_path or "")
                dataset_name = Path(package_path).parent.name
                metadata_rows.append({
                    "tile_id": tile_id,
                    "slide_id": record.slide_id,
                    "patient_id": record.patient_id,
                    "dataset": dataset_name,
                    "split": split,
                    "tile_package_path": package_path,
                    "row_index": len(metadata_rows),
                    "x": record.x,
                    "y": record.y,
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"student_embeddings_{split}.npz", embedding_norm=np.concatenate(embeddings))
    for name, values in teachers.items():
        np.savez_compressed(output_dir / f"teacher_embeddings_{name}_{split}.npz", embedding_norm=np.concatenate(values))
    _write_metadata(output_dir / f"tile_metadata_{split}.csv", metadata_rows)
    provenance = {
        "split": split,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "tile_packages": tile_packages,
        "teacher_packages": teacher_packages,
        "exported_tiles": len(metadata_rows),
        "max_eval_batches": max_batches,
    }
    (output_dir / f"export_manifest_{split}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"embedding_export_ok split={split} tiles={len(metadata_rows)} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for split in args.split or ["val", "exval"]:
        export_split(cfg, Path(args.checkpoint), split, Path(args.output_dir))


if __name__ == "__main__":
    main()
