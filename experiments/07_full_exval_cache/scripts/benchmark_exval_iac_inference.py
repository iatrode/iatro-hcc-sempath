from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hcc_sempath.io.tile_package import TilePackageReader, read_package_manifest, read_package_metadata
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.training.config import embedding_dim, load_config, manifest_data_paths, teacher_dims, teacher_names
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


def _localize_cfg(cfg: dict, manifest: str, prototype_dir: str, device: str, batch_size: int) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("runtime", {})["device"] = device
    cfg.setdefault("data", {})["train_manifest_path"] = manifest
    cfg["data"]["num_workers"] = 0
    cfg["data"]["val_tile_fraction"] = 1.0
    cfg["data"]["exval_tile_fraction"] = 1.0
    cfg["data"]["prototype_paths"] = {
        "gigapath": f"{prototype_dir}/gigapath_hcc_semantic_prototypes.pt",
        "h_optimus_1": f"{prototype_dir}/h_optimus_1_hcc_semantic_prototypes.pt",
        "uni2_h": f"{prototype_dir}/uni2_h_hcc_semantic_prototypes.pt",
        "virchow2": f"{prototype_dir}/virchow2_hcc_semantic_prototypes.pt",
    }
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["train"]["amp"] = False
    cfg["model"]["grad_checkpointing"] = False
    return cfg


def _batch_tensor(arrays: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real exval IAC decode + final-model MPS inference.")
    parser.add_argument("--config", default="artifacts/models/hcc-sempath-full/resolved_config.json")
    parser.add_argument("--checkpoint", default="artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt")
    parser.add_argument("--manifest", default="configs/local/mac/manifest.yaml")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--split", default="exval")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tiles", type=int, default=1024)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--output", default="experiments/07_full_exval_cache/reports/exval_iac_inference_benchmark.json")
    args = parser.parse_args()

    cfg = _localize_cfg(load_config(args.config), args.manifest, args.prototype_dir, args.device, args.batch_size)
    device = torch.device(cfg["runtime"]["device"])
    model = _load_model(cfg, Path(args.checkpoint), device)
    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, _ = manifest_data_paths(cfg, manifest, args.split)
    metadata = read_package_metadata(tile_packages[0])
    expected_size = (int(metadata["tile_height"]), int(metadata["tile_width"]))

    total_available = sum(len(read_package_manifest(path)) for path in tile_packages)
    target = min(int(args.max_tiles), total_available)
    read_seconds = 0.0
    infer_seconds = 0.0
    total_tiles = 0
    total_batches = 0
    warmup_batches_left = int(args.warmup_batches)
    first_tile_ids: list[str] = []
    last_tile_id = ""
    started = time.perf_counter()
    batch_arrays: list[np.ndarray] = []

    def run_batch() -> None:
        nonlocal infer_seconds, total_batches, warmup_batches_left
        if not batch_arrays:
            return
        batch = {"images": _batch_tensor(batch_arrays), "images_uint8": True}
        if warmup_batches_left > 0:
            with torch.no_grad():
                model(_prepare_images(batch, cfg, device))
            warmup_batches_left -= 1
            return
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(_prepare_images(batch, cfg, device))
            _ = outputs["embedding_norm"].detach().cpu()
        infer_seconds += time.perf_counter() - t0
        total_batches += 1

    for package_path in tile_packages:
        reader = TilePackageReader(package_path)
        try:
            count = reader.record_count
            for row in range(count):
                if total_tiles >= target:
                    break
                t0 = time.perf_counter()
                arr = reader.read_array_at(row)
                tile_id = reader.tile_id_at(row)
                read_seconds += time.perf_counter() - t0
                if arr.shape[:2] != expected_size:
                    raise ValueError(f"tile shape mismatch: {tile_id} got={arr.shape[:2]} expected={expected_size}")
                if len(first_tile_ids) < 3:
                    first_tile_ids.append(tile_id)
                last_tile_id = tile_id
                batch_arrays.append(arr)
                total_tiles += 1
                if len(batch_arrays) >= int(args.batch_size):
                    run_batch()
                    batch_arrays.clear()
            if total_tiles >= target:
                break
        finally:
            reader.close()
    if batch_arrays:
        run_batch()

    elapsed = time.perf_counter() - started
    measured_tiles = max(0, total_tiles - int(args.warmup_batches) * int(args.batch_size))
    measured_tiles = min(measured_tiles, total_batches * int(args.batch_size))
    result = {
        "split": args.split,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "target_tiles": target,
        "total_available_tiles": total_available,
        "processed_tiles_including_warmup": total_tiles,
        "measured_tiles": measured_tiles,
        "measured_batches": total_batches,
        "elapsed_seconds_total": elapsed,
        "read_decode_seconds": read_seconds,
        "inference_seconds_measured": infer_seconds,
        "end_to_end_tiles_per_sec": total_tiles / elapsed if elapsed > 0 else 0.0,
        "inference_tiles_per_sec": measured_tiles / infer_seconds if infer_seconds > 0 else 0.0,
        "first_tile_ids": first_tile_ids,
        "last_tile_id": last_tile_id,
        "checkpoint": args.checkpoint,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "exval_iac_benchmark_ok "
        f"device={result['device']} batch_size={result['batch_size']} "
        f"tiles={result['processed_tiles_including_warmup']} "
        f"end_to_end_tiles_per_sec={result['end_to_end_tiles_per_sec']:.2f} "
        f"inference_tiles_per_sec={result['inference_tiles_per_sec']:.2f} "
        f"output={out}"
    )


if __name__ == "__main__":
    main()
