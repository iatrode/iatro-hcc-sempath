from __future__ import annotations

import argparse
import time

import torch

from iatro.iac.adapters.tiles import read_package_metadata
from .config import (
    embedding_dim,
    image_tile_package_paths,
    load_config,
    manifest_data_paths,
)
from ..modeling.models import (
    StudentEncoder,
    STUDENT_BACKBONE_NAME,
)
from .manifest import load_training_manifest
from .train import _paths_from_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark student encoder throughput.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    saved_cfg = payload.get("config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("checkpoint has no resolved training config")
    requested_encoder = {
        "embedding_dim": embedding_dim(cfg),
        "projector_type": cfg["model"].get("projector_type", "linear"),
        "projector_hidden_dim": int(
            cfg["model"].get("projector_hidden_dim", 2048)
        ),
        "mean": cfg["data"].get("mean"),
        "std": cfg["data"].get("std"),
    }
    saved_encoder = {
        "embedding_dim": embedding_dim(saved_cfg),
        "projector_type": saved_cfg["model"].get(
            "projector_type",
            "linear",
        ),
        "projector_hidden_dim": int(
            saved_cfg["model"].get("projector_hidden_dim", 2048)
        ),
        "mean": saved_cfg["data"].get("mean"),
        "std": saved_cfg["data"].get("std"),
    }
    if requested_encoder != saved_encoder:
        raise ValueError(
            "benchmark config differs from the checkpoint encoder contract"
        )
    device = torch.device(cfg["runtime"]["device"])
    if cfg["data"].get("train_manifest_path"):
        manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
        tile_packages, _ = manifest_data_paths(cfg, manifest, "train")
    elif "train_image_tile_package_paths" in cfg["data"]:
        tile_packages = _paths_from_data(
            cfg,
            "train_image_tile_package_paths",
        )
    else:
        tile_packages = image_tile_package_paths(cfg)
    tile_metadata = read_package_metadata(tile_packages[0])
    image_height = int(tile_metadata["tile_height"])
    image_width = int(tile_metadata["tile_width"])
    model = StudentEncoder(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(
            cfg["model"].get("projector_hidden_dim", 2048)
        ),
    ).to(device).eval()
    encoder_state = {}
    for key, value in payload["model"].items():
        normalized = key.removeprefix("_orig_mod.")
        if normalized.startswith("encoder."):
            encoder_state[normalized.removeprefix("encoder.")] = value
    if not encoder_state:
        raise ValueError("checkpoint has no student encoder state")
    model.load_state_dict(encoder_state)
    batch = torch.randn(cfg["train"]["batch_size"], 3, image_height, image_width, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tiles_per_sec = args.steps * batch.shape[0] / elapsed
    print(f"benchmark_ok tiles_per_sec={tiles_per_sec:.3f}")


if __name__ == "__main__":
    main()
