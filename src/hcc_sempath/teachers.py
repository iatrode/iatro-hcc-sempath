from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from tqdm import tqdm
from .manifests import read_tile_manifest
from .tile_package import iter_package_tiles, read_package_manifest


class TimmTeacherEncoder(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        model_path = Path(model_name)
        if model_path.is_dir():
            config_path = model_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"missing local teacher config: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            architecture = config.get("architecture")
            if not architecture:
                raise ValueError(f"local teacher config missing architecture: {config_path}")
            model_args = dict(config.get("model_args", {}))
            model_args.setdefault("num_classes", 0)
            self.model = timm.create_model(architecture, pretrained=False, **model_args)
            weight_path = model_path / "pytorch_model.bin"
            if pretrained:
                if not weight_path.exists():
                    raise FileNotFoundError(f"missing local teacher weights: {weight_path}")
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


@torch.no_grad()
def cache_teacher_features(
    model: torch.nn.Module,
    manifest_path: str | Path,
    output_dir: str | Path,
    image_size: int,
    batch_size: int,
    device: str,
) -> None:
    records = read_tile_manifest(manifest_path)
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_only = [record for record in records if not (cache_dir / f"{record.tile_id}.npy").exists()]
    # Reuse the standard dataset by placing temporary zero vectors is intentionally avoided;
    # teacher extraction does not need pre-existing cache files.
    from PIL import Image

    data_config = resolve_model_data_config(model.model)
    data_config["input_size"] = (3, image_size, image_size)
    transform = create_transform(**data_config, is_training=False)
    model = model.to(device).eval()
    for start in tqdm(range(0, len(missing_only), batch_size), desc="teacher batches"):
        chunk = missing_only[start : start + batch_size]
        images = []
        for record in chunk:
            with Image.open(record.tile_path) as image:
                images.append(transform(image.convert("RGB")))
        batch = torch.stack(images).to(device)
        features = model(batch).detach().cpu().numpy().astype(np.float32)
        for record, feature in zip(chunk, features):
            np.save(cache_dir / f"{record.tile_id}.npy", feature)


@torch.no_grad()
def cache_teacher_features_from_package(
    model: torch.nn.Module,
    package_path: str | Path,
    output_dir: str | Path,
    image_size: int,
    batch_size: int,
    device: str,
) -> None:
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_config = resolve_model_data_config(model.model)
    data_config["input_size"] = (3, image_size, image_size)
    transform = create_transform(**data_config, is_training=False)
    model = model.to(device).eval()
    total_records = len(read_package_manifest(package_path))
    tile_ids = []
    images = []
    progress = tqdm(total=total_records, desc="teacher tiles")
    try:
        for record, image in iter_package_tiles(package_path):
            progress.update(1)
            if (cache_dir / f"{record.tile_id}.npy").exists():
                continue
            tile_ids.append(record.tile_id)
            images.append(transform(image.convert("RGB")))
            if len(images) >= batch_size:
                batch = torch.stack(images).to(device)
                features = model(batch).detach().cpu().numpy().astype(np.float32)
                for tile_id, feature in zip(tile_ids, features):
                    np.save(cache_dir / f"{tile_id}.npy", feature)
                tile_ids = []
                images = []
    finally:
        progress.close()
    if images:
        batch = torch.stack(images).to(device)
        features = model(batch).detach().cpu().numpy().astype(np.float32)
        for tile_id, feature in zip(tile_ids, features):
            np.save(cache_dir / f"{tile_id}.npy", feature)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache teacher features from a tile manifest or IatroCache package.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--tile-package", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="hf_hub:bioptimus/H-optimus-1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.tile_package):
        raise ValueError("provide exactly one of --manifest or --tile-package")
    model = TimmTeacherEncoder(args.model_name, pretrained=args.pretrained)
    if args.tile_package:
        cache_teacher_features_from_package(
            model=model,
            package_path=args.tile_package,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        cache_teacher_features(
            model=model,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
        )
    print(f"cache_ok output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
