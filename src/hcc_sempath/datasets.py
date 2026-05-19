from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .manifests import TileRecord
from .tile_package import TilePackageReader


class DistillationTileDataset(Dataset):
    def __init__(
        self,
        records: list[TileRecord],
        teacher_cache_dir: str | Path,
        image_size: int,
        mean: list[float] | tuple[float, ...] | None = None,
        std: list[float] | tuple[float, ...] | None = None,
        tile_package_path: str | Path | None = None,
    ) -> None:
        self.records = records
        self.teacher_cache_dir = Path(teacher_cache_dir)
        self.package_reader = TilePackageReader(tile_package_path) if tile_package_path else None
        transform_steps = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            transform_steps.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        teacher_path = self.teacher_cache_dir / f"{record.tile_id}.npy"
        if not teacher_path.exists():
            raise FileNotFoundError(f"missing teacher feature: {teacher_path}")
        if self.package_reader is not None:
            image = self.package_reader.read_image(record.tile_id)
            image_tensor = self.transform(image.convert("RGB"))
        else:
            with Image.open(record.tile_path) as image:
                image_tensor = self.transform(image.convert("RGB"))
        teacher_feature = torch.from_numpy(np.load(teacher_path).astype(np.float32))
        if teacher_feature.ndim != 1:
            raise ValueError(f"teacher feature must be 1D: {teacher_path}")
        return {
            "tile_id": record.tile_id,
            "image": image_tensor,
            "teacher_feature": teacher_feature,
        }


def validate_teacher_cache(
    records: list[TileRecord],
    teacher_cache_dir: str | Path,
    expected_dim: int | None = None,
) -> None:
    teacher_cache_dir = Path(teacher_cache_dir)
    missing = []
    wrong_shape = []
    for record in records:
        teacher_path = teacher_cache_dir / f"{record.tile_id}.npy"
        if not teacher_path.exists():
            missing.append(str(teacher_path))
            continue
        feature = np.load(teacher_path, mmap_mode="r")
        if feature.ndim != 1 or (expected_dim is not None and feature.shape[0] != expected_dim):
            wrong_shape.append(f"{teacher_path}:{tuple(feature.shape)}")
    if missing:
        sample = ", ".join(missing[:3])
        raise FileNotFoundError(f"missing teacher features: count={len(missing)} sample={sample}")
    if wrong_shape:
        sample = ", ".join(wrong_shape[:3])
        raise ValueError(f"invalid teacher feature shapes: count={len(wrong_shape)} sample={sample}")


def collate_distillation(batch: list[dict]) -> dict:
    return {
        "tile_id": [item["tile_id"] for item in batch],
        "images": torch.stack([item["image"] for item in batch]),
        "teacher_features": torch.stack([item["teacher_feature"] for item in batch]),
    }
