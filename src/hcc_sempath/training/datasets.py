from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..io.feature_cache import FeatureCacheReader
from ..io.manifests import TileRecord
from ..io.tile_package import TilePackageReader


class DistillationTileDataset(Dataset):
    def __init__(
        self,
        records: list[TileRecord],
        teacher_cache_dir: str | Path | None,
        image_size: int | tuple[int, int],
        mean: list[float] | tuple[float, ...] | None = None,
        std: list[float] | tuple[float, ...] | None = None,
        tile_package_path: str | Path | None = None,
        teacher_cache_package_path: str | Path | None = None,
    ) -> None:
        self.records = records
        if teacher_cache_dir is not None:
            raise ValueError("loose teacher feature directories are not supported; use a .features.iac package")
        self.package_reader = TilePackageReader(tile_package_path) if tile_package_path else None
        self.feature_reader = FeatureCacheReader(teacher_cache_package_path) if teacher_cache_package_path else None
        if self.feature_reader is None:
            raise ValueError("teacher_cache_package_path is required")
        resize_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        transform_steps = [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            transform_steps.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if self.package_reader is not None:
            image = self.package_reader.read_image(record.tile_id)
            image_tensor = self.transform(image.convert("RGB"))
        else:
            with Image.open(record.tile_path) as image:
                image_tensor = self.transform(image.convert("RGB"))
        teacher_feature = torch.from_numpy(self.feature_reader.read_feature(record.tile_id))
        if teacher_feature.ndim != 1:
            raise ValueError(f"teacher feature must be 1D: {record.tile_id}")
        return {
            "tile_id": record.tile_id,
            "image": image_tensor,
            "teacher_feature": teacher_feature,
        }


def validate_teacher_cache(
    records: list[TileRecord],
    teacher_cache_dir: str | Path | None = None,
    expected_dim: int | None = None,
    teacher_cache_package_path: str | Path | None = None,
) -> None:
    if teacher_cache_package_path:
        reader = FeatureCacheReader(teacher_cache_package_path)
        try:
            wrong_shape = []
            for record in records:
                feature = reader.read_feature(record.tile_id)
                if feature.ndim != 1 or (expected_dim is not None and feature.shape[0] != expected_dim):
                    wrong_shape.append(f"{record.tile_id}:{tuple(feature.shape)}")
            if wrong_shape:
                sample = ", ".join(wrong_shape[:3])
                raise ValueError(f"invalid packaged teacher feature shapes: count={len(wrong_shape)} sample={sample}")
        finally:
            reader.close()
        return
    raise ValueError("teacher_cache_package_path is required; loose .npy teacher caches are not supported")


def collate_distillation(batch: list[dict]) -> dict:
    return {
        "tile_id": [item["tile_id"] for item in batch],
        "images": torch.stack([item["image"] for item in batch]),
        "teacher_features": torch.stack([item["teacher_feature"] for item in batch]),
    }
