from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..io.feature_cache import FeatureCacheReader
from ..io.manifests import TileRecord
from ..io.tile_package import TilePackageReader


@dataclass(frozen=True)
class PackagedTileRecord:
    record: TileRecord
    tile_package_path: Path | None = None


def _record_tile_id(record: TileRecord | PackagedTileRecord) -> str:
    return record.record.tile_id if isinstance(record, PackagedTileRecord) else record.tile_id


def _unwrap_record(record: TileRecord | PackagedTileRecord) -> TileRecord:
    return record.record if isinstance(record, PackagedTileRecord) else record


def _record_package_path(record: TileRecord | PackagedTileRecord) -> Path | None:
    return record.tile_package_path if isinstance(record, PackagedTileRecord) else None


def read_packaged_tile_records(package_paths: list[str | Path]) -> list[PackagedTileRecord]:
    from ..io.tile_package import read_package_manifest

    records: list[PackagedTileRecord] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for package_path in package_paths:
        path = Path(package_path)
        for record in read_package_manifest(path):
            if record.tile_id in seen:
                duplicates.append(record.tile_id)
            seen.add(record.tile_id)
            records.append(PackagedTileRecord(record=record, tile_package_path=path))
    if duplicates:
        sample = ", ".join(duplicates[:3])
        raise ValueError(f"duplicate tile_id values across tile packages: count={len(duplicates)} sample={sample}")
    return records


def apply_split_overrides(
    records: list[PackagedTileRecord],
    split_manifest_path: str | Path | None,
    split_key: str = "slide_id",
) -> list[PackagedTileRecord]:
    if split_manifest_path is None:
        return records
    if split_key not in {"slide_id", "patient_id", "tile_id"}:
        raise ValueError(f"unsupported split_key: {split_key}")
    split_map: dict[str, str] = {}
    with Path(split_manifest_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {split_key, "split"}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split manifest missing columns: {sorted(missing)}")
        for row in reader:
            split_map[row[split_key]] = row["split"]
    updated = []
    missing_keys: set[str] = set()
    for item in records:
        record = item.record
        key = getattr(record, split_key)
        split = split_map.get(key)
        if split is None:
            missing_keys.add(key)
            updated.append(item)
        else:
            updated.append(PackagedTileRecord(record=replace(record, split=split), tile_package_path=item.tile_package_path))
    if missing_keys:
        sample = ", ".join(sorted(missing_keys)[:3])
        raise ValueError(f"split manifest missing {split_key} values: count={len(missing_keys)} sample={sample}")
    return updated


def resolve_teacher_feature_packages(
    teacher_cache_package_path: str | Path | None = None,
    teacher_cache_package_paths: (
        dict[str, str | Path | list[str | Path] | tuple[str | Path, ...]]
        | list[str | Path]
        | tuple[str | Path, ...]
        | None
    ) = None,
) -> dict[str, list[Path]]:
    packages: dict[str, list[Path]] = {}
    if teacher_cache_package_paths is not None:
        if isinstance(teacher_cache_package_paths, dict):
            for name, value in teacher_cache_package_paths.items():
                if isinstance(value, (list, tuple)):
                    packages[str(name)] = [Path(path) for path in value]
                else:
                    packages[str(name)] = [Path(value)]
        else:
            for path in teacher_cache_package_paths:
                reader = FeatureCacheReader(path)
                try:
                    teacher_name = str(reader.header.get("teacher") or Path(path).stem)
                finally:
                    reader.close()
                if teacher_name in packages:
                    raise ValueError(f"duplicate teacher feature package name: {teacher_name}")
                packages[teacher_name] = [Path(path)]
    elif teacher_cache_package_path is not None:
        reader = FeatureCacheReader(teacher_cache_package_path)
        try:
            teacher_name = str(reader.header.get("teacher") or "teacher")
        finally:
            reader.close()
        packages[teacher_name] = [Path(teacher_cache_package_path)]
    if not packages:
        raise ValueError("at least one teacher feature package is required")
    return packages


class _FeatureReaderSet:
    def __init__(self, package_paths: list[Path]) -> None:
        self.readers = [FeatureCacheReader(path) for path in package_paths]
        self.tile_to_reader: dict[str, FeatureCacheReader] = {}
        for reader in self.readers:
            tile_ids = reader.record_table.column("tile_id").to_pylist()
            for tile_id in tile_ids:
                if tile_id in self.tile_to_reader:
                    raise ValueError(f"duplicate tile_id across feature packages: {tile_id}")
                self.tile_to_reader[tile_id] = reader

    def read_feature(self, tile_id: str):
        reader = self.tile_to_reader.get(tile_id)
        if reader is None:
            raise FileNotFoundError(f"missing packaged teacher feature: {tile_id}")
        return reader.read_feature(tile_id)

    def close(self) -> None:
        for reader in self.readers:
            reader.close()

    def __del__(self) -> None:
        self.close()


class DistillationTileDataset(Dataset):
    def __init__(
        self,
        records: list[TileRecord | PackagedTileRecord],
        teacher_cache_dir: str | Path | None,
        image_size: int | tuple[int, int],
        mean: list[float] | tuple[float, ...] | None = None,
        std: list[float] | tuple[float, ...] | None = None,
        tile_package_path: str | Path | None = None,
        teacher_cache_package_path: str | Path | None = None,
        teacher_cache_package_paths: (
            dict[str, str | Path | list[str | Path] | tuple[str | Path, ...]]
            | list[str | Path]
            | tuple[str | Path, ...]
            | None
        ) = None,
    ) -> None:
        self.records = records
        if teacher_cache_dir is not None:
            raise ValueError("loose teacher feature directories are not supported; use a .features.iac package")
        self.package_reader = TilePackageReader(tile_package_path) if tile_package_path else None
        self.package_readers: dict[Path, TilePackageReader] = {}
        for record in records:
            package_path = _record_package_path(record)
            if package_path is not None and package_path not in self.package_readers:
                self.package_readers[package_path] = TilePackageReader(package_path)
        self.teacher_package_paths = resolve_teacher_feature_packages(
            teacher_cache_package_path=teacher_cache_package_path,
            teacher_cache_package_paths=teacher_cache_package_paths,
        )
        self.feature_readers = {name: _FeatureReaderSet(paths) for name, paths in self.teacher_package_paths.items()}
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
        item = self.records[index]
        record = _unwrap_record(item)
        package_path = _record_package_path(item)
        if package_path is not None:
            image = self.package_readers[package_path].read_image(record.tile_id)
            image_tensor = self.transform(image.convert("RGB"))
        elif self.package_reader is not None:
            image = self.package_reader.read_image(record.tile_id)
            image_tensor = self.transform(image.convert("RGB"))
        else:
            with Image.open(record.tile_path) as image:
                image_tensor = self.transform(image.convert("RGB"))
        teacher_features = {}
        for name, reader in self.feature_readers.items():
            teacher_feature = torch.from_numpy(reader.read_feature(record.tile_id))
            if teacher_feature.ndim != 1:
                raise ValueError(f"teacher feature must be 1D: teacher={name} tile_id={record.tile_id}")
            teacher_features[name] = teacher_feature
        return {
            "tile_id": record.tile_id,
            "image": image_tensor,
            "teacher_features": teacher_features,
        }


def validate_teacher_cache(
    records: list[TileRecord | PackagedTileRecord],
    teacher_cache_dir: str | Path | None = None,
    expected_dim: int | dict[str, int] | None = None,
    teacher_cache_package_path: str | Path | None = None,
    teacher_cache_package_paths: (
        dict[str, str | Path | list[str | Path] | tuple[str | Path, ...]]
        | list[str | Path]
        | tuple[str | Path, ...]
        | None
    ) = None,
) -> None:
    packages = resolve_teacher_feature_packages(
        teacher_cache_package_path=teacher_cache_package_path,
        teacher_cache_package_paths=teacher_cache_package_paths,
    )
    for name, package_paths in packages.items():
        reader = _FeatureReaderSet(package_paths)
        try:
            if isinstance(expected_dim, dict):
                dim = expected_dim.get(name)
            else:
                dim = expected_dim
            wrong_shape = []
            for item in records:
                tile_id = _record_tile_id(item)
                feature = reader.read_feature(tile_id)
                if feature.ndim != 1 or (dim is not None and feature.shape[0] != dim):
                    wrong_shape.append(f"{tile_id}:{tuple(feature.shape)}")
            if wrong_shape:
                sample = ", ".join(wrong_shape[:3])
                raise ValueError(
                    f"invalid packaged teacher feature shapes: teacher={name} count={len(wrong_shape)} sample={sample}"
                )
        finally:
            reader.close()


def collate_distillation(batch: list[dict]) -> dict:
    teacher_names = list(batch[0]["teacher_features"].keys())
    return {
        "tile_id": [item["tile_id"] for item in batch],
        "images": torch.stack([item["image"] for item in batch]),
        "teacher_features": {
            name: torch.stack([item["teacher_features"][name] for item in batch]) for name in teacher_names
        },
    }
