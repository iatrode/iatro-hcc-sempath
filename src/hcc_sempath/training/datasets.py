from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Iterator
import threading

import torch
import numpy as np
from PIL import Image

from . import _pipeline_probe as _probe
from torch.utils.data import Dataset
from torchvision import transforms

from iatro.iac.adapters.features import FeatureCacheReader
from iatro.iac import read_header, read_tables
from iatro.iac.adapters.manifests import TileRecord
from iatro.iac.adapters.tiles import TilePackageReader
from .feature_pack_merge import (
    MERGED_FEATURE_PAYLOAD_TYPE,
    MERGED_FEATURE_SUFFIX,
    MergedTeacherFeatureCacheReader,
)
from .prototype_labels import PrototypeLabel
from .roi import SpatialRoiTarget, spatial_roi_payload


@dataclass(frozen=True)
class PackagedTileRecord:
    record: TileRecord
    tile_package_path: Path | None = None


def _build_image_transform(
    image_size: int | tuple[int, int],
    mean: list[float] | tuple[float, ...] | None,
    std: list[float] | tuple[float, ...] | None,
    *,
    resize: bool = True,
) -> transforms.Compose:
    resize_size = (image_size, image_size) if isinstance(image_size, int) else image_size
    transform_steps = []
    if resize:
        transform_steps.append(transforms.Resize(resize_size))
    transform_steps.append(transforms.ToTensor())
    if mean is not None and std is not None:
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_steps)


def _prototype_payload(tile_id: str, prototype_labels: dict[str, PrototypeLabel] | None) -> dict:
    if not prototype_labels:
        return {
            "prototype_mask": False,
            "prototype_level1": -1,
        }
    label = prototype_labels.get(tile_id)
    if label is None:
        return {
            "prototype_mask": False,
            "prototype_level1": -1,
        }
    return {
        "prototype_mask": True,
        "prototype_level1": label.level1,
    }


def _record_tile_id(record: TileRecord | PackagedTileRecord) -> str:
    return record.record.tile_id if isinstance(record, PackagedTileRecord) else record.tile_id


def _unwrap_record(record: TileRecord | PackagedTileRecord) -> TileRecord:
    return record.record if isinstance(record, PackagedTileRecord) else record


def _record_package_path(record: TileRecord | PackagedTileRecord) -> Path | None:
    return record.tile_package_path if isinstance(record, PackagedTileRecord) else None


def read_packaged_tile_records(package_paths: list[str | Path]) -> list[PackagedTileRecord]:
    from iatro.iac.adapters.tiles import read_package_manifest

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
                header = read_header(path)
                if header.get("payload_type") == MERGED_FEATURE_PAYLOAD_TYPE:
                    for teacher_name in header.get("teachers", []):
                        packages.setdefault(str(teacher_name), []).append(Path(path))
                else:
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


def _open_feature_source(path: Path):
    header = read_header(path)
    if header.get("payload_type") == MERGED_FEATURE_PAYLOAD_TYPE:
        return MergedTeacherFeatureCacheReader(path)
    return FeatureCacheReader(path)


def _source_has_teacher(source, teacher_name: str) -> bool:
    if isinstance(source, MergedTeacherFeatureCacheReader):
        return source.has_teacher(teacher_name)
    return True


def _read_feature_from_source(source, row: int, teacher_name: str):
    if isinstance(source, MergedTeacherFeatureCacheReader):
        return source.read_feature_at(row, teacher_name)
    return source.read_feature_at(row)


def _read_teacher_features_at(feature_sources: dict[Path, object], teacher_paths: dict[str, Path], row: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    merged_groups: dict[Path, list[str]] = {}
    for name, path in teacher_paths.items():
        source = feature_sources[path]
        if isinstance(source, MergedTeacherFeatureCacheReader):
            merged_groups.setdefault(path, []).append(name)
        else:
            result[name] = source.read_feature_at(row)
    for path, names in merged_groups.items():
        source = feature_sources[path]
        result.update(source.read_features_at(row, names))
    return result


def _read_teacher_features_many_at(
    feature_sources: dict[Path, object],
    teacher_paths: dict[str, Path],
    rows: list[int],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    merged_groups: dict[Path, list[str]] = {}
    for name, path in teacher_paths.items():
        source = feature_sources[path]
        if isinstance(source, MergedTeacherFeatureCacheReader):
            merged_groups.setdefault(path, []).append(name)
        else:
            result[name] = source.read_features_at(rows)
    for path, names in merged_groups.items():
        source = feature_sources[path]
        result.update(source.read_features_many_at(rows, names))
    return result


class _TeacherFeatureStore:
    def __init__(self, package_paths_by_teacher: dict[str, list[Path]]) -> None:
        unique_paths = sorted({path for paths in package_paths_by_teacher.values() for path in paths})
        self.sources = {path: _open_feature_source(path) for path in unique_paths}
        self.tile_path_by_teacher: dict[str, dict[str, Path]] = {}
        for teacher_name, package_paths in package_paths_by_teacher.items():
            teacher_name = str(teacher_name)
            tile_to_path: dict[str, Path] = {}
            for path in package_paths:
                source = self.sources[path]
                if not _source_has_teacher(source, teacher_name):
                    raise ValueError(f"feature package does not contain teacher={teacher_name}")
                tile_ids = source.record_table.column("tile_id").to_pylist()
                for tile_id in tile_ids:
                    tile_id = str(tile_id)
                    if tile_id in tile_to_path:
                        raise ValueError(f"duplicate tile_id across feature packages: {tile_id}")
                    tile_to_path[tile_id] = path
            self.tile_path_by_teacher[teacher_name] = tile_to_path

    def read_features(self, tile_id: str) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        merged_groups: dict[Path, list[str]] = {}
        for teacher_name, tile_to_path in self.tile_path_by_teacher.items():
            path = tile_to_path.get(tile_id)
            if path is None:
                raise FileNotFoundError(f"missing packaged teacher feature: {tile_id}")
            source = self.sources[path]
            if isinstance(source, MergedTeacherFeatureCacheReader):
                merged_groups.setdefault(path, []).append(teacher_name)
            else:
                result[teacher_name] = source.read_feature(tile_id)
        for path, teacher_names in merged_groups.items():
            source = self.sources[path]
            result.update(source.read_features(tile_id, teacher_names))
        return result

    def read_feature(self, tile_id: str, teacher_name: str):
        return self.read_features(tile_id)[str(teacher_name)]

    def close(self) -> None:
        for reader in self.sources.values():
            reader.close()

    def __del__(self) -> None:
        self.close()


def _strip_required_suffix(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"package name does not match expected suffix: path={path} suffix={suffix}")
    return name[: -len(suffix)]


def _feature_package_matches_tile_stem(feature_path: Path, tile_stem: str) -> bool:
    if feature_path.name.endswith(MERGED_FEATURE_SUFFIX):
        feature_stem = _strip_required_suffix(feature_path, MERGED_FEATURE_SUFFIX)
        return feature_stem == tile_stem or feature_stem.startswith(f"{tile_stem}.")
    feature_stem = _strip_required_suffix(feature_path, ".features.iac")
    return feature_stem == tile_stem or feature_stem.startswith(f"{tile_stem}.")


def validate_teacher_feature_package_pairs(
    image_tile_package_paths: list[str | Path],
    teacher_cache_package_paths: dict[str, list[str | Path]],
    expected_dims: dict[str, int] | None = None,
) -> list[int]:
    """Fast package-pair validation using names and IAC headers only."""
    tile_paths = [Path(path) for path in image_tile_package_paths]
    if not tile_paths:
        raise ValueError("at least one tile package is required")
    teacher_paths_by_name = {
        name: [Path(path) for path in paths] for name, paths in teacher_cache_package_paths.items()
    }
    for name, paths in teacher_paths_by_name.items():
        if len(paths) != len(tile_paths):
            raise ValueError(
                f"dynamic package sampling requires one feature package per tile package: "
                f"teacher={name} features={len(paths)} tiles={len(tile_paths)}"
            )

    counts: list[int] = []
    for package_idx, tile_path in enumerate(tile_paths):
        tile_header = read_header(tile_path)
        if tile_header.get("payload_type") != "image_tiles":
            raise ValueError(f"not an image tile package: {tile_path}")
        tile_stem = _strip_required_suffix(tile_path, ".tiles.iac")
        count = int(tile_header["num_records"])
        if count <= 0:
            raise ValueError(f"empty tile package: {tile_path}")
        counts.append(count)
        tile_record_ids: list[str] | None = None
        feature_ids_by_path: dict[Path, list[str]] = {}
        for teacher_name, feature_paths in teacher_paths_by_name.items():
            feature_path = feature_paths[package_idx]
            feature_header = read_header(feature_path)
            payload_type = feature_header.get("payload_type")
            if payload_type not in {"teacher_features", MERGED_FEATURE_PAYLOAD_TYPE}:
                raise ValueError(f"not a teacher feature package: teacher={teacher_name} path={feature_path}")
            if not _feature_package_matches_tile_stem(feature_path, tile_stem):
                raise ValueError(
                    f"feature/tile package stem mismatch: teacher={teacher_name} "
                    f"tile={tile_path.name} feature={feature_path.name}"
                )
            feature_count = int(feature_header["num_records"])
            if feature_count != count:
                raise ValueError(
                    f"feature/tile record count mismatch for teacher={teacher_name} package={tile_path}: "
                    f"features={feature_count} tiles={count}"
                )
            if payload_type == MERGED_FEATURE_PAYLOAD_TYPE:
                teachers = {str(name) for name in feature_header.get("teachers", [])}
                if teacher_name not in teachers:
                    raise ValueError(f"merged feature package missing teacher={teacher_name}: path={feature_path}")
            if tile_record_ids is None:
                _, _, tile_records = read_tables(tile_path)
                tile_record_ids = [
                    str(value)
                    for value in tile_records.column("tile_id").to_pylist()
                ]
            feature_ids = feature_ids_by_path.get(feature_path)
            if feature_ids is None:
                _, _, feature_records = read_tables(feature_path)
                feature_ids = [
                    str(value)
                    for value in feature_records.column("tile_id").to_pylist()
                ]
                feature_ids_by_path[feature_path] = feature_ids
            if feature_ids != tile_record_ids:
                raise ValueError(
                    "feature/tile tile_id order mismatch: "
                    f"teacher={teacher_name} path={feature_path}"
                )
            for key in ("tile_width", "tile_height", "stride_x", "stride_y"):
                if int(feature_header[key]) != int(tile_header[key]):
                    raise ValueError(
                        f"feature/tile header mismatch: teacher={teacher_name} key={key} "
                        f"tile={tile_header[key]} feature={feature_header[key]} package={tile_path}"
                    )
            if expected_dims is not None:
                expected_dim = expected_dims.get(teacher_name)
                if expected_dim is None:
                    raise ValueError(f"missing expected teacher dim for {teacher_name}")
                if payload_type == MERGED_FEATURE_PAYLOAD_TYPE:
                    feature_dim = int(feature_header["teacher_dims"][teacher_name])
                else:
                    feature_dim = int(feature_header["feature_dim"])
                if feature_dim != int(expected_dim):
                    raise ValueError(
                        f"feature dimension mismatch: teacher={teacher_name} expected={expected_dim} "
                        f"got={feature_dim} path={feature_path}"
                    )
    return counts


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
        prototype_labels: dict[str, PrototypeLabel] | None = None,
        spatial_targets: dict[str, SpatialRoiTarget] | None = None,
        spatial_component_count: int = 0,
        spatial_grid_size: tuple[int, int] = (0, 0),
    ) -> None:
        self.records = records
        if teacher_cache_dir is not None:
            raise ValueError("loose teacher feature directories are not supported; use a .features.iac package")
        self.package_reader = TilePackageReader(tile_package_path) if tile_package_path else None
        self.active_package_path: Path | None = None
        self.active_package_reader: TilePackageReader | None = None
        self.teacher_package_paths = resolve_teacher_feature_packages(
            teacher_cache_package_path=teacher_cache_package_path,
            teacher_cache_package_paths=teacher_cache_package_paths,
        )
        self.feature_store = _TeacherFeatureStore(self.teacher_package_paths)
        self.transform = _build_image_transform(
            image_size,
            mean,
            std,
            resize=True,
        )
        self.prototype_labels = prototype_labels or {}
        self.spatial_targets = spatial_targets or {}
        self.spatial_component_count = int(spatial_component_count)
        self.spatial_grid_size = tuple(int(value) for value in spatial_grid_size)
        self._thread_local = threading.local()

    def _packaged_reader(self, package_path: Path) -> TilePackageReader:
        if not hasattr(self._thread_local, "readers"):
            self._thread_local.readers = {}
        reader = self._thread_local.readers.get(package_path)
        if reader is None:
            reader = TilePackageReader(package_path)
            self._thread_local.readers[package_path] = reader
        return reader

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        item = self.records[index]
        record = _unwrap_record(item)
        package_path = _record_package_path(item)
        if package_path is not None:
            image = self._packaged_reader(package_path).read_image(record.tile_id)
            image_tensor = self.transform(image.convert("RGB"))
        elif self.package_reader is not None:
            image = self.package_reader.read_image(record.tile_id)
            image_tensor = self.transform(image.convert("RGB"))
        else:
            with Image.open(record.tile_path) as image:
                image_tensor = self.transform(image.convert("RGB"))
        teacher_features = {}
        for name, feature in self.feature_store.read_features(record.tile_id).items():
            teacher_feature = torch.from_numpy(feature).float()
            if teacher_feature.ndim != 1:
                raise ValueError(f"teacher feature must be 1D: teacher={name} tile_id={record.tile_id}")
            teacher_features[name] = teacher_feature
        return {
            "tile_id": record.tile_id,
            "image": image_tensor,
            "teacher_features": teacher_features,
            **_prototype_payload(record.tile_id, self.prototype_labels),
            **spatial_roi_payload(
                record.tile_id,
                self.spatial_targets,
                component_count=self.spatial_component_count,
                grid_size=self.spatial_grid_size,
            ),
        }

    def close(self) -> None:
        if self.package_reader is not None:
            self.package_reader.close()
            self.package_reader = None
        if self.active_package_reader is not None:
            self.active_package_reader.close()
            self.active_package_reader = None
            self.active_package_path = None
        if getattr(self, "feature_store", None) is not None:
            self.feature_store.close()
        if getattr(self, "_thread_local", None) is not None:
            readers = getattr(self._thread_local, "readers", None)
            if readers:
                for reader in readers.values():
                    reader.close()
                self._thread_local.readers = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["package_reader"] = None
        state["active_package_path"] = None
        state["active_package_reader"] = None
        state["feature_store"] = _TeacherFeatureStore(self.teacher_package_paths)
        state["_thread_local"] = None
        return state

    def __del__(self) -> None:
        self.close()


class PackageSampledDistillationDataset(Dataset):
    def __init__(
        self,
        image_tile_package_paths: list[str | Path],
        teacher_cache_package_paths: dict[str, list[str | Path]],
        image_size: int | tuple[int, int],
        max_records: int = 0,
        seed: int = 13,
        mean: list[float] | tuple[float, ...] | None = None,
        std: list[float] | tuple[float, ...] | None = None,
        expected_dims: dict[str, int] | None = None,
        prototype_labels: dict[str, PrototypeLabel] | None = None,
        tensor_collate: bool = False,
        spatial_targets: dict[str, SpatialRoiTarget] | None = None,
        spatial_component_count: int = 0,
        spatial_grid_size: tuple[int, int] = (0, 0),
    ) -> None:
        self.tile_paths = [Path(path) for path in image_tile_package_paths]
        self.teacher_package_paths = {
            name: [Path(path) for path in paths] for name, paths in teacher_cache_package_paths.items()
        }
        self.active_package_idx: int | None = None
        self.active_tile_reader: TilePackageReader | None = None
        self.active_feature_readers: dict[Path, object] = {}
        self.active_teacher_feature_paths: dict[str, Path] = {}
        counts = validate_teacher_feature_package_pairs(
            self.tile_paths,
            self.teacher_package_paths,
            expected_dims=expected_dims,
        )
        self.package_counts = counts
        self.sequential_iac_rows = all(int(read_header(path).get("row_order_seed", 0) or 0) > 0 for path in self.tile_paths)
        self.cumulative_counts: list[int] = []
        running_total = 0
        for count in counts:
            running_total += count
            self.cumulative_counts.append(running_total)
        self.total_records = running_total
        self.sample_count = self.total_records if max_records <= 0 else min(int(max_records), self.total_records)
        if 0 < self.sample_count < len(self.tile_paths):
            raise ValueError(
                f"max_records must be 0 or at least the selected package count so every package participates: "
                f"max_records={self.sample_count} packages={len(self.tile_paths)}"
            )
        rng = np.random.default_rng(seed)
        self.package_order, self.block_offsets, self.package_sample_rows = self._build_package_ordered_plan(
            self.sample_count,
            rng,
        )
        self.transform = _build_image_transform(
            image_size,
            mean,
            std,
            resize=False,
        )
        self.prototype_labels = prototype_labels or {}
        self.spatial_targets = spatial_targets or {}
        self.spatial_component_count = int(spatial_component_count)
        self.spatial_grid_size = tuple(int(value) for value in spatial_grid_size)
        self.tensor_collate = bool(tensor_collate)
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1) if mean is not None else None
        self.std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1) if std is not None else None
        self._image_hw = (int(image_size), int(image_size)) if isinstance(image_size, int) else (int(image_size[0]), int(image_size[1]))
        self._teacher_dims: dict[str, int] = dict(expected_dims) if expected_dims else {}
        # Per-thread reader LRU cap (entries = tile/feature readers). Keeps total
        # open file handles bounded across many worker threads. Each cached entry
        # is one open package file; with ~1.8k packages an unbounded cache would
        # exhaust the process FD limit. 32 entries/thread balances handle count
        # against reopen churn (a worker streams a few packages at a time).
        self._reader_cache_cap = 32
        self._thread_local = threading.local()

    def _package_sample_counts(self, sample_count: int, rng: np.random.Generator) -> list[int]:
        if sample_count >= self.total_records:
            return list(self.package_counts)
        expected = np.asarray(self.package_counts, dtype=np.float64) * (sample_count / self.total_records)
        counts = np.floor(expected).astype(np.int64)
        if sample_count >= len(counts):
            counts = np.maximum(counts, 1)
        counts = np.minimum(counts, np.asarray(self.package_counts, dtype=np.int64))
        overflow = int(counts.sum() - sample_count)
        if overflow > 0:
            candidates = np.flatnonzero(counts > 0)
            removable = rng.choice(candidates, size=overflow, replace=False)
            counts[removable] -= 1
        remainder = int(sample_count - counts.sum())
        if remainder > 0:
            fractions = expected - np.floor(expected)
            capacity = np.asarray(self.package_counts, dtype=np.int64) - counts
            candidates = np.flatnonzero(capacity > 0)
            if len(candidates) < remainder:
                raise ValueError(f"could not distribute sample budget: remainder={remainder}")
            weights = fractions[candidates]
            if float(weights.sum()) <= 0:
                chosen = rng.choice(candidates, size=remainder, replace=False)
            else:
                chosen = rng.choice(candidates, size=remainder, replace=False, p=weights / weights.sum())
            counts[chosen] += 1
        return [int(value) for value in counts]

    def _ordered_rows_for_package(
        self,
        package_count: int,
        take: int,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        if take <= 0:
            return np.empty((0,), dtype=np.int64)
        if take >= package_count:
            return None
        if self.sequential_iac_rows:
            return np.arange(take, dtype=np.int64)
        stride = package_count / take
        offset = float(rng.random()) * stride
        rows = np.floor(offset + np.arange(take, dtype=np.float64) * stride).astype(np.int64)
        rows = np.minimum(rows, package_count - 1)
        rows = np.unique(rows)
        while len(rows) < take:
            missing = take - len(rows)
            extra = rng.choice(package_count, size=missing, replace=False)
            rows = np.unique(np.concatenate([rows, extra.astype(np.int64, copy=False)]))
        rows.sort()
        return rows[:take]

    def _build_package_ordered_plan(
        self,
        sample_count: int,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[int], list[np.ndarray | None]]:
        package_order: list[int] = []
        package_rows: list[np.ndarray | None] = [np.empty((0,), dtype=np.int64) for _ in self.package_counts]
        for package_idx, take in enumerate(self._package_sample_counts(sample_count, rng)):
            rows = self._ordered_rows_for_package(
                self.package_counts[package_idx],
                take,
                rng,
            )
            if rows is None or len(rows):
                package_order.append(package_idx)
                package_rows[package_idx] = rows
        if not self.sequential_iac_rows:
            rng.shuffle(package_order)
        block_offsets: list[int] = []
        running_total = 0
        for package_idx in package_order:
            block_offsets.append(running_total)
            rows = package_rows[package_idx]
            running_total += self.package_counts[package_idx] if rows is None else len(rows)
        if running_total != sample_count:
            raise ValueError(f"invalid package sampling plan: expected={sample_count} got={running_total}")
        return package_order, block_offsets, package_rows

    def iter_package_row_chunks(
        self,
        chunk_size: int,
        seed: int,
    ) -> Iterator[tuple[int, np.ndarray]]:
        chunk_size = max(1, int(chunk_size))
        per_package_chunks: dict[int, list[np.ndarray]] = {}
        max_chunks = 0
        for package_idx in self.package_order:
            rows = self.package_sample_rows[package_idx]
            package_count = self.package_counts[package_idx]
            if rows is None:
                chunks = [
                    np.arange(start, min(start + chunk_size, package_count), dtype=np.int64)
                    for start in range(0, package_count, chunk_size)
                ]
            else:
                chunks = [
                    np.asarray(rows[start : start + chunk_size], dtype=np.int64)
                    for start in range(0, len(rows), chunk_size)
                ]
            if chunks:
                per_package_chunks[package_idx] = chunks
                max_chunks = max(max_chunks, len(chunks))
        rng = np.random.default_rng(seed)
        for chunk_idx in range(max_chunks):
            package_indices = [
                package_idx
                for package_idx, chunks in per_package_chunks.items()
                if chunk_idx < len(chunks)
            ]
            rng.shuffle(package_indices)
            for package_idx in package_indices:
                yield package_idx, per_package_chunks[package_idx][chunk_idx]

    def iter_global_index_chunks(self, chunk_size: int, seed: int) -> "list[np.ndarray]":
        """Same seed-ordered chunk plan as iter_package_row_chunks, but each
        chunk is expressed as GLOBAL dataset indices (the [0, sample_count)
        space __getitem__ expects). Lets a standard multiprocessing DataLoader
        reproduce the exact sampling order via a batch_sampler — each worker is
        a separate PROCESS with its own GIL, so decode no longer contends with
        the main-thread GPU inference for one process-wide GIL.

        A package occupies a contiguous global-index block at block_offsets[b]
        (b = position of that package in package_order), in the SAME order its
        sampled rows are laid out in package_sample_rows. So the k-th sampled
        row of a package maps to global index block_offsets[b] + k.
        """
        chunk_size = max(1, int(chunk_size))
        pkg_block_start: dict[int, int] = {}
        for b, package_idx in enumerate(self.package_order):
            pkg_block_start[package_idx] = self.block_offsets[b]
        per_package_chunks: dict[int, list[np.ndarray]] = {}
        max_chunks = 0
        for package_idx in self.package_order:
            rows = self.package_sample_rows[package_idx]
            n = self.package_counts[package_idx] if rows is None else len(rows)
            base = pkg_block_start[package_idx]
            chunks = [
                np.arange(base + start, base + min(start + chunk_size, n), dtype=np.int64)
                for start in range(0, n, chunk_size)
            ]
            if chunks:
                per_package_chunks[package_idx] = chunks
                max_chunks = max(max_chunks, len(chunks))
        rng = np.random.default_rng(seed)
        out: list[np.ndarray] = []
        for chunk_idx in range(max_chunks):
            package_indices = [pi for pi, ch in per_package_chunks.items() if chunk_idx < len(ch)]
            rng.shuffle(package_indices)
            for package_idx in package_indices:
                out.append(per_package_chunks[package_idx][chunk_idx])
        return out


    def _thread_reader_cache(self):
        if not hasattr(self._thread_local, "readers"):
            from collections import OrderedDict
            self._thread_local.readers = OrderedDict()
        return self._thread_local.readers

    def _cached_thread_reader(self, kind: str, path: Path, opener):
        cache = self._thread_reader_cache()
        key = (kind, path)
        reader = cache.get(key)
        if reader is not None:
            cache.move_to_end(key)
            return reader
        reader = opener(path)
        cache[key] = reader
        # Bound the per-thread open-file handles. With many worker threads and
        # ~1.8k packages an unbounded cache exhausts the process FD limit
        # (OSError: Too many open files). Evict + close the least-recently-used
        # readers once the per-thread cache exceeds the cap.
        cap = self._reader_cache_cap
        while len(cache) > cap:
            _old_key, old_reader = cache.popitem(last=False)
            close = getattr(old_reader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        return reader

    def read_package_rows(self, package_idx: int, rows: np.ndarray) -> list[dict]:
        tile_reader = self._cached_thread_reader("tile", self.tile_paths[package_idx], TilePackageReader)
        teacher_paths = {name: paths[package_idx] for name, paths in self.teacher_package_paths.items()}
        feature_sources = {
            path: self._cached_thread_reader("feature", path, _open_feature_source)
            for path in set(teacher_paths.values())
        }
        sorted_rows = sorted(int(row) for row in rows)
        images = tile_reader.read_arrays_at(sorted_rows)
        feature_batches = _read_teacher_features_many_at(feature_sources, teacher_paths, sorted_rows)
        samples = []
        for index, row in enumerate(sorted_rows):
            tile_id = tile_reader.tile_id_at(row)
            samples.append(
                {
                    "_package_idx": package_idx,
                    "tile_id": tile_id,
                    "image": images[index],
                    "teacher_features": {name: values[index] for name, values in feature_batches.items()},
                    **_prototype_payload(tile_id, self.prototype_labels),
                    **spatial_roi_payload(
                        tile_id,
                        self.spatial_targets,
                        component_count=self.spatial_component_count,
                        grid_size=self.spatial_grid_size,
                    ),
                }
            )
        return samples

    def batch_buffer_spec(self) -> dict:
        """Return shapes needed to pre-allocate one batch's pinned tensors.

        teacher dims come from ``expected_dims`` when available, otherwise they
        are probed once by reading a single feature vector per teacher.
        """
        teacher_dims = dict(self._teacher_dims)
        missing = [name for name in self.teacher_package_paths if name not in teacher_dims]
        if missing:
            teacher_paths = {name: paths[0] for name, paths in self.teacher_package_paths.items()}
            sources = {path: _open_feature_source(path) for path in set(teacher_paths.values())}
            feats = _read_teacher_features_at(sources, teacher_paths, 0)
            for name in missing:
                teacher_dims[name] = int(np.asarray(feats[name]).reshape(-1).shape[0])
            for src in sources.values():
                src.close()
            self._teacher_dims = teacher_dims
        return {
            "image_hw": self._image_hw,
            "teacher_dims": teacher_dims,
            "spatial_shape": (self.spatial_component_count, *self.spatial_grid_size),
        }

    def scatter_package_rows(
        self,
        package_idx: int,
        rows: np.ndarray,
        positions: list,
        buffers: list,
    ) -> None:
        """Decode rows and write them into pre-allocated pinned batch tensors at
        their final positions. ``positions[k]`` is the (buffer, slot) for
        ``rows[k]``; all positions in one call share the same ring buffer.

        GIL strategy: only the JXL decode + feature reads run per-tile (decode
        releases the GIL, so those parallelize across workers). The tensor
        writes (copy_/scalar assignment) HOLD the GIL and do NOT parallelize —
        a per-tile copy_ loop makes 8 GIL-bound ATen calls per tile, and the
        GIL hand-off thrash across workers made copy_feats scale at 0.5x (12
        threads SLOWER than 1). So we accumulate decoded rows into local numpy
        arrays and flush them with ONE index_copy_ per tensor — turning
        ~8*chunk GIL-held ATen ops into ~6 per chunk, collapsing the thrash.
        """
        with _probe.section("scatter_open_readers"):
            tile_reader = self._cached_thread_reader("tile", self.tile_paths[package_idx], TilePackageReader)
            teacher_paths = {name: paths[package_idx] for name, paths in self.teacher_package_paths.items()}
            feature_sources = {
                path: self._cached_thread_reader("feature", path, _open_feature_source)
                for path in set(teacher_paths.values())
            }
        n = len(rows)
        if n == 0:
            return
        order = sorted(range(n), key=lambda k: int(rows[k]))
        buf = buffers[positions[0].buffer_idx]
        h, w = int(buf.images.shape[1]), int(buf.images.shape[2])
        teacher_names = list(buf.teacher_features.keys())

        sorted_rows = [int(rows[k]) for k in order]
        with _probe.section("decode_image"):
            tile_ids = [tile_reader.tile_id_at(row) for row in sorted_rows]
            images = tile_reader.read_arrays_at(sorted_rows)
        with _probe.section("read_feats"):
            feature_batches = _read_teacher_features_many_at(feature_sources, teacher_paths, sorted_rows)

        # Local staging arrays (no GIL contention — plain numpy in this thread).
        img_stage = np.empty((n, h, w, 3), dtype=np.uint8)
        feat_stage = {name: np.empty((n, buf.teacher_features[name].shape[1]), dtype=np.float32) for name in teacher_names}
        pos_stage = np.empty(n, dtype=np.int64)
        mask_stage = np.zeros(n, dtype=bool)
        l1_stage = np.full(n, -1, dtype=np.int64)
        tid_stage: list = [None] * n

        for i, k in enumerate(order):
            pos_stage[i] = positions[k].pos
            tile_id = tile_ids[i]
            img_stage[i] = images[i]
            for name, features in feature_batches.items():
                feat_stage[name][i] = features[i]
            with _probe.section("proto"):
                proto = _prototype_payload(tile_id, self.prototype_labels)
                tid_stage[i] = tile_id
                mask_stage[i] = bool(proto["prototype_mask"])
                l1_stage[i] = int(proto["prototype_level1"])
                buf.spatial_targets[int(pos_stage[i])] = self.spatial_targets.get(
                    tile_id
                )

        # Single batched, GIL-held flush per tensor (was 8*n per-tile ops).
        with _probe.section("copy_flush"):
            pos_t = torch.from_numpy(pos_stage)
            buf.images.index_copy_(0, pos_t, torch.from_numpy(img_stage))
            for name in teacher_names:
                buf.teacher_features[name].index_copy_(0, pos_t, torch.from_numpy(feat_stage[name]))
            buf.prototype_mask.index_copy_(0, pos_t, torch.from_numpy(mask_stage))
            buf.prototype_level1.index_copy_(0, pos_t, torch.from_numpy(l1_stage))
            for i in range(n):
                buf.tile_id[int(pos_stage[i])] = tid_stage[i]
        _probe.flush_thread()



    def _tile_reader(self, package_idx: int) -> TilePackageReader:
        self._activate_package(package_idx)
        assert self.active_tile_reader is not None
        return self.active_tile_reader

    def _activate_package(self, package_idx: int) -> None:
        if self.active_package_idx == package_idx:
            return
        self._close_active_readers()
        self.active_package_idx = package_idx
        self.active_tile_reader = TilePackageReader(self.tile_paths[package_idx])
        self.active_teacher_feature_paths = {
            name: paths[package_idx] for name, paths in self.teacher_package_paths.items()
        }
        self.active_feature_readers = {
            path: _open_feature_source(path)
            for path in set(self.active_teacher_feature_paths.values())
        }

    def __len__(self) -> int:
        return self.sample_count

    def _locate_sample(self, index: int) -> tuple[int, int]:
        block_idx = bisect_right(self.block_offsets, index) - 1
        package_idx = self.package_order[block_idx]
        offset = index - self.block_offsets[block_idx]
        rows = self.package_sample_rows[package_idx]
        row = offset if rows is None else int(rows[offset])
        return package_idx, row

    def _arrays_to_tensor(self, arrays: list[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
        if self.mean_tensor is not None and self.std_tensor is not None:
            tensor.sub_(self.mean_tensor).div_(self.std_tensor)
        return tensor

    def __getitem__(self, index: int) -> dict:
        package_idx, row = self._locate_sample(index)
        image_reader = self._tile_reader(package_idx)
        tile_id = image_reader.tile_id_at(row)
        image = image_reader.read_array_at(row)
        teacher_features = _read_teacher_features_at(
            self.active_feature_readers,
            self.active_teacher_feature_paths,
            row,
        )
        return {
            "tile_id": tile_id,
            "image": image,
            "teacher_features": teacher_features,
            **_prototype_payload(tile_id, self.prototype_labels),
            **spatial_roi_payload(
                tile_id,
                self.spatial_targets,
                component_count=self.spatial_component_count,
                grid_size=self.spatial_grid_size,
            ),
        }

    def __getitems__(self, indices: list[int]) -> list[dict]:
        results: list[dict | None] = [None for _ in indices]
        by_package: dict[int, list[tuple[int, int]]] = {}
        for out_idx, index in enumerate(indices):
            package_idx, row = self._locate_sample(index)
            by_package.setdefault(package_idx, []).append((out_idx, row))
        for package_idx, items in by_package.items():
            tile_reader = self._cached_thread_reader("tile", self.tile_paths[package_idx], TilePackageReader)
            teacher_paths = {name: paths[package_idx] for name, paths in self.teacher_package_paths.items()}
            feature_sources = {
                path: self._cached_thread_reader("feature", path, _open_feature_source)
                for path in set(teacher_paths.values())
            }
            ordered_items = sorted(items, key=lambda item: item[1])
            rows = [row for _, row in ordered_items]
            images = tile_reader.read_arrays_at(rows)
            feature_batches = _read_teacher_features_many_at(feature_sources, teacher_paths, rows)
            for index, (out_idx, row) in enumerate(ordered_items):
                tile_id = tile_reader.tile_id_at(row)
                results[out_idx] = {
                    "tile_id": tile_id,
                    "image": images[index],
                    "teacher_features": {name: values[index] for name, values in feature_batches.items()},
                    **_prototype_payload(tile_id, self.prototype_labels),
                    **spatial_roi_payload(
                        tile_id,
                        self.spatial_targets,
                        component_count=self.spatial_component_count,
                        grid_size=self.spatial_grid_size,
                    ),
                }
        return [item for item in results if item is not None]

    def collate(self, batch: list[dict]) -> dict:
        teacher_names = list(batch[0]["teacher_features"].keys())
        if self.tensor_collate:
            images = torch.from_numpy(np.stack([item["image"] for item in batch], axis=0)).permute(0, 3, 1, 2).contiguous()
        else:
            images = torch.stack([self.transform(Image.fromarray(item["image"]).convert("RGB")) for item in batch])
        return {
            "tile_id": [item["tile_id"] for item in batch],
            "images": images,
            "images_uint8": self.tensor_collate,
            "teacher_features": {
                name: torch.from_numpy(
                    np.stack([item["teacher_features"][name] for item in batch], axis=0)
                ).float()
                for name in teacher_names
            },
            "prototype_mask": torch.tensor([bool(item["prototype_mask"]) for item in batch], dtype=torch.bool),
            "prototype_level1": torch.tensor([int(item["prototype_level1"]) for item in batch], dtype=torch.long),
            "l2_point_centers": torch.stack([item["l2_point_centers"] for item in batch]),
            "l2_brush_bag_ids": torch.stack([item["l2_brush_bag_ids"] for item in batch]),
            "l2_area_positive": torch.stack([item["l2_area_positive"] for item in batch]),
            "l2_explicit_negative": torch.stack([item["l2_explicit_negative"] for item in batch]),
            "l2_implicit_negative": torch.stack([item["l2_implicit_negative"] for item in batch]),
            "l2_spatial_supervised": torch.stack([item["l2_spatial_supervised"] for item in batch]),
        }

    def close(self) -> None:
        self._close_active_readers()

    def _close_active_readers(self) -> None:
        if getattr(self, "active_tile_reader", None) is not None:
            self.active_tile_reader.close()
        for reader in getattr(self, "active_feature_readers", {}).values():
            reader.close()
        self.active_package_idx = None
        self.active_tile_reader = None
        self.active_feature_readers = {}
        self.active_teacher_feature_paths = {}
        if getattr(self, "_thread_local", None) is not None:
            readers = getattr(self._thread_local, "readers", None)
            if readers:
                for reader in readers.values():
                    reader.close()
                self._thread_local.readers = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["active_package_idx"] = None
        state["active_tile_reader"] = None
        state["active_feature_readers"] = {}
        state["active_teacher_feature_paths"] = {}
        state["_thread_local"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._thread_local = threading.local()

    def __del__(self) -> None:
        self.close()


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
        reader = _TeacherFeatureStore({name: package_paths})
        try:
            if isinstance(expected_dim, dict):
                dim = expected_dim.get(name)
            else:
                dim = expected_dim
            wrong_shape = []
            for item in records:
                tile_id = _record_tile_id(item)
                feature = reader.read_feature(tile_id, name)
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
        "prototype_mask": torch.tensor([bool(item.get("prototype_mask", False)) for item in batch], dtype=torch.bool),
        "prototype_level1": torch.tensor([int(item.get("prototype_level1", -1)) for item in batch], dtype=torch.long),
        "l2_point_centers": torch.stack([item.get("l2_point_centers", torch.zeros((0, 0, 0))) for item in batch]),
        "l2_brush_bag_ids": torch.stack([item.get("l2_brush_bag_ids", torch.zeros((0, 0, 0), dtype=torch.long)) for item in batch]),
        "l2_area_positive": torch.stack([item.get("l2_area_positive", torch.zeros((0, 0, 0), dtype=torch.bool)) for item in batch]),
        "l2_explicit_negative": torch.stack([item.get("l2_explicit_negative", torch.zeros((0, 0, 0), dtype=torch.bool)) for item in batch]),
        "l2_implicit_negative": torch.stack([item.get("l2_implicit_negative", torch.zeros((0, 0, 0), dtype=torch.bool)) for item in batch]),
        "l2_spatial_supervised": torch.stack([item.get("l2_spatial_supervised", torch.zeros(0, dtype=torch.bool)) for item in batch]),
    }
