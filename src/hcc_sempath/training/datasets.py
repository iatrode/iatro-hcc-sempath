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
from torch.utils.data import Dataset
from torchvision import transforms

from ..io.feature_cache import FeatureCacheReader
from ..io.iatrocache import read_header, read_tables
from ..io.manifests import TileRecord
from ..io.tile_package import TilePackageReader
from .feature_pack_merge import (
    MERGED_FEATURE_PAYLOAD_TYPE,
    MERGED_FEATURE_SUFFIX,
    MergedTeacherFeatureCacheReader,
)
from .prototype_labels import PrototypeLabel


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


_EMPTY_PROTOTYPE_LEVEL2 = torch.zeros(0, dtype=torch.float32)
_cached_zeros: dict[tuple[int, ...], torch.Tensor] = {}


def _get_zero_tensor(like_tensor: torch.Tensor) -> torch.Tensor:
    shape = tuple(like_tensor.shape)
    if shape not in _cached_zeros:
        _cached_zeros[shape] = torch.zeros(shape, dtype=like_tensor.dtype, device=like_tensor.device)
    return _cached_zeros[shape]


def _prototype_payload(tile_id: str, prototype_labels: dict[str, PrototypeLabel] | None) -> dict:
    if not prototype_labels:
        return {
            "prototype_mask": False,
            "prototype_level1": -1,
            "prototype_level2": _EMPTY_PROTOTYPE_LEVEL2,
        }
    label = prototype_labels.get(tile_id)
    if label is None:
        first = next(iter(prototype_labels.values()))
        return {
            "prototype_mask": False,
            "prototype_level1": -1,
            "prototype_level2": _get_zero_tensor(first.level2),
        }
    return {
        "prototype_mask": True,
        "prototype_level1": label.level1,
        "prototype_level2": label.level2,
    }


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
                _, _, tile_records = read_tables(tile_path)
                _, _, feature_records = read_tables(feature_path)
                tile_ids = [str(value) for value in tile_records.column("tile_id").to_pylist()]
                feature_ids = [str(value) for value in feature_records.column("tile_id").to_pylist()]
                if feature_ids != tile_ids:
                    raise ValueError(f"merged feature/tile tile_id order mismatch: teacher={teacher_name} path={feature_path}")
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
        self.tensor_collate = bool(tensor_collate)
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1) if mean is not None else None
        self.std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1) if std is not None else None
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

    def _thread_reader_cache(self) -> dict[tuple[str, Path], object]:
        if not hasattr(self._thread_local, "readers"):
            self._thread_local.readers = {}
        return self._thread_local.readers

    def _cached_thread_reader(self, kind: str, path: Path, opener):
        cache = self._thread_reader_cache()
        key = (kind, path)
        reader = cache.get(key)
        if reader is None:
            reader = opener(path)
            cache[key] = reader
        return reader

    def read_package_rows(self, package_idx: int, rows: np.ndarray) -> list[dict]:
        tile_reader = self._cached_thread_reader("tile", self.tile_paths[package_idx], TilePackageReader)
        teacher_paths = {name: paths[package_idx] for name, paths in self.teacher_package_paths.items()}
        feature_sources = {
            path: self._cached_thread_reader("feature", path, _open_feature_source)
            for path in set(teacher_paths.values())
        }
        samples = []
        for row in sorted(int(row) for row in rows):
            tile_id = tile_reader.tile_id_at(row)
            image = tile_reader.read_array_at(row)
            samples.append(
                {
                    "_package_idx": package_idx,
                    "tile_id": tile_id,
                    "image": image,
                    "teacher_features": _read_teacher_features_at(feature_sources, teacher_paths, row),
                    **_prototype_payload(tile_id, self.prototype_labels),
                }
            )
        return samples

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
            for out_idx, row in sorted(items, key=lambda item: item[1]):
                tile_id = tile_reader.tile_id_at(row)
                image = tile_reader.read_array_at(row)
                results[out_idx] = {
                    "tile_id": tile_id,
                    "image": image,
                    "teacher_features": _read_teacher_features_at(feature_sources, teacher_paths, row),
                    **_prototype_payload(tile_id, self.prototype_labels),
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
            "prototype_level2": torch.stack([item["prototype_level2"] for item in batch]),
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
        "prototype_level2": torch.stack([item.get("prototype_level2", torch.zeros(0, dtype=torch.float32)) for item in batch]),
    }
