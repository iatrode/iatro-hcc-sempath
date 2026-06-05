from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow as pa

from ..io.iatrocache import PackReader, build_pack_data_segment_from_file, read_header, read_tables


MERGED_FEATURE_PAYLOAD_TYPE = "merged_teacher_features"
MERGED_FEATURE_SUFFIX = ".merged.features.iac"
TILE_SUFFIX = ".tiles.iac"


def is_merged_teacher_feature_package(path: str | Path) -> bool:
    try:
        return read_header(path).get("payload_type") == MERGED_FEATURE_PAYLOAD_TYPE
    except FileNotFoundError:
        return False


class MergedTeacherFeatureCacheReader:
    def __init__(self, package_path: str | Path) -> None:
        self._reader = PackReader(package_path)
        self._tile_index: dict[str, int] | None = None
        self._tile_ids: list[str] | None = None
        self._teacher_offsets: dict[str, int] | None = None
        self._teacher_record_bytes: dict[str, int] | None = None
        self._teacher_dtypes: dict[str, np.dtype] | None = None
        self._teacher_dims: dict[str, int] | None = None
        self._merged_record_bytes: int | None = None

    @property
    def header(self) -> dict:
        return self._reader.header

    @property
    def record_table(self):
        return self._reader.record_table

    def _ensure_index(self) -> None:
        if self._tile_index is not None:
            return
        header = self._reader.header
        if header.get("payload_type") != MERGED_FEATURE_PAYLOAD_TYPE:
            raise ValueError(f"not a merged teacher feature package: {header.get('payload_type')}")
        teachers = [str(name) for name in header["teachers"]]
        teacher_offsets = {str(k): int(v) for k, v in header["teacher_offsets"].items()}
        teacher_record_bytes = {str(k): int(v) for k, v in header["teacher_record_bytes"].items()}
        teacher_dtypes = {str(k): np.dtype(v) for k, v in header["teacher_dtypes"].items()}
        teacher_dims = {str(k): int(v) for k, v in header["teacher_dims"].items()}
        missing = set(teachers).difference(teacher_offsets, teacher_record_bytes, teacher_dtypes, teacher_dims)
        if missing:
            raise ValueError(f"merged feature header missing teacher metadata: {sorted(missing)}")
        for name in teachers:
            expected = teacher_dims[name] * teacher_dtypes[name].itemsize
            if teacher_record_bytes[name] != expected:
                raise ValueError(
                    f"invalid merged feature record bytes for {name}: "
                    f"expected={expected} got={teacher_record_bytes[name]}"
                )
        tile_ids = self._reader.record_table.column("tile_id").to_pylist()
        seen: set[str] = set()
        duplicates: list[str] = []
        for tile_id in tile_ids:
            if tile_id in seen:
                duplicates.append(str(tile_id))
            seen.add(str(tile_id))
        if duplicates:
            sample = ", ".join(duplicates[:3])
            raise ValueError(f"duplicate tile_id values in merged feature package: count={len(duplicates)} sample={sample}")
        self._tile_ids = [str(x) for x in tile_ids]
        self._tile_index = {self._tile_ids[i]: i for i in range(len(self._tile_ids))}
        self._teacher_offsets = teacher_offsets
        self._teacher_record_bytes = teacher_record_bytes
        self._teacher_dtypes = teacher_dtypes
        self._teacher_dims = teacher_dims
        self._merged_record_bytes = int(header["merged_record_bytes"])

    def has_teacher(self, name: str) -> bool:
        self._ensure_index()
        assert self._teacher_dims is not None
        return str(name) in self._teacher_dims

    def read_feature(self, tile_id: str, teacher_name: str) -> np.ndarray:
        self._ensure_index()
        assert self._tile_index is not None
        row = self._tile_index.get(tile_id)
        if row is None:
            raise FileNotFoundError(f"missing packaged teacher feature: {tile_id}")
        return self.read_feature_at(row, teacher_name)

    def read_features(self, tile_id: str, teacher_names: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
        self._ensure_index()
        assert self._tile_index is not None
        row = self._tile_index.get(tile_id)
        if row is None:
            raise FileNotFoundError(f"missing packaged teacher feature: {tile_id}")
        return self.read_features_at(row, teacher_names)

    def read_feature_at(self, row: int, teacher_name: str) -> np.ndarray:
        return self.read_features_at(row, [teacher_name])[str(teacher_name)]

    def read_features_at(self, row: int, teacher_names: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
        self._ensure_index()
        assert self._teacher_offsets is not None
        assert self._teacher_record_bytes is not None
        assert self._teacher_dtypes is not None
        assert self._teacher_dims is not None
        assert self._merged_record_bytes is not None
        if row < 0 or row >= int(self._reader.header["num_records"]):
            raise IndexError(f"feature row out of range: {row}")
        payload = self._reader.read_data_span(row * self._merged_record_bytes, self._merged_record_bytes)
        result: dict[str, np.ndarray] = {}
        for raw_name in teacher_names:
            name = str(raw_name)
            if name not in self._teacher_offsets:
                raise KeyError(f"merged feature package does not contain teacher={name}")
            start = self._teacher_offsets[name]
            end = start + self._teacher_record_bytes[name]
            feature = np.frombuffer(payload[start:end], dtype=self._teacher_dtypes[name])
            expected_shape = (self._teacher_dims[name],)
            if feature.shape != expected_shape:
                raise ValueError(f"invalid merged feature shape: teacher={name} row={row} shape={feature.shape}")
            result[name] = feature.astype(np.float32, copy=True)
        return result

    def tile_id_at(self, row: int) -> str:
        self._ensure_index()
        assert self._tile_ids is not None
        if row < 0 or row >= len(self._tile_ids):
            raise IndexError(f"feature row out of range: {row}")
        return self._tile_ids[row]

    @property
    def record_count(self) -> int:
        return int(self._reader.header["num_records"])

    def close(self) -> None:
        self._reader.close()
        self._tile_index = None
        self._tile_ids = None
        self._teacher_offsets = None
        self._teacher_record_bytes = None
        self._teacher_dtypes = None
        self._teacher_dims = None
        self._merged_record_bytes = None

    def __getstate__(self) -> dict:
        return {"package_path": self._reader.package_path}

    def __setstate__(self, state: dict) -> None:
        self._reader = PackReader(state["package_path"])
        self._tile_index = None
        self._tile_ids = None
        self._teacher_offsets = None
        self._teacher_record_bytes = None
        self._teacher_dtypes = None
        self._teacher_dims = None
        self._merged_record_bytes = None

    def __del__(self) -> None:
        self.close()


def maybe_prepare_merged_teacher_feature_packages(
    *,
    cfg: dict,
    split: str,
    tile_packages: list[str],
    teacher_package_paths: dict[str, list[str]],
    expected_dims: dict[str, int],
) -> dict[str, list[str]]:
    data_cfg = cfg.get("data", {})
    auto_merge = bool(data_cfg.get("auto_merge_teacher_features", False))
    prefer_existing = bool(data_cfg.get("prefer_merged_teacher_features", True))
    delete_sources = bool(data_cfg.get("auto_merge_delete_source_features", False))
    if len(teacher_package_paths) <= 1:
        return teacher_package_paths
    if not auto_merge and not prefer_existing:
        return teacher_package_paths

    teacher_names = list(teacher_package_paths)
    merged_by_teacher = {name: [] for name in teacher_names}
    worker_count = _prepare_worker_count(data_cfg, len(tile_packages))

    def prepare_one(package_idx: int) -> tuple[int, dict[str, str], str | None]:
        return _prepare_one_merged_teacher_feature_package(
            package_idx=package_idx,
            split=split,
            tile_path=Path(tile_packages[package_idx]),
            teacher_names=teacher_names,
            source_paths={name: Path(teacher_package_paths[name][package_idx]) for name in teacher_names},
            expected_dims=expected_dims,
            auto_merge=auto_merge,
            delete_sources=delete_sources,
            dtype=str(data_cfg.get("auto_merge_feature_dtype", "float32")),
        )

    if worker_count == 1:
        results = [prepare_one(package_idx) for package_idx in range(len(tile_packages))]
    else:
        print(f"merged_feature_pack_prepare split={split} workers={worker_count} packages={len(tile_packages)}")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(prepare_one, range(len(tile_packages))))

    for _, paths_by_teacher, message in sorted(results, key=lambda item: item[0]):
        if message:
            print(message)
        for name in teacher_names:
            merged_by_teacher[name].append(paths_by_teacher[name])
    return merged_by_teacher


def _prepare_worker_count(data_cfg: dict, package_count: int) -> int:
    workers = int(data_cfg.get("auto_iac_prepare_workers", 1) or 1)
    return max(1, min(int(package_count), workers))


def _prepare_one_merged_teacher_feature_package(
    *,
    package_idx: int,
    split: str,
    tile_path: Path,
    teacher_names: list[str],
    source_paths: dict[str, Path],
    expected_dims: dict[str, int],
    auto_merge: bool,
    delete_sources: bool,
    dtype: str,
) -> tuple[int, dict[str, str], str | None]:
    merged_path = _merged_output_path(tile_path, source_paths[teacher_names[0]])
    unique_source_paths = set(source_paths.values())
    if len(unique_source_paths) == 1 and is_merged_teacher_feature_package(next(iter(unique_source_paths))):
        merged_path = next(iter(unique_source_paths))
        _validate_merged_package(
            tile_path=tile_path,
            merged_path=merged_path,
            teacher_names=teacher_names,
            expected_dims=expected_dims,
        )
    elif merged_path.exists():
        _validate_merged_package(
            tile_path=tile_path,
            merged_path=merged_path,
            teacher_names=teacher_names,
            expected_dims=expected_dims,
        )
        if delete_sources:
            existing_sources = [path for path in set(source_paths.values()) if path.exists()]
            if len(existing_sources) == len(set(source_paths.values())):
                _validate_merged_against_sources(
                    source_paths=source_paths,
                    merged_path=merged_path,
                    expected_dims=expected_dims,
                )
                _delete_source_packages(source_paths, keep_path=merged_path)
            elif existing_sources:
                raise ValueError(
                    f"refusing partial source deletion after existing merged package: "
                    f"merged={merged_path} remaining_sources={existing_sources}"
                )
    elif auto_merge:
        _build_merged_package(
            tile_path=tile_path,
            source_paths=source_paths,
            merged_path=merged_path,
            expected_dims=expected_dims,
            dtype=dtype,
        )
        _validate_merged_package(
            tile_path=tile_path,
            merged_path=merged_path,
            teacher_names=teacher_names,
            expected_dims=expected_dims,
        )
        _validate_merged_against_sources(
            source_paths=source_paths,
            merged_path=merged_path,
            expected_dims=expected_dims,
        )
        if delete_sources:
            _delete_source_packages(source_paths, keep_path=merged_path)
    else:
        return package_idx, {name: str(source_paths[name]) for name in teacher_names}, None
    return (
        package_idx,
        {name: str(merged_path) for name in teacher_names},
        f"merged_feature_pack split={split} tile={tile_path.name} "
        f"path={merged_path} teachers={','.join(teacher_names)}",
    )


def _strip_suffix(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"package name does not match expected suffix: path={path} suffix={suffix}")
    return name[: -len(suffix)]


def _merged_output_path(tile_path: Path, first_feature_path: Path) -> Path:
    tile_stem = _strip_suffix(tile_path, TILE_SUFFIX)
    return first_feature_path.with_name(f"{tile_stem}{MERGED_FEATURE_SUFFIX}")


def _feature_tile_ids(path: Path) -> list[str]:
    _, _, record_table = read_tables(path)
    return [str(value) for value in record_table.column("tile_id").to_pylist()]


def _feature_record_table_from_tile_record_table(tile_record_table: pa.Table) -> pa.Table:
    required = ["slide_idx", "tile_x", "tile_y", "tile_id", "flags"]
    missing = [name for name in required if name not in tile_record_table.column_names]
    if missing:
        raise ValueError(f"tile record table missing columns for feature merge: {missing}")
    return pa.table({name: tile_record_table.column(name) for name in required})


def _validate_merged_package(
    *,
    tile_path: Path,
    merged_path: Path,
    teacher_names: list[str],
    expected_dims: dict[str, int],
) -> None:
    tile_header, _, tile_records = read_tables(tile_path)
    header, _, record_table = read_tables(merged_path)
    if header.get("payload_type") != MERGED_FEATURE_PAYLOAD_TYPE:
        raise ValueError(f"not a merged teacher feature package: {merged_path}")
    if int(header["num_records"]) != int(tile_header["num_records"]):
        raise ValueError(
            f"merged feature/tile record count mismatch: merged={header['num_records']} "
            f"tiles={tile_header['num_records']} path={merged_path}"
        )
    tile_ids = [str(value) for value in tile_records.column("tile_id").to_pylist()]
    merged_tile_ids = [str(value) for value in record_table.column("tile_id").to_pylist()]
    if merged_tile_ids != tile_ids:
        raise ValueError(f"merged feature tile_id order mismatch: {merged_path}")
    merged_teachers = {str(name) for name in header.get("teachers", [])}
    missing = sorted(set(teacher_names).difference(merged_teachers))
    if missing:
        raise ValueError(f"merged feature package missing teachers: path={merged_path} teachers={missing}")
    dims = {str(k): int(v) for k, v in header.get("teacher_dims", {}).items()}
    dtypes = {str(k): str(v) for k, v in header.get("teacher_dtypes", {}).items()}
    offsets = {str(k): int(v) for k, v in header.get("teacher_offsets", {}).items()}
    record_bytes = {str(k): int(v) for k, v in header.get("teacher_record_bytes", {}).items()}
    merged_record_bytes = int(header.get("merged_record_bytes", 0))
    expected_offset = 0
    for name in teacher_names:
        expected_dim = int(expected_dims[name])
        got_dim = dims.get(name)
        if got_dim != expected_dim:
            raise ValueError(
                f"merged feature dim mismatch: teacher={name} expected={expected_dim} got={got_dim} path={merged_path}"
            )
        dtype = np.dtype(dtypes.get(name, ""))
        expected_bytes = expected_dim * dtype.itemsize
        if record_bytes.get(name) != expected_bytes:
            raise ValueError(
                f"merged feature bytes mismatch: teacher={name} expected={expected_bytes} "
                f"got={record_bytes.get(name)} path={merged_path}"
            )
        if offsets.get(name) != expected_offset:
            raise ValueError(
                f"merged feature offset mismatch: teacher={name} expected={expected_offset} "
                f"got={offsets.get(name)} path={merged_path}"
            )
        expected_offset += expected_bytes
    if merged_record_bytes != expected_offset:
        raise ValueError(
            f"merged record bytes mismatch: expected={expected_offset} got={merged_record_bytes} path={merged_path}"
        )
    expected_data_length = int(header["num_records"]) * merged_record_bytes
    if int(header["data_length"]) != expected_data_length:
        raise ValueError(
            f"merged data length mismatch: expected={expected_data_length} got={header['data_length']} path={merged_path}"
        )


def _sample_rows(count: int) -> list[int]:
    if count <= 0:
        return []
    candidates = {0, count - 1, count // 2, count // 3, (2 * count) // 3}
    return sorted(row for row in candidates if 0 <= row < count)


def _validate_merged_against_sources(
    *,
    source_paths: dict[str, Path],
    merged_path: Path,
    expected_dims: dict[str, int],
) -> None:
    if not all(path.exists() for path in source_paths.values()):
        return
    merged_reader = MergedTeacherFeatureCacheReader(merged_path)
    source_readers = {}
    try:
        for name, path in source_paths.items():
            header = read_header(path)
            if header.get("payload_type") == MERGED_FEATURE_PAYLOAD_TYPE:
                continue
            source_readers[name] = _SourceFeatureReader(path)
        for row in _sample_rows(merged_reader.record_count):
            for name, source_reader in source_readers.items():
                merged_feature = merged_reader.read_feature_at(row, name)
                source_feature = source_reader.read_feature_at(row)
                expected_shape = (int(expected_dims[name]),)
                if source_feature.shape != expected_shape or merged_feature.shape != expected_shape:
                    raise ValueError(
                        f"feature sample shape mismatch: teacher={name} row={row} "
                        f"source={source_feature.shape} merged={merged_feature.shape}"
                    )
                np.testing.assert_allclose(
                    merged_feature,
                    source_feature.astype(np.float32, copy=False),
                    rtol=1e-6,
                    atol=1e-6,
                )
    finally:
        merged_reader.close()
        for reader in source_readers.values():
            reader.close()


def _build_merged_package(
    *,
    tile_path: Path,
    source_paths: dict[str, Path],
    merged_path: Path,
    expected_dims: dict[str, int],
    dtype: str,
) -> None:
    out_dtype = np.dtype(dtype)
    tile_header, tile_slides, tile_record_table = read_tables(tile_path)
    tile_ids = [str(value) for value in tile_record_table.column("tile_id").to_pylist()]
    teacher_names = list(source_paths)
    source_readers = {}
    try:
        for name, path in source_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"source teacher feature package missing: teacher={name} path={path}")
            header, _, record_table = read_tables(path)
            if header.get("payload_type") != "teacher_features":
                raise ValueError(f"not a teacher feature package: teacher={name} path={path}")
            if int(header["feature_dim"]) != int(expected_dims[name]):
                raise ValueError(
                    f"feature dim mismatch: teacher={name} expected={expected_dims[name]} "
                    f"got={header['feature_dim']} path={path}"
                )
            source_tile_ids = [str(value) for value in record_table.column("tile_id").to_pylist()]
            if source_tile_ids != tile_ids:
                raise ValueError(f"feature/tile tile_id order mismatch: teacher={name} path={path}")
            for key in ("tile_width", "tile_height", "stride_x", "stride_y"):
                if int(header[key]) != int(tile_header[key]):
                    raise ValueError(
                        f"feature/tile header mismatch: teacher={name} key={key} "
                        f"feature={header[key]} tile={tile_header[key]} path={path}"
                    )
            source_readers[name] = _SourceFeatureReader(path)

        offsets: dict[str, int] = {}
        record_bytes: dict[str, int] = {}
        offset = 0
        for name in teacher_names:
            offsets[name] = offset
            bytes_for_teacher = int(expected_dims[name]) * out_dtype.itemsize
            record_bytes[name] = bytes_for_teacher
            offset += bytes_for_teacher
        merged_record_bytes = offset

        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=merged_path.parent, delete=False) as tmp:
            data_path = Path(tmp.name)
            for row in range(len(tile_ids)):
                for name in teacher_names:
                    feature = source_readers[name].read_feature_at(row)
                    if feature.shape != (int(expected_dims[name]),):
                        raise ValueError(f"invalid source feature shape: teacher={name} row={row} shape={feature.shape}")
                    tmp.write(np.ascontiguousarray(feature.astype(out_dtype, copy=False)).tobytes(order="C"))
        try:
            expected_length = len(tile_ids) * merged_record_bytes
            if data_path.stat().st_size != expected_length:
                raise ValueError(f"merged data size mismatch: expected={expected_length} got={data_path.stat().st_size}")
            header = {
                "payload_type": MERGED_FEATURE_PAYLOAD_TYPE,
                "teachers": teacher_names,
                "teacher_dims": {name: int(expected_dims[name]) for name in teacher_names},
                "teacher_dtypes": {name: out_dtype.name for name in teacher_names},
                "teacher_offsets": offsets,
                "teacher_record_bytes": record_bytes,
                "merged_record_bytes": merged_record_bytes,
                "source_teachers": {
                    name: str(source_readers[name].header.get("teacher") or name)
                    for name in teacher_names
                },
                "dtype": out_dtype.name,
                "tile_width": int(tile_header["tile_width"]),
                "tile_height": int(tile_header["tile_height"]),
                "stride_x": int(tile_header["stride_x"]),
                "stride_y": int(tile_header["stride_y"]),
                "coordinate_mode": "tile_grid",
                "origin": "top_left",
                "slide_idx_dtype": "uint8",
                "tile_xy_dtype": "uint16",
                "flags_dtype": "uint8",
                "created_by": "hcc-sempath-auto-merge",
            }
            feature_record_table = _feature_record_table_from_tile_record_table(tile_record_table)
            build_pack_data_segment_from_file(
                merged_path,
                header,
                tile_slides,
                feature_record_table,
                data_path,
                data_length=expected_length,
                overwrite=False,
            )
        finally:
            if data_path.exists():
                data_path.unlink()
    finally:
        for reader in source_readers.values():
            reader.close()


def _delete_source_packages(source_paths: dict[str, Path], *, keep_path: Path) -> None:
    for path in sorted(set(source_paths.values())):
        if path == keep_path or not path.exists():
            continue
        path.unlink()


class _SourceFeatureReader:
    def __init__(self, path: Path) -> None:
        from ..io.feature_cache import FeatureCacheReader

        self._reader = FeatureCacheReader(path)

    @property
    def header(self) -> dict:
        return self._reader.header

    def read_feature_at(self, row: int) -> np.ndarray:
        return self._reader.read_feature_at(row)

    def close(self) -> None:
        self._reader.close()
