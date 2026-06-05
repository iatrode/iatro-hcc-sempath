from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow as pa

from .iatrocache import PackReader, build_pack_data_segment_from_file, read_tables
from .manifests import TileRecord
from .tile_package import read_package_manifest, read_package_metadata
from .tile_package import _build_slide_table, _ensure_unique_column, _infer_coordinate_stride, _slide_map


def _ensure_unique_tile_ids(records: list[TileRecord]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        if record.tile_id in seen:
            duplicates.append(record.tile_id)
        seen.add(record.tile_id)
    if duplicates:
        sample = ", ".join(duplicates[:3])
        raise ValueError(f"duplicate tile_id values: count={len(duplicates)} sample={sample}")


def _build_feature_table(
    records: list[TileRecord],
    slide_to_idx: dict[str, int],
    stride_x: int,
    stride_y: int,
) -> pa.Table:
    tile_x = []
    tile_y = []
    for record in records:
        if record.x < 0 or record.y < 0:
            raise ValueError(f"negative tile coordinate: {record.tile_id} ({record.x}, {record.y})")
        if record.x % stride_x != 0 or record.y % stride_y != 0:
            raise ValueError(
                f"tile coordinate is not aligned to stride for {record.tile_id}: "
                f"({record.x}, {record.y}) stride=({stride_x}, {stride_y})"
            )
        tile_x.append(record.x // stride_x)
        tile_y.append(record.y // stride_y)
    if tile_x and (max(tile_x) > 65535 or max(tile_y) > 65535):
        raise ValueError("tile-grid coordinates exceed uint16; split the pack or use a wider schema")
    return pa.table(
        {
            "slide_idx": pa.array(
                np.array([slide_to_idx[record.slide_id] for record in records], dtype=np.uint8),
                type=pa.uint8(),
            ),
            "tile_x": pa.array(np.array(tile_x, dtype=np.uint16), type=pa.uint16()),
            "tile_y": pa.array(np.array(tile_y, dtype=np.uint16), type=pa.uint16()),
            "tile_id": [record.tile_id for record in records],
            "flags": pa.array(np.zeros(len(records), dtype=np.uint8), type=pa.uint8()),
        }
    )


def build_teacher_feature_package(
    records: list[TileRecord],
    features: Iterable[np.ndarray],
    output_path: str | Path,
    teacher_name: str = "",
    dtype: str = "float32",
    feature_dim: int | None = None,
    tile_width: int | None = None,
    tile_height: int | None = None,
    stride_x: int | None = None,
    stride_y: int | None = None,
    overwrite: bool = False,
) -> None:
    if not records:
        raise ValueError("records are empty")
    if not teacher_name:
        raise ValueError("teacher_name is required for teacher feature packages")
    dtype = np.dtype(dtype).name
    _ensure_unique_tile_ids(records)
    slide_table, slide_to_idx = _build_slide_table(records)
    if len(slide_to_idx) > 255:
        raise ValueError(f"too many slides: {len(slide_to_idx)} > 255")
    stride_x = _infer_coordinate_stride(records, "x", 1) if stride_x is None else int(stride_x)
    stride_y = _infer_coordinate_stride(records, "y", 1) if stride_y is None else int(stride_y)
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"stride must be positive, got ({stride_x}, {stride_y})")
    tile_width = stride_x if tile_width is None else int(tile_width)
    tile_height = stride_y if tile_height is None else int(tile_height)
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError(f"tile size must be positive, got ({tile_width}, {tile_height})")

    def write_matrix_stream(data_path: Path) -> tuple[int, int]:
        nonlocal feature_dim
        count = 0
        with data_path.open("wb") as handle:
            for feature in features:
                feature = np.asarray(feature).astype(dtype, copy=False)
                if feature.ndim != 1:
                    raise ValueError(f"teacher feature must be 1D, got {feature.shape}")
                feature_dim = feature.shape[0] if feature_dim is None else feature_dim
                if feature.shape[0] != feature_dim:
                    raise ValueError(f"inconsistent feature dim: got {feature.shape[0]} expected {feature_dim}")
                handle.write(np.ascontiguousarray(feature).tobytes(order="C"))
                count += 1
        if count != len(records):
            raise ValueError(f"feature count mismatch: features={count} records={len(records)}")
        assert feature_dim is not None
        return count, int(feature_dim)

    first_feature = None
    if feature_dim is None:
        rest_features = iter(features)
        try:
            first_feature = np.asarray(next(rest_features)).astype(dtype, copy=False)
        except StopIteration as exc:
            raise ValueError("features are empty") from exc
        if first_feature.ndim != 1:
            raise ValueError(f"teacher feature must be 1D, got {first_feature.shape}")
        feature_dim = first_feature.shape[0]

        def features_with_first() -> Iterable[np.ndarray]:
            assert first_feature is not None
            yield first_feature
            yield from rest_features

        features = features_with_first()

    table = _build_feature_table(records, slide_to_idx, stride_x, stride_y)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        data_path = Path(tmp.name)
    try:
        feature_count, resolved_feature_dim = write_matrix_stream(data_path)
        feature_dim = resolved_feature_dim
        record_bytes = int(feature_dim) * np.dtype(dtype).itemsize
        data_length = feature_count * record_bytes
        if data_path.stat().st_size != data_length:
            raise ValueError(f"feature data size mismatch: expected={data_length} got={data_path.stat().st_size}")
        header = {
            "payload_type": "teacher_features",
            "teacher": teacher_name,
            "feature_dim": int(feature_dim),
            "dtype": dtype,
            "feature_record_bytes": record_bytes,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "stride_x": stride_x,
            "stride_y": stride_y,
            "coordinate_mode": "tile_grid",
            "origin": "top_left",
            "slide_idx_dtype": "uint8",
            "tile_xy_dtype": "uint16",
            "flags_dtype": "uint8",
            "checksum": "crc32",
            "created_by": "hcc-sempath",
        }
        build_pack_data_segment_from_file(
            output_path,
            header,
            slide_table,
            table,
            data_path,
            data_length=data_length,
            overwrite=overwrite,
        )
    finally:
        if data_path.exists():
            data_path.unlink()


def build_teacher_feature_package_from_feature_map(
    records: list[TileRecord],
    feature_by_tile_id: dict[str, np.ndarray],
    output_path: str | Path,
    teacher_name: str = "",
    dtype: str = "float32",
    tile_width: int | None = None,
    tile_height: int | None = None,
    stride_x: int | None = None,
    stride_y: int | None = None,
    overwrite: bool = False,
) -> None:
    def features() -> Iterable[np.ndarray]:
        for record in records:
            try:
                yield feature_by_tile_id[record.tile_id]
            except KeyError as exc:
                raise FileNotFoundError(f"missing teacher feature: {record.tile_id}") from exc

    build_teacher_feature_package(
        records,
        features(),
        output_path,
        teacher_name=teacher_name,
        dtype=dtype,
        tile_width=tile_width,
        tile_height=tile_height,
        stride_x=stride_x,
        stride_y=stride_y,
        overwrite=overwrite,
    )


def build_teacher_feature_package_from_tile_package(
    tile_package_path: str | Path,
    features: Iterable[np.ndarray],
    output_path: str | Path,
    teacher_name: str = "",
    dtype: str = "float32",
    feature_dim: int | None = None,
    overwrite: bool = False,
) -> None:
    metadata = read_package_metadata(tile_package_path)
    if metadata.get("payload_type") != "image_tiles":
        raise ValueError(f"not an image tile package: {tile_package_path}")
    build_teacher_feature_package(
        read_package_manifest(tile_package_path),
        features,
        output_path,
        teacher_name=teacher_name,
        dtype=dtype,
        feature_dim=feature_dim,
        tile_width=int(metadata["tile_width"]),
        tile_height=int(metadata["tile_height"]),
        stride_x=int(metadata["stride_x"]),
        stride_y=int(metadata["stride_y"]),
        overwrite=overwrite,
    )


class FeatureCacheReader:
    def __init__(self, package_path: str | Path) -> None:
        self._reader = PackReader(package_path)
        self._tile_index: dict[str, int] | None = None
        self._tile_ids: list[str] | None = None

    @property
    def header(self) -> dict:
        return self._reader.header

    @property
    def record_table(self):
        return self._reader.record_table

    def _ensure_index(self) -> None:
        if self._tile_index is not None:
            return
        if self._reader.header.get("payload_type") != "teacher_features":
            raise ValueError(f"not a teacher feature package: {self._reader.header.get('payload_type')}")
        tile_ids = self._reader.record_table.column("tile_id")
        values = tile_ids.to_pylist()
        seen: set[str] = set()
        duplicates: list[str] = []
        for tile_id in values:
            if tile_id in seen:
                duplicates.append(tile_id)
            seen.add(tile_id)
        if duplicates:
            sample = ", ".join(duplicates[:3])
            raise ValueError(f"duplicate tile_id values in feature package: count={len(duplicates)} sample={sample}")
        self._tile_ids = [str(x) for x in values]
        self._tile_index = {self._tile_ids[i]: i for i in range(len(self._tile_ids))}

    def _record_bytes(self) -> int:
        header = self._reader.header
        dtype = np.dtype(header["dtype"])
        feature_dim = int(header["feature_dim"])
        record_bytes = int(header["feature_record_bytes"])
        expected = feature_dim * dtype.itemsize
        if record_bytes != expected:
            raise ValueError(f"invalid feature_record_bytes: expected={expected} got={record_bytes}")
        return record_bytes

    def read_feature(self, tile_id: str) -> np.ndarray:
        self._ensure_index()
        assert self._tile_index is not None
        row = self._tile_index.get(tile_id)
        if row is None:
            raise FileNotFoundError(f"missing packaged teacher feature: {tile_id}")
        return self.read_feature_at(row)

    def read_feature_at(self, row: int) -> np.ndarray:
        if row < 0 or row >= int(self._reader.header["num_records"]):
            raise IndexError(f"feature row out of range: {row}")
        record_bytes = self._record_bytes()
        dtype = np.dtype(self._reader.header["dtype"])
        feature_dim = int(self._reader.header["feature_dim"])
        payload = self._reader.read_data_span(row * record_bytes, record_bytes)
        feature = np.frombuffer(payload, dtype=dtype)
        if feature.shape != (feature_dim,):
            raise ValueError(f"invalid feature payload shape at row={row}: {feature.shape}")
        return feature.astype(np.float32, copy=True)

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

    def __getstate__(self) -> dict:
        return {"package_path": self._reader.package_path}

    def __setstate__(self, state: dict) -> None:
        self._reader = PackReader(state["package_path"])
        self._tile_index = None
        self._tile_ids = None

    def __del__(self) -> None:
        self.close()


def read_feature_package_records(package_path: str | Path) -> list[TileRecord]:
    header, slide_table, table = read_tables(package_path)
    if header.get("payload_type") != "teacher_features":
        raise ValueError(f"not a teacher feature package: {header.get('payload_type')}")
    _ensure_unique_column(table, "tile_id")
    sm = _slide_map(slide_table)
    stride_x = int(header["stride_x"])
    stride_y = int(header["stride_y"])
    records = []
    for i in range(len(table)):
        slide_idx = table.column("slide_idx")[i].as_py()
        slide_id, patient_id = sm[slide_idx]
        tile_id = table.column("tile_id")[i].as_py()
        tile_x = table.column("tile_x")[i].as_py()
        tile_y = table.column("tile_y")[i].as_py()
        records.append(
            TileRecord(
                tile_id=tile_id,
                patient_id=patient_id,
                slide_id=slide_id,
                tile_path=Path(f"features/{tile_id}.bin"),
                x=tile_x * stride_x,
                y=tile_y * stride_y,
                split="",
            )
        )
    return records
