from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from .iatrocache import PackReader, build_pack, read_tables
from .manifests import TileRecord, read_tile_manifest
from .tile_package import read_package_manifest


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


def _build_feature_table(records: list[TileRecord]) -> pa.Table:
    return pa.table(
        {
            "tile_id": [record.tile_id for record in records],
            "patient_id": [record.patient_id for record in records],
            "slide_id": [record.slide_id for record in records],
            "tile_x": pa.array([record.x for record in records], type=pa.uint32()),
            "tile_y": pa.array([record.y for record in records], type=pa.uint32()),
            "split": [record.split for record in records],
            "offset": pa.array(np.zeros(len(records), dtype=np.uint64), type=pa.uint64()),
            "length": pa.array(np.zeros(len(records), dtype=np.uint32), type=pa.uint32()),
            "crc32": pa.array(np.zeros(len(records), dtype=np.uint32), type=pa.uint32()),
            "flags": pa.array(np.zeros(len(records), dtype=np.uint8), type=pa.uint8()),
        }
    )


def build_teacher_feature_package(
    records: list[TileRecord],
    feature_dir: str | Path,
    output_path: str | Path,
    teacher_name: str = "",
    dtype: str = "float32",
    overwrite: bool = False,
) -> None:
    feature_dir = Path(feature_dir)
    if not records:
        raise ValueError("records are empty")
    _ensure_unique_tile_ids(records)
    payloads = []
    feature_dim = None
    for record in records:
        feature_path = feature_dir / f"{record.tile_id}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"missing teacher feature: {feature_path}")
        feature = np.load(feature_path).astype(dtype, copy=False)
        if feature.ndim != 1:
            raise ValueError(f"teacher feature must be 1D: {feature_path}")
        feature_dim = feature.shape[0] if feature_dim is None else feature_dim
        if feature.shape[0] != feature_dim:
            raise ValueError(f"inconsistent feature dim: {feature_path} got {feature.shape[0]} expected {feature_dim}")
        payloads.append(feature.tobytes(order="C"))
    table = _build_feature_table(records)
    header = {
        "payload_type": "teacher_features",
        "teacher": teacher_name,
        "feature_dim": int(feature_dim),
        "dtype": dtype,
        "record_schema": "tile_feature_cache",
        "checksum": "crc32",
        "created_by": "hcc-sempath",
    }
    build_pack(output_path, header, pa.table({}), table, payloads, overwrite=overwrite)


def build_teacher_feature_package_from_manifest(
    manifest_path: str | Path,
    feature_dir: str | Path,
    output_path: str | Path,
    teacher_name: str = "",
    dtype: str = "float32",
    overwrite: bool = False,
) -> None:
    build_teacher_feature_package(
        read_tile_manifest(manifest_path),
        feature_dir,
        output_path,
        teacher_name=teacher_name,
        dtype=dtype,
        overwrite=overwrite,
    )


def build_teacher_feature_package_from_tile_package(
    tile_package_path: str | Path,
    feature_dir: str | Path,
    output_path: str | Path,
    teacher_name: str = "",
    dtype: str = "float32",
    overwrite: bool = False,
) -> None:
    build_teacher_feature_package(
        read_package_manifest(tile_package_path),
        feature_dir,
        output_path,
        teacher_name=teacher_name,
        dtype=dtype,
        overwrite=overwrite,
    )


class FeatureCacheReader:
    def __init__(self, package_path: str | Path) -> None:
        self._reader = PackReader(package_path)
        self._tile_index: dict[str, int] | None = None

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
        self._tile_index = {tile_ids[i].as_py(): i for i in range(len(tile_ids))}

    def read_feature(self, tile_id: str) -> np.ndarray:
        self._ensure_index()
        assert self._tile_index is not None
        row = self._tile_index.get(tile_id)
        if row is None:
            raise FileNotFoundError(f"missing packaged teacher feature: {tile_id}")
        dtype = np.dtype(self._reader.header["dtype"])
        feature_dim = int(self._reader.header["feature_dim"])
        payload = self._reader.read_payload(row)
        feature = np.frombuffer(payload, dtype=dtype)
        if feature.shape != (feature_dim,):
            raise ValueError(f"invalid feature payload shape for {tile_id}: {feature.shape}")
        return feature.astype(np.float32, copy=True)

    def close(self) -> None:
        self._reader.close()
        self._tile_index = None

    def __getstate__(self) -> dict:
        return {"package_path": self._reader.package_path}

    def __setstate__(self, state: dict) -> None:
        self._reader = PackReader(state["package_path"])
        self._tile_index = None

    def __del__(self) -> None:
        self.close()


def read_feature_package_records(package_path: str | Path) -> list[TileRecord]:
    header, _, table = read_tables(package_path)
    if header.get("payload_type") != "teacher_features":
        raise ValueError(f"not a teacher feature package: {header.get('payload_type')}")
    records = []
    for i in range(len(table)):
        tile_id = table.column("tile_id")[i].as_py()
        records.append(
            TileRecord(
                tile_id=tile_id,
                patient_id=table.column("patient_id")[i].as_py(),
                slide_id=table.column("slide_id")[i].as_py(),
                tile_path=Path(f"features/{tile_id}.bin"),
                x=table.column("tile_x")[i].as_py(),
                y=table.column("tile_y")[i].as_py(),
                split=table.column("split")[i].as_py(),
            )
        )
    return records
