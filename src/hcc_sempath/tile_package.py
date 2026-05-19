from __future__ import annotations

import csv
import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

import imagecodecs
import numpy as np
from PIL import Image

from .manifests import TILE_COLUMNS, TileRecord, read_tile_manifest


PACKAGE_VERSION = "HCCSPK-v1"


def _tarinfo(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info


def _manifest_bytes(records: list[TileRecord]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TILE_COLUMNS)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "tile_id": record.tile_id,
                "patient_id": record.patient_id,
                "slide_id": record.slide_id,
                "tile_path": f"tiles/{record.tile_id}.jxl",
                "x": record.x,
                "y": record.y,
                "split": record.split,
            }
        )
    return buffer.getvalue().encode("utf-8")


def encode_jxl(image_path: Path, lossless: bool, distance: float | None, effort: int | None) -> bytes:
    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"))
    return imagecodecs.jpegxl_encode(arr, lossless=lossless, distance=distance, effort=effort)


def decode_jxl(payload: bytes) -> Image.Image:
    arr = imagecodecs.jpegxl_decode(payload)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")


def build_tile_package(
    manifest_path: str | Path,
    output_path: str | Path,
    tile_root: str | Path | None = None,
    lossless: bool = False,
    distance: float | None = 1.0,
    effort: int | None = 7,
    overwrite: bool = False,
) -> None:
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    tile_root_path = Path(tile_root) if tile_root is not None else None
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"package already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = read_tile_manifest(manifest_path)
    metadata = {
        "format": PACKAGE_VERSION,
        "tile_count": len(records),
        "codec": "jpegxl",
        "lossless": lossless,
        "distance": distance,
        "effort": effort,
        "manifest": "manifest.csv",
    }
    with NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "w") as tar:
            metadata_payload = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")
            tar.addfile(_tarinfo("metadata.json", metadata_payload), io.BytesIO(metadata_payload))
            manifest_payload = _manifest_bytes(records)
            tar.addfile(_tarinfo("manifest.csv", manifest_payload), io.BytesIO(manifest_payload))
            for record in records:
                image_path = record.tile_path
                if not image_path.is_absolute() and tile_root_path is not None:
                    image_path = tile_root_path / image_path
                payload = encode_jxl(image_path, lossless=lossless, distance=distance, effort=effort)
                tar.addfile(_tarinfo(f"tiles/{record.tile_id}.jxl", payload), io.BytesIO(payload))
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_package_metadata(package_path: str | Path) -> dict:
    with tarfile.open(package_path, "r") as tar:
        member = tar.getmember("metadata.json")
        handle = tar.extractfile(member)
        if handle is None:
            raise ValueError("package metadata is not readable")
        metadata = json.loads(handle.read().decode("utf-8"))
    if metadata.get("format") != PACKAGE_VERSION:
        raise ValueError(f"unsupported package format: {metadata.get('format')}")
    return metadata


def read_package_manifest(package_path: str | Path) -> list[TileRecord]:
    with tarfile.open(package_path, "r") as tar:
        manifest_handle = tar.extractfile("manifest.csv")
        if manifest_handle is None:
            raise ValueError("package manifest is not readable")
        manifest_text = manifest_handle.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(manifest_text))
    records = []
    for row in reader:
        records.append(
            TileRecord(
                tile_id=row["tile_id"],
                patient_id=row["patient_id"],
                slide_id=row["slide_id"],
                tile_path=Path(row["tile_path"]),
                x=int(row["x"]),
                y=int(row["y"]),
                split=row["split"],
            )
        )
    return records


class TilePackageReader:
    def __init__(self, package_path: str | Path) -> None:
        self.package_path = Path(package_path)
        self._tar: tarfile.TarFile | None = None

    def _handle(self) -> tarfile.TarFile:
        if self._tar is None:
            self._tar = tarfile.open(self.package_path, "r")
        return self._tar

    def read_image(self, tile_id: str) -> Image.Image:
        tile_handle = self._handle().extractfile(f"tiles/{tile_id}.jxl")
        if tile_handle is None:
            raise FileNotFoundError(f"missing packaged tile: {tile_id}")
        return decode_jxl(tile_handle.read())

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __getstate__(self) -> dict:
        return {"package_path": self.package_path}

    def __setstate__(self, state: dict) -> None:
        self.package_path = state["package_path"]
        self._tar = None

    def __del__(self) -> None:
        self.close()


def iter_package_tiles(package_path: str | Path) -> Iterator[tuple[TileRecord, Image.Image]]:
    with tarfile.open(package_path, "r") as tar:
        manifest_handle = tar.extractfile("manifest.csv")
        if manifest_handle is None:
            raise ValueError("package manifest is not readable")
        manifest_text = manifest_handle.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(manifest_text))
        for row in reader:
            record = TileRecord(
                tile_id=row["tile_id"],
                patient_id=row["patient_id"],
                slide_id=row["slide_id"],
                tile_path=Path(row["tile_path"]),
                x=int(row["x"]),
                y=int(row["y"]),
                split=row["split"],
            )
            tile_handle = tar.extractfile(f"tiles/{record.tile_id}.jxl")
            if tile_handle is None:
                raise FileNotFoundError(f"missing packaged tile: {record.tile_id}")
            yield record, decode_jxl(tile_handle.read())
