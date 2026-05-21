"""Project-level adapter: IatroCache packs with TileRecord / JXL payloads."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from PIL import Image
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import imagecodecs
from tqdm import tqdm

from .iatrocache import PackReader, build_pack, read_header, read_tables
from .manifests import TileRecord


def encode_jxl_array(arr: np.ndarray, lossless: bool, distance: float | None, effort: int | None) -> bytes:
    return imagecodecs.jpegxl_encode(arr, lossless=lossless, distance=distance, effort=effort)


def encode_jxl(image_path: Path, lossless: bool, distance: float | None, effort: int | None) -> bytes:
    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"))
    return encode_jxl_array(arr, lossless=lossless, distance=distance, effort=effort)


def decode_jxl(payload: bytes) -> Image.Image:
    arr = imagecodecs.jpegxl_decode(payload)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")


def _build_slide_table(records: list[TileRecord]) -> tuple[pa.Table, dict[str, int]]:
    seen: list[tuple[str, str]] = []
    slide_to_idx: dict[str, int] = {}
    for r in records:
        if r.slide_id not in slide_to_idx:
            slide_to_idx[r.slide_id] = len(seen)
            seen.append((r.slide_id, r.patient_id))
    table = pa.table({
        "slide_idx": pa.array(np.arange(len(seen), dtype=np.uint8), type=pa.uint8()),
        "slide_id": [s[0] for s in seen],
        "patient_id": [s[1] for s in seen],
    })
    return table, slide_to_idx


def _build_record_table(
    records: list[TileRecord],
    slide_to_idx: dict[str, int],
    stride_x: int,
    stride_y: int,
    offsets: list[int],
    lengths: list[int],
    crcs: list[int],
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
    return pa.table({
        "slide_idx": pa.array(
            np.array([slide_to_idx[r.slide_id] for r in records], dtype=np.uint8),
            type=pa.uint8(),
        ),
        "tile_x": pa.array(np.array(tile_x, dtype=np.uint16), type=pa.uint16()),
        "tile_y": pa.array(np.array(tile_y, dtype=np.uint16), type=pa.uint16()),
        "tile_id": [r.tile_id for r in records],
        "split": [r.split for r in records],
        "offset": pa.array(np.array(offsets, dtype=np.uint64), type=pa.uint64()),
        "length": pa.array(np.array(lengths, dtype=np.uint32), type=pa.uint32()),
        "crc32": pa.array(np.array(crcs, dtype=np.uint32), type=pa.uint32()),
        "flags": pa.array(np.zeros(len(records), dtype=np.uint8), type=pa.uint8()),
    })


def _infer_coordinate_stride(records: list[TileRecord], axis: str, default: int) -> int:
    values = [getattr(record, axis) for record in records if getattr(record, axis) > 0]
    if not values:
        return default
    stride = values[0]
    for value in values[1:]:
        stride = gcd(stride, value)
    return stride or default


def _slide_map(slide_table: pa.Table) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    for i in range(len(slide_table)):
        idx = slide_table.column("slide_idx")[i].as_py()
        result[idx] = (
            slide_table.column("slide_id")[i].as_py(),
            slide_table.column("patient_id")[i].as_py(),
        )
    return result


def _to_tile_record(
    header: dict,
    record_table: pa.Table,
    slide_map: dict[int, tuple[str, str]],
    i: int,
) -> TileRecord:
    slide_idx = record_table.column("slide_idx")[i].as_py()
    slide_id, patient_id = slide_map[slide_idx]
    tile_id = record_table.column("tile_id")[i].as_py()
    tile_x = record_table.column("tile_x")[i].as_py()
    tile_y = record_table.column("tile_y")[i].as_py()
    if header.get("coordinate_mode") == "tile_grid":
        x = tile_x * int(header["stride_x"])
        y = tile_y * int(header["stride_y"])
    else:
        x = tile_x
        y = tile_y
    return TileRecord(
        tile_id=tile_id,
        patient_id=patient_id,
        slide_id=slide_id,
        tile_path=Path(f"tiles/{tile_id}.jxl"),
        x=x,
        y=y,
        split=record_table.column("split")[i].as_py(),
    )


# ---- public API (unchanged signatures) ---------------------------------------

def build_tile_package(
    manifest_path: str | Path,
    output_path: str | Path,
    tile_root: str | Path | None = None,
    lossless: bool = False,
    distance: float | None = 1.0,
    effort: int | None = 7,
    overwrite: bool = False,
    stride_x: int | None = None,
    stride_y: int | None = None,
    workers: int = 1,
    show_progress: bool = False,
) -> None:
    from .manifests import read_tile_manifest

    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    tile_root_path = Path(tile_root) if tile_root is not None else None

    records = read_tile_manifest(manifest_path)
    if not records:
        raise ValueError("manifest is empty")

    slide_table, slide_to_idx = _build_slide_table(records)
    if len(slide_to_idx) > 255:
        raise ValueError(f"too many slides: {len(slide_to_idx)} > 255")

    image_paths = []
    tile_sizes: set[tuple[int, int]] = set()
    for record in records:
        image_path = record.tile_path
        if not image_path.is_absolute() and tile_root_path is not None:
            image_path = tile_root_path / image_path
        with Image.open(image_path) as image:
            tile_sizes.add(image.size)
        image_paths.append(image_path)
    if len(tile_sizes) != 1:
        raise ValueError(f"all packaged tiles must share one size, got {sorted(tile_sizes)}")
    tile_width, tile_height = next(iter(tile_sizes))
    stride_x = _infer_coordinate_stride(records, "x", tile_width) if stride_x is None else int(stride_x)
    stride_y = _infer_coordinate_stride(records, "y", tile_height) if stride_y is None else int(stride_y)
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"stride must be positive, got ({stride_x}, {stride_y})")

    def encode_path(image_path: Path) -> bytes:
        return encode_jxl(image_path, lossless=lossless, distance=distance, effort=effort)

    workers = max(1, int(workers))
    if workers == 1:
        iterator = image_paths
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding JXL tiles", unit="tile")
        payloads = [encode_path(image_path) for image_path in iterator]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            payload_iter = executor.map(encode_path, image_paths)
            if show_progress:
                payload_iter = tqdm(payload_iter, total=len(image_paths), desc="Encoding JXL tiles", unit="tile")
            payloads = list(payload_iter)

    # compute offsets/lengths/crcs without iatrocache import internals
    offsets: list[int] = []
    lengths: list[int] = []
    crcs: list[int] = []
    total = 0
    for p in payloads:
        offsets.append(total)
        lengths.append(len(p))
        crcs.append(0)  # crc filled by build_pack
        total += len(p)

    record_table = _build_record_table(records, slide_to_idx, stride_x, stride_y, offsets, lengths, crcs)

    header_json = {
        "payload_type": "image_tiles",
        "codec": "jxl",
        "codec_params": {
            "mode": "lossless" if lossless else "lossy",
            "lossless": lossless,
            "distance": distance,
            "effort": effort,
            "tile_color_space": "RGB",
            "input_dtype": "uint8",
        },
        "tile_width": tile_width,
        "tile_height": tile_height,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "coordinate_mode": "tile_grid",
        "origin": "top_left",
        "slide_idx_dtype": "uint8",
        "tile_xy_dtype": "uint16",
        "offset_dtype": "uint64",
        "length_dtype": "uint32",
        "flags_dtype": "uint8",
        "checksum": "crc32",
        "max_slides_per_pack": 255,
        "created_by": "hcc-sempath",
    }
    build_pack(
        output_path,
        header_json,
        slide_table,
        record_table,
        payloads,
        overwrite=overwrite,
    )


def build_tile_package_from_records(
    records: list[TileRecord],
    payloads: list[bytes],
    output_path: str | Path,
    *,
    tile_width: int,
    tile_height: int,
    lossless: bool = False,
    distance: float | None = 1.0,
    effort: int | None = 7,
    overwrite: bool = False,
    stride_x: int | None = None,
    stride_y: int | None = None,
    extra_header: dict | None = None,
) -> None:
    output_path = Path(output_path)
    if not records:
        raise ValueError("records are empty")
    if len(records) != len(payloads):
        raise ValueError(f"record/payload count mismatch: {len(records)} != {len(payloads)}")

    slide_table, slide_to_idx = _build_slide_table(records)
    if len(slide_to_idx) > 255:
        raise ValueError(f"too many slides: {len(slide_to_idx)} > 255")

    stride_x = _infer_coordinate_stride(records, "x", tile_width) if stride_x is None else int(stride_x)
    stride_y = _infer_coordinate_stride(records, "y", tile_height) if stride_y is None else int(stride_y)
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"stride must be positive, got ({stride_x}, {stride_y})")

    offsets: list[int] = []
    lengths: list[int] = []
    crcs: list[int] = []
    total = 0
    for payload in payloads:
        offsets.append(total)
        lengths.append(len(payload))
        crcs.append(0)
        total += len(payload)
    record_table = _build_record_table(records, slide_to_idx, stride_x, stride_y, offsets, lengths, crcs)

    header_json = {
        "payload_type": "image_tiles",
        "codec": "jxl",
        "codec_params": {
            "mode": "lossless" if lossless else "lossy",
            "lossless": lossless,
            "distance": distance,
            "effort": effort,
            "tile_color_space": "RGB",
            "input_dtype": "uint8",
        },
        "tile_width": int(tile_width),
        "tile_height": int(tile_height),
        "stride_x": stride_x,
        "stride_y": stride_y,
        "coordinate_mode": "tile_grid",
        "origin": "top_left",
        "slide_idx_dtype": "uint8",
        "tile_xy_dtype": "uint16",
        "offset_dtype": "uint64",
        "length_dtype": "uint32",
        "flags_dtype": "uint8",
        "checksum": "crc32",
        "max_slides_per_pack": 255,
        "created_by": "hcc-sempath",
    }
    if extra_header:
        header_json.update(extra_header)
    build_pack(output_path, header_json, slide_table, record_table, payloads, overwrite=overwrite)


def read_package_metadata(package_path: str | Path) -> dict:
    return read_header(package_path)


def read_package_manifest(package_path: str | Path) -> list[TileRecord]:
    header, slide_table, record_table = read_tables(package_path)
    sm = _slide_map(slide_table)
    return [_to_tile_record(header, record_table, sm, i) for i in range(len(record_table))]


class TilePackageReader:
    def __init__(self, package_path: str | Path) -> None:
        self._reader = PackReader(package_path)
        self._slide_map: dict[int, tuple[str, str]] | None = None
        self._tile_index: dict[str, int] | None = None

    def _ensure_index(self) -> None:
        if self._tile_index is not None:
            return
        self._slide_map = _slide_map(self._reader.slide_table)
        tile_ids = self._reader.record_table.column("tile_id")
        self._tile_index = {tile_ids[i].as_py(): i for i in range(len(tile_ids))}

    def read_image(self, tile_id: str) -> Image.Image:
        self._ensure_index()
        assert self._tile_index is not None
        row = self._tile_index.get(tile_id)
        if row is None:
            raise FileNotFoundError(f"missing packaged tile: {tile_id}")
        return decode_jxl(self._reader.read_payload(row))

    def close(self) -> None:
        self._reader.close()
        self._slide_map = None
        self._tile_index = None

    def __getstate__(self) -> dict:
        return {"package_path": self._reader.package_path}

    def __setstate__(self, state: dict) -> None:
        self._reader = PackReader(state["package_path"])
        self._slide_map = None
        self._tile_index = None

    def __del__(self) -> None:
        self.close()


def iter_package_tiles(package_path: str | Path) -> Iterator[tuple[TileRecord, Image.Image]]:
    header, slide_table, record_table = read_tables(package_path)
    sm = _slide_map(slide_table)
    reader = PackReader(package_path)
    try:
        for i in range(len(record_table)):
            payload = reader.read_payload(i)
            yield _to_tile_record(header, record_table, sm, i), decode_jxl(payload)
    finally:
        reader.close()
