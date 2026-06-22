"""Compatibility shim.

The IatroCache container format has been extracted to the standalone
``iatro_iac`` package (../../iatro-iac), so it can be shared across the
HCC-CAMoE projects (SemPath image/feature caches, Course clinical-text caches).

This module re-exports the format layer unchanged. Project adapters
(``tile_package``, ``feature_cache``) keep importing from here, so no call
sites changed. New code may import from ``iatro_iac`` directly.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
from pathlib import Path

import pyarrow as pa

from iatro_iac import (  # noqa: F401
    FORMAT_VERSION,
    HEADER_BYTES,
    MAGIC,
    build_pack,
    build_pack_data_segment,
    build_pack_data_segment_from_file,
    build_pack_streaming,
    iter_payloads,
)
from iatro_iac import PackReader as _V2PackReader
from iatro_iac import read_header as _read_v2_header
from iatro_iac import read_payload as _read_v2_payload
from iatro_iac import read_tables as _read_v2_tables


LEGACY_MAGIC = b"IATROC\x00\x01"


def _read_arrow_table(handle, offset: int, length: int) -> pa.Table:
    handle.seek(offset)
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"short Arrow table read at offset={offset}: expected={length} got={len(raw)}")
    return pa.ipc.open_stream(pa.py_buffer(raw)).read_all()


def _read_fixed_header_compat(handle) -> dict:
    raw = handle.read(HEADER_BYTES)
    if len(raw) < 16:
        raise ValueError("file too small for IatroCache header")
    magic = raw[0:8]
    if magic == MAGIC:
        handle.seek(0)
        return _read_v2_header_from_handle(handle)
    if magic != LEGACY_MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    header_len = struct.unpack_from("<I", raw, 8)[0]
    version = struct.unpack_from("<I", raw, 12)[0]
    if version != 1:
        raise ValueError(f"unsupported legacy IatroCache version: {version}")
    if header_len > HEADER_BYTES - 16:
        raise ValueError(f"invalid IatroCache header length: {header_len}")
    header = json.loads(raw[16:16 + header_len].decode("utf-8"))
    header.setdefault("version", version)
    return header


def _read_v2_header_from_handle(handle) -> dict:
    # The standalone package does not expose a handle-level header reader.
    # Re-read through the public path in normal v2 entry points instead.
    raise RuntimeError("v2 handle reader should not be called")


def _is_legacy_path(path: str | Path) -> bool:
    with Path(path).open("rb") as handle:
        return handle.read(8) == LEGACY_MAGIC


def _file_size(handle) -> int:
    return os.fstat(handle.fileno()).st_size


def _table_segment_names(header: dict) -> tuple[str, str]:
    if "index_table_offset" in header:
        return "slide_table", "index_table"
    return "slide_table", "record_table"


def _validate_layout_compat(header: dict, file_size: int) -> None:
    header_bytes = int(header.get("header_bytes", HEADER_BYTES))
    if header_bytes != HEADER_BYTES:
        raise ValueError(f"unsupported header_bytes: {header_bytes}")
    slide_name, record_name = _table_segment_names(header)
    for name in (slide_name, record_name):
        offset = int(header[f"{name}_offset"])
        length = int(header[f"{name}_length"])
        if offset < 0 or length < 0 or offset + length > file_size:
            raise ValueError(
                f"{name} segment outside file bounds: offset={offset} length={length} file_size={file_size}"
            )
    data_offset = int(header["data_offset"])
    data_length = int(header["data_length"])
    if data_offset < 0 or data_length < 0 or data_offset + data_length > file_size:
        raise ValueError(
            f"data segment outside file bounds: offset={data_offset} length={data_length} file_size={file_size}"
        )


def read_header(package_path: str | Path) -> dict:
    if not _is_legacy_path(package_path):
        return _read_v2_header(package_path)
    with Path(package_path).open("rb") as handle:
        header = _read_fixed_header_compat(handle)
        _validate_layout_compat(header, _file_size(handle))
        return header


def read_tables(package_path: str | Path) -> tuple[dict, pa.Table, pa.Table]:
    if not _is_legacy_path(package_path):
        return _read_v2_tables(package_path)
    with Path(package_path).open("rb") as handle:
        header = _read_fixed_header_compat(handle)
        _validate_layout_compat(header, _file_size(handle))
        _, record_name = _table_segment_names(header)
        slides = _read_arrow_table(handle, int(header["slide_table_offset"]), int(header["slide_table_length"]))
        records = _read_arrow_table(handle, int(header[f"{record_name}_offset"]), int(header[f"{record_name}_length"]))
    if int(header["num_slides"]) != len(slides):
        raise ValueError(f"num_slides mismatch: header={header['num_slides']} table={len(slides)}")
    if int(header["num_records"]) != len(records):
        raise ValueError(f"num_records mismatch: header={header['num_records']} table={len(records)}")
    return header, slides, records


def read_payload(package_path: str | Path, header: dict, offset: int, length: int) -> bytes:
    if int(header.get("version", FORMAT_VERSION)) != 1:
        return _read_v2_payload(package_path, header, offset, length)
    if offset < 0 or length < 0 or offset + length > int(header["data_length"]):
        raise ValueError(f"payload span outside data segment: offset={offset} length={length}")
    with Path(package_path).open("rb") as handle:
        handle.seek(int(header["data_offset"]) + offset)
        payload = handle.read(length)
    if len(payload) != length:
        raise ValueError(f"short payload read at offset={offset}: expected={length} got={len(payload)}")
    return payload


class PackReader:
    def __init__(self, package_path: str | Path) -> None:
        self.package_path = Path(package_path)
        self._reader = None if _is_legacy_path(self.package_path) else _V2PackReader(self.package_path)
        self._lock = threading.Lock()
        self._file = None
        self._mmap = None
        self._header: dict | None = None
        self._slide_table: pa.Table | None = None
        self._record_table: pa.Table | None = None
        self._offsets = None
        self._lengths = None

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        if self._record_table is not None:
            return
        with self._lock:
            if self._record_table is not None:
                return
            handle = self.package_path.open("rb")
            header = _read_fixed_header_compat(handle)
            _validate_layout_compat(header, _file_size(handle))
            _, record_name = _table_segment_names(header)
            try:
                mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            except (ValueError, OSError):
                mm = None
            slide_table = _read_arrow_table(handle, int(header["slide_table_offset"]), int(header["slide_table_length"]))
            record_table = _read_arrow_table(handle, int(header[f"{record_name}_offset"]), int(header[f"{record_name}_length"]))
            self._file = handle
            self._mmap = mm
            self._header = header
            self._slide_table = slide_table
            self._record_table = record_table
            self._offsets = record_table.column("offset").to_numpy()
            self._lengths = record_table.column("length").to_numpy()

    @property
    def header(self) -> dict:
        if self._reader is not None:
            return self._reader.header
        self._ensure_loaded()
        assert self._header is not None
        return self._header

    @property
    def slide_table(self) -> pa.Table:
        if self._reader is not None:
            return self._reader.slide_table
        self._ensure_loaded()
        assert self._slide_table is not None
        return self._slide_table

    @property
    def index_table(self) -> pa.Table:
        return self.record_table

    @property
    def record_table(self) -> pa.Table:
        if self._reader is not None:
            return self._reader.record_table
        self._ensure_loaded()
        assert self._record_table is not None
        return self._record_table

    def read_payload(self, row: int) -> bytes:
        if self._reader is not None:
            return self._reader.read_payload(row)
        self._ensure_loaded()
        assert self._header is not None
        offset = int(self._offsets[row])
        length = int(self._lengths[row])
        start = int(self._header["data_offset"]) + offset
        if offset < 0 or length < 0 or offset + length > int(self._header["data_length"]):
            raise ValueError(f"payload span outside data segment at row {row}: offset={offset} length={length}")
        if self._mmap is not None:
            payload = self._mmap[start:start + length]
        else:
            with self._lock:
                assert self._file is not None
                self._file.seek(start)
                payload = self._file.read(length)
        if len(payload) != length:
            raise ValueError(f"short read at row {row}: expected={length} got={len(payload)}")
        return payload

    def read_data_span(self, offset: int, length: int) -> bytes:
        if self._reader is not None:
            return self._reader.read_data_span(offset, length)
        return read_payload(self.package_path, self.header, offset, length)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            return
        with self._lock:
            if self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            if self._file is not None:
                self._file.close()
                self._file = None
            self._header = None
            self._slide_table = None
            self._record_table = None
            self._offsets = None
            self._lengths = None
