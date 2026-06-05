"""IatroCache v1 format: indexed payload container for offline medical image
and feature cache construction.

This module provides format-level read/write with no dependency on any
project-specific data model (TileRecord, manifest, etc.).
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa

HEADER_BYTES = 65536
MAGIC = b"IATROC\x00\x01"
FORMAT_VERSION = 1


# ---- low-level helpers -------------------------------------------------------

def _arrow_to_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()


def _replace_column(table: pa.Table, name: str, values: list[int], pa_type: pa.DataType) -> pa.Table:
    column = pa.array(values, type=pa_type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, column)
    return table.append_column(name, column)


def _read_arrow_table(f, offset: int, length: int) -> pa.Table:
    f.seek(offset)
    raw = f.read(length)
    if len(raw) != length:
        raise ValueError(f"short Arrow table read at offset={offset}: expected={length} got={len(raw)}")
    return pa.ipc.open_stream(pa.py_buffer(raw)).read_all()


def _build_fixed_header(header_json_bytes: bytes) -> bytes:
    if len(header_json_bytes) > HEADER_BYTES - 16:
        raise ValueError(f"header JSON too large: {len(header_json_bytes)} bytes")
    buf = bytearray(HEADER_BYTES)
    buf[0:8] = MAGIC
    struct.pack_into("<I", buf, 8, len(header_json_bytes))
    struct.pack_into("<I", buf, 12, FORMAT_VERSION)
    buf[16:16 + len(header_json_bytes)] = header_json_bytes
    return bytes(buf)


def _read_fixed_header(f) -> dict:
    raw = f.read(HEADER_BYTES)
    if len(raw) < 16:
        raise ValueError("file too small for IatroCache header")
    if raw[0:8] != MAGIC:
        raise ValueError(f"bad magic: {raw[0:8]!r}")
    header_len = struct.unpack_from("<I", raw, 8)[0]
    version = struct.unpack_from("<I", raw, 12)[0]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported IatroCache version: {version}")
    if header_len > HEADER_BYTES - 16:
        raise ValueError(f"invalid IatroCache header length: {header_len}")
    if len(raw) < 16 + header_len:
        raise ValueError(f"truncated IatroCache header: expected={header_len} got={max(0, len(raw) - 16)}")
    return json.loads(raw[16:16 + header_len].decode("utf-8"))


def _file_size(f) -> int:
    return os.fstat(f.fileno()).st_size


def _validate_segment(header: dict, name: str, file_size: int) -> None:
    offset = int(header[f"{name}_offset"])
    length = int(header[f"{name}_length"])
    if offset < 0 or length < 0 or offset + length > file_size:
        raise ValueError(
            f"{name} segment outside file bounds: offset={offset} length={length} file_size={file_size}"
        )


def _validate_layout(header: dict, file_size: int) -> None:
    header_bytes = int(header.get("header_bytes", HEADER_BYTES))
    if header_bytes != HEADER_BYTES:
        raise ValueError(f"unsupported header_bytes: {header_bytes}")
    if int(header.get("slide_table_offset", -1)) != HEADER_BYTES:
        raise ValueError(f"unexpected slide_table_offset: {header.get('slide_table_offset')}")
    _validate_segment(header, "slide_table", file_size)
    _validate_segment(header, "record_table", file_size)
    expected_record_offset = int(header["slide_table_offset"]) + int(header["slide_table_length"])
    if int(header["record_table_offset"]) != expected_record_offset:
        raise ValueError(
            f"unexpected record_table_offset: {header['record_table_offset']} expected={expected_record_offset}"
        )
    data_offset = int(header["data_offset"])
    data_length = int(header["data_length"])
    expected_data_offset = int(header["record_table_offset"]) + int(header["record_table_length"])
    if data_offset != expected_data_offset:
        raise ValueError(f"unexpected data_offset: {data_offset} expected={expected_data_offset}")
    if data_offset < 0 or data_length < 0 or data_offset + data_length > file_size:
        raise ValueError(
            f"data segment outside file bounds: offset={data_offset} length={data_length} file_size={file_size}"
        )


# ---- public API: build -------------------------------------------------------

def build_pack(
    output_path: str | Path,
    header_json: dict,
    slide_table: pa.Table,
    record_table: pa.Table,
    payloads: list[bytes],
    *,
    overwrite: bool = False,
) -> None:
    """Assemble an IatroCache pack from pre-built tables and payloads.

    ``header_json`` is merged with required layout fields (offsets, lengths,
    counts).  Caller-supplied fields (codec, tile size, etc.) are preserved.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"pack already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offsets: list[int] = []
    lengths: list[int] = []
    crcs: list[int] = []
    total = 0
    for p in payloads:
        if len(p) > 0xFFFFFFFF:
            raise ValueError(f"payload too large for uint32 length: {len(p)} bytes")
        offsets.append(total)
        lengths.append(len(p))
        crcs.append(zlib.crc32(p) & 0xFFFFFFFF)
        total += len(p)

    record_table = _replace_column(record_table, "offset", offsets, pa.uint64())
    record_table = _replace_column(record_table, "length", lengths, pa.uint32())
    record_table = _replace_column(record_table, "crc32", crcs, pa.uint32())

    slide_bytes = _arrow_to_bytes(slide_table)
    record_bytes = _arrow_to_bytes(record_table)
    data_offset = HEADER_BYTES + len(slide_bytes) + len(record_bytes)

    # merge caller fields with layout fields
    full_header = {
        **header_json,
        "format": "IatroCache",
        "version": FORMAT_VERSION,
        "header_bytes": HEADER_BYTES,
        "slide_table_offset": HEADER_BYTES,
        "slide_table_length": len(slide_bytes),
        "record_table_offset": HEADER_BYTES + len(slide_bytes),
        "record_table_length": len(record_bytes),
        "data_offset": data_offset,
        "data_length": total,
        "num_slides": len(slide_table),
        "num_records": len(record_table),
    }
    fixed = _build_fixed_header(
        json.dumps(full_header, indent=2, sort_keys=True).encode("utf-8")
    )

    with NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("wb") as f:
            f.write(fixed)
            f.write(slide_bytes)
            f.write(record_bytes)
            for p in payloads:
                f.write(p)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_pack_streaming(
    output_path: str | Path,
    header_json: dict,
    slide_table: pa.Table,
    record_table: pa.Table,
    payloads: Iterable[bytes],
    *,
    overwrite: bool = False,
) -> None:
    """Assemble an IatroCache pack while streaming payload bytes through disk."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"pack already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offsets: list[int] = []
    lengths: list[int] = []
    crcs: list[int] = []
    total = 0
    with NamedTemporaryFile(dir=output_path.parent, delete=False) as data_tmp:
        data_tmp_path = Path(data_tmp.name)
        for payload in payloads:
            if len(payload) > 0xFFFFFFFF:
                raise ValueError(f"payload too large for uint32 length: {len(payload)} bytes")
            offsets.append(total)
            lengths.append(len(payload))
            crcs.append(zlib.crc32(payload) & 0xFFFFFFFF)
            data_tmp.write(payload)
            total += len(payload)
    try:
        if len(offsets) != len(record_table):
            raise ValueError(f"payload count mismatch: payloads={len(offsets)} records={len(record_table)}")
        record_table = _replace_column(record_table, "offset", offsets, pa.uint64())
        record_table = _replace_column(record_table, "length", lengths, pa.uint32())
        record_table = _replace_column(record_table, "crc32", crcs, pa.uint32())

        slide_bytes = _arrow_to_bytes(slide_table)
        record_bytes = _arrow_to_bytes(record_table)
        data_offset = HEADER_BYTES + len(slide_bytes) + len(record_bytes)
        full_header = {
            **header_json,
            "format": "IatroCache",
            "version": FORMAT_VERSION,
            "header_bytes": HEADER_BYTES,
            "slide_table_offset": HEADER_BYTES,
            "slide_table_length": len(slide_bytes),
            "record_table_offset": HEADER_BYTES + len(slide_bytes),
            "record_table_length": len(record_bytes),
            "data_offset": data_offset,
            "data_length": total,
            "num_slides": len(slide_table),
            "num_records": len(record_table),
        }
        fixed = _build_fixed_header(
            json.dumps(full_header, indent=2, sort_keys=True).encode("utf-8")
        )

        with NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("wb") as out, data_tmp_path.open("rb") as data_in:
                out.write(fixed)
                out.write(slide_bytes)
                out.write(record_bytes)
                while chunk := data_in.read(1024 * 1024 * 16):
                    out.write(chunk)
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    finally:
        if data_tmp_path.exists():
            data_tmp_path.unlink()


def build_pack_data_segment(
    output_path: str | Path,
    header_json: dict,
    slide_table: pa.Table,
    record_table: pa.Table,
    data: bytes,
    *,
    overwrite: bool = False,
) -> None:
    """Assemble an IatroCache pack with one caller-managed data segment."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"pack already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slide_bytes = _arrow_to_bytes(slide_table)
    record_bytes = _arrow_to_bytes(record_table)
    data_offset = HEADER_BYTES + len(slide_bytes) + len(record_bytes)
    full_header = {
        **header_json,
        "format": "IatroCache",
        "version": FORMAT_VERSION,
        "header_bytes": HEADER_BYTES,
        "slide_table_offset": HEADER_BYTES,
        "slide_table_length": len(slide_bytes),
        "record_table_offset": HEADER_BYTES + len(slide_bytes),
        "record_table_length": len(record_bytes),
        "data_offset": data_offset,
        "data_length": len(data),
        "num_slides": len(slide_table),
        "num_records": len(record_table),
    }
    fixed = _build_fixed_header(
        json.dumps(full_header, indent=2, sort_keys=True).encode("utf-8")
    )

    with NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("wb") as f:
            f.write(fixed)
            f.write(slide_bytes)
            f.write(record_bytes)
            f.write(data)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_pack_data_segment_from_file(
    output_path: str | Path,
    header_json: dict,
    slide_table: pa.Table,
    record_table: pa.Table,
    data_path: str | Path,
    *,
    data_length: int,
    overwrite: bool = False,
) -> None:
    """Assemble an IatroCache pack while copying a caller-managed data file."""
    output_path = Path(output_path)
    data_path = Path(data_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"pack already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slide_bytes = _arrow_to_bytes(slide_table)
    record_bytes = _arrow_to_bytes(record_table)
    data_offset = HEADER_BYTES + len(slide_bytes) + len(record_bytes)
    full_header = {
        **header_json,
        "format": "IatroCache",
        "version": FORMAT_VERSION,
        "header_bytes": HEADER_BYTES,
        "slide_table_offset": HEADER_BYTES,
        "slide_table_length": len(slide_bytes),
        "record_table_offset": HEADER_BYTES + len(slide_bytes),
        "record_table_length": len(record_bytes),
        "data_offset": data_offset,
        "data_length": int(data_length),
        "num_slides": len(slide_table),
        "num_records": len(record_table),
    }
    fixed = _build_fixed_header(
        json.dumps(full_header, indent=2, sort_keys=True).encode("utf-8")
    )

    with NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        copied = 0
        with tmp_path.open("wb") as out, data_path.open("rb") as data_in:
            out.write(fixed)
            out.write(slide_bytes)
            out.write(record_bytes)
            while chunk := data_in.read(1024 * 1024 * 16):
                copied += len(chunk)
                out.write(chunk)
        if copied != int(data_length):
            raise ValueError(f"data_length mismatch: expected={data_length} copied={copied}")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---- public API: read --------------------------------------------------------

def read_header(package_path: str | Path) -> dict:
    """Read the JSON header from an IatroCache pack."""
    with Path(package_path).open("rb") as f:
        header = _read_fixed_header(f)
        _validate_layout(header, _file_size(f))
        return header


def read_tables(package_path: str | Path) -> tuple[dict, pa.Table, pa.Table]:
    """Return (header, slide_table, record_table)."""
    with Path(package_path).open("rb") as f:
        header = _read_fixed_header(f)
        _validate_layout(header, _file_size(f))
        slides = _read_arrow_table(f, header["slide_table_offset"], header["slide_table_length"])
        records = _read_arrow_table(f, header["record_table_offset"], header["record_table_length"])
    if int(header["num_slides"]) != len(slides):
        raise ValueError(f"num_slides mismatch: header={header['num_slides']} table={len(slides)}")
    if int(header["num_records"]) != len(records):
        raise ValueError(f"num_records mismatch: header={header['num_records']} table={len(records)}")
    return header, slides, records


def read_payload(package_path: str | Path, header: dict, offset: int, length: int) -> bytes:
    """Read a single payload record from the data segment."""
    if offset < 0 or length < 0 or offset + length > int(header["data_length"]):
        raise ValueError(f"payload span outside data segment: offset={offset} length={length}")
    with Path(package_path).open("rb") as f:
        f.seek(header["data_offset"] + offset)
        payload = f.read(length)
    if len(payload) != length:
        raise ValueError(f"short payload read at offset={offset}: expected={length} got={len(payload)}")
    return payload


def iter_payloads(package_path: str | Path) -> Iterator[bytes]:
    """Yield all payloads in data-offset order."""
    with Path(package_path).open("rb") as f:
        header = _read_fixed_header(f)
        _validate_layout(header, _file_size(f))
        record_table = _read_arrow_table(
            f, header["record_table_offset"], header["record_table_length"]
        )
        offsets = record_table.column("offset")
        lengths = record_table.column("length")
        for i in range(len(record_table)):
            offset = offsets[i].as_py()
            length = lengths[i].as_py()
            if offset < 0 or length < 0 or offset + length > int(header["data_length"]):
                raise ValueError(f"payload span outside data segment at row {i}: offset={offset} length={length}")
            f.seek(header["data_offset"] + offset)
            payload = f.read(length)
            if len(payload) != length:
                raise ValueError(f"short payload read at row {i}: expected={length} got={len(payload)}")
            yield payload


class PackReader:
    """Stateful reader that keeps the file handle open for random access."""

    def __init__(self, package_path: str | Path) -> None:
        self.package_path = Path(package_path)
        self._file = None
        self._header: dict | None = None
        self._slide_table: pa.Table | None = None
        self._record_table: pa.Table | None = None
        self._offsets = None
        self._lengths = None

    def _ensure_header(self) -> None:
        if self._header is not None:
            return
        self._file = self.package_path.open("rb")
        self._header = _read_fixed_header(self._file)
        _validate_layout(self._header, _file_size(self._file))

    def _ensure_loaded(self) -> None:
        if self._slide_table is not None and self._record_table is not None:
            return
        self._ensure_header()
        assert self._file is not None
        assert self._header is not None
        self._slide_table = _read_arrow_table(
            self._file, self._header["slide_table_offset"], self._header["slide_table_length"]
        )
        self._record_table = _read_arrow_table(
            self._file, self._header["record_table_offset"], self._header["record_table_length"]
        )
        if int(self._header["num_slides"]) != len(self._slide_table):
            raise ValueError(f"num_slides mismatch: header={self._header['num_slides']} table={len(self._slide_table)}")
        if int(self._header["num_records"]) != len(self._record_table):
            raise ValueError(
                f"num_records mismatch: header={self._header['num_records']} table={len(self._record_table)}"
            )
        if "offset" in self._record_table.column_names:
            import numpy as np
            self._offsets = self._record_table.column("offset").to_numpy()
            self._lengths = self._record_table.column("length").to_numpy()

    @property
    def header(self) -> dict:
        self._ensure_header()
        assert self._header is not None
        return self._header

    @property
    def slide_table(self) -> pa.Table:
        self._ensure_loaded()
        assert self._slide_table is not None
        return self._slide_table

    @property
    def record_table(self) -> pa.Table:
        self._ensure_loaded()
        assert self._record_table is not None
        return self._record_table

    def read_payload(self, row: int) -> bytes:
        self._ensure_loaded()
        assert self._file is not None
        assert self._header is not None
        assert self._record_table is not None
        offset = int(self._offsets[row])
        length = int(self._lengths[row])
        if offset < 0 or length < 0 or offset + length > int(self._header["data_length"]):
            raise ValueError(f"payload span outside data segment at row {row}: offset={offset} length={length}")
        self._file.seek(self._header["data_offset"] + offset)
        payload = self._file.read(length)
        if len(payload) != length:
            raise ValueError(f"short payload read at row {row}: expected={length} got={len(payload)}")
        return payload

    def read_data_span(self, offset: int, length: int) -> bytes:
        self._ensure_header()
        assert self._file is not None
        assert self._header is not None
        if offset < 0 or length < 0 or offset + length > int(self._header["data_length"]):
            raise ValueError(f"data span outside data segment: offset={offset} length={length}")
        self._file.seek(self._header["data_offset"] + offset)
        payload = self._file.read(length)
        if len(payload) != length:
            raise ValueError(f"short data span read: expected={length} got={len(payload)}")
        return payload

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._header = None
        self._slide_table = None
        self._record_table = None
        self._offsets = None
        self._lengths = None

    def __getstate__(self) -> dict:
        return {"package_path": self.package_path}

    def __setstate__(self, state: dict) -> None:
        self.package_path = state["package_path"]
        self._file = None
        self._header = None
        self._slide_table = None
        self._record_table = None

    def __del__(self) -> None:
        self.close()
