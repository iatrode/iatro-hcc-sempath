#!/usr/bin/env python3
r"""Upgrade IatroCache v1 containers to v2 without rewriting payload data.

This is a temporary migration utility for HCC-SemPath pathology caches.  It
performs a complete read-only preflight before changing any file.  Applied
conversions clone each package on the same APFS volume, rewrite only the fixed
64 KiB header, validate the clone with the current v2 reader, and atomically
replace the source.

The migration is intentionally narrow:

* ``IATROC\0\1`` becomes ``IATROC\0\2``;
* the binary and JSON format versions become 2;
* ``record_table_offset/length`` become ``index_table_offset/length``;
* Arrow tables and payload bytes are not modified.

Legacy compressed feature packs carrying ``feature_layout=matrix`` require a
data rewrite and are rejected here.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import struct
import sys
import uuid
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa


HEADER_BYTES = 65_536
V1_MAGIC = b"IATROC\x00\x01"
V2_MAGIC = b"IATROC\x00\x02"
V1_VERSION = 1
V2_VERSION = 2
ALLOWED_PAYLOAD_TYPES = {
    "image_tiles",
    "teacher_features",
    "merged_teacher_features",
}


class UpgradeError(RuntimeError):
    """Raised when a package is unsafe or unsupported for this migration."""


@dataclass(frozen=True)
class Inspection:
    path: Path
    state: str
    payload_type: str
    size: int
    mtime_ns: int
    inode: int
    device: int
    fixed_header_sha256: str
    header: dict
    table_columns: tuple[str, ...]


def _path_token(path: Path) -> str:
    return hashlib.sha256(os.fsencode(path)).hexdigest()[:12]


def _display_path(path: Path, show_paths: bool) -> str:
    return str(path) if show_paths else f"path_token={_path_token(path)}"


def _read_fixed_header(path: Path) -> tuple[bytes, bytes, int, int, dict]:
    with path.open("rb") as handle:
        fixed = handle.read(HEADER_BYTES)
    if len(fixed) != HEADER_BYTES:
        raise UpgradeError(f"file is smaller than the fixed header: bytes={len(fixed)}")
    magic = fixed[:8]
    header_length = struct.unpack_from("<I", fixed, 8)[0]
    binary_version = struct.unpack_from("<I", fixed, 12)[0]
    if header_length > HEADER_BYTES - 16:
        raise UpgradeError(f"invalid header JSON length: {header_length}")
    try:
        header = json.loads(fixed[16 : 16 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpgradeError(f"invalid header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise UpgradeError("header JSON must be an object")
    return fixed, magic, header_length, binary_version, header


def _segment(header: dict, name: str, file_size: int) -> tuple[int, int]:
    try:
        offset = int(header[f"{name}_offset"])
        length = int(header[f"{name}_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UpgradeError(f"missing or invalid {name} segment") from exc
    if offset < 0 or length < 0 or offset + length > file_size:
        raise UpgradeError(
            f"{name} segment outside file bounds: offset={offset} length={length} file_size={file_size}"
        )
    return offset, length


def _read_arrow_table(path: Path, offset: int, length: int, name: str) -> pa.Table:
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if len(raw) != length:
        raise UpgradeError(f"short {name} read: expected={length} got={len(raw)}")
    try:
        return pa.ipc.open_stream(pa.py_buffer(raw)).read_all()
    except (pa.ArrowException, OSError) as exc:
        raise UpgradeError(f"invalid {name} Arrow stream: {exc}") from exc


def _require_columns(table: pa.Table, required: Iterable[str], table_name: str) -> None:
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise UpgradeError(f"{table_name} missing columns: {missing}")


def _validate_payload_contract(header: dict, index_table: pa.Table) -> None:
    payload_type = str(header.get("payload_type") or "")
    if payload_type not in ALLOWED_PAYLOAD_TYPES:
        raise UpgradeError(f"unsupported payload_type: {payload_type or '<missing>'}")

    if header.get("feature_layout") == "matrix":
        raise UpgradeError("legacy compressed feature_layout=matrix requires a data rewrite")

    if payload_type == "image_tiles":
        if header.get("codec") != "jxl":
            raise UpgradeError(f"image_tiles codec must be jxl, got: {header.get('codec')}")
        _require_columns(index_table, ("offset", "length", "crc32"), "image tile index table")
        offsets = index_table.column("offset").to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        lengths = index_table.column("length").to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
        if len(offsets):
            ends = offsets + lengths
            if bool(np.any(ends < offsets)) or int(ends.max()) > int(header["data_length"]):
                raise UpgradeError("image tile payload span lies outside data segment")
        return

    _require_columns(index_table, ("tile_id",), "feature index table")
    if payload_type == "teacher_features":
        try:
            feature_dim = int(header["feature_dim"])
            dtype = np.dtype(header["dtype"])
            record_bytes = int(header["feature_record_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UpgradeError("teacher_features fixed-record metadata is incomplete") from exc
        expected_record_bytes = feature_dim * dtype.itemsize
        if record_bytes != expected_record_bytes:
            raise UpgradeError(
                f"feature_record_bytes mismatch: expected={expected_record_bytes} got={record_bytes}"
            )
        expected_data_length = int(header["num_records"]) * record_bytes
        if int(header["data_length"]) != expected_data_length:
            raise UpgradeError(
                f"teacher feature data_length mismatch: expected={expected_data_length} got={header['data_length']}"
            )


def _validate_layout_and_tables(path: Path, header: dict, table_name: str, file_size: int) -> pa.Table:
    if int(header.get("header_bytes", -1)) != HEADER_BYTES:
        raise UpgradeError(f"unsupported header_bytes: {header.get('header_bytes')}")
    slide_offset, slide_length = _segment(header, "slide_table", file_size)
    table_offset, table_length = _segment(header, table_name, file_size)
    data_offset, data_length = _segment(header, "data", file_size)
    if slide_offset != HEADER_BYTES:
        raise UpgradeError(f"unexpected slide_table_offset: {slide_offset}")
    if table_offset != slide_offset + slide_length:
        raise UpgradeError(
            f"non-contiguous {table_name}: offset={table_offset} expected={slide_offset + slide_length}"
        )
    if data_offset != table_offset + table_length:
        raise UpgradeError(f"non-contiguous data: offset={data_offset} expected={table_offset + table_length}")
    if data_offset + data_length > file_size:
        raise UpgradeError("data segment exceeds file size")

    slide_table = _read_arrow_table(path, slide_offset, slide_length, "slide_table")
    index_table = _read_arrow_table(path, table_offset, table_length, table_name)
    if int(header.get("num_slides", -1)) != len(slide_table):
        raise UpgradeError(
            f"num_slides mismatch: header={header.get('num_slides')} table={len(slide_table)}"
        )
    if int(header.get("num_records", -1)) != len(index_table):
        raise UpgradeError(
            f"num_records mismatch: header={header.get('num_records')} table={len(index_table)}"
        )
    _validate_payload_contract(header, index_table)
    return index_table


def inspect_package(path: Path) -> Inspection:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise UpgradeError("path is not a regular file")
    stat = path.stat()
    fixed, magic, _, binary_version, header = _read_fixed_header(path)
    json_version = int(header.get("version", -1))
    if magic == V1_MAGIC and binary_version == V1_VERSION and json_version == V1_VERSION:
        if "index_table_offset" in header or "index_table_length" in header:
            raise UpgradeError("v1 header unexpectedly contains v2 index_table keys")
        table_name = "record_table"
        state = "v1"
    elif magic == V2_MAGIC and binary_version == V2_VERSION and json_version == V2_VERSION:
        if "record_table_offset" in header or "record_table_length" in header:
            raise UpgradeError("v2 header unexpectedly contains v1 record_table keys")
        table_name = "index_table"
        state = "v2"
    else:
        raise UpgradeError(
            f"inconsistent or unsupported version: magic={magic!r} binary={binary_version} json={json_version}"
        )
    table = _validate_layout_and_tables(path, header, table_name, stat.st_size)
    return Inspection(
        path=path,
        state=state,
        payload_type=str(header["payload_type"]),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        inode=stat.st_ino,
        device=stat.st_dev,
        fixed_header_sha256=hashlib.sha256(fixed).hexdigest(),
        header=header,
        table_columns=tuple(table.column_names),
    )


def discover_packages(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[tuple[int, int]] = set()
    for root in roots:
        root = root.resolve(strict=True)
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = []
            for base, directories, filenames in os.walk(root, followlinks=False):
                directories[:] = [name for name in directories if not (Path(base) / name).is_symlink()]
                candidates.extend(Path(base) / name for name in filenames if name.endswith(".iac"))
        else:
            raise UpgradeError(f"unsupported input root: {root}")
        for path in candidates:
            if path.is_symlink():
                raise UpgradeError(f"direct .iac symlinks are not accepted: {_path_token(path)}")
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            found.append(path)
    return sorted(found, key=lambda item: os.fsencode(item))


def preflight(paths: list[Path], workers: int, show_paths: bool) -> list[Inspection]:
    inspections: list[Inspection] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(inspect_package, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                inspections.append(future.result())
            except Exception as exc:
                errors.append(f"{_display_path(path, show_paths)} error={type(exc).__name__}: {exc}")
    if errors:
        for message in sorted(errors):
            print(f"preflight_error {message}", file=sys.stderr, flush=True)
        raise UpgradeError(f"preflight failed for {len(errors)} package(s); no files were changed")
    inspections.sort(key=lambda item: os.fsencode(item.path))
    return inspections


def build_v2_fixed_header(v1_header: dict) -> bytes:
    header = dict(v1_header)
    try:
        header["index_table_offset"] = header.pop("record_table_offset")
        header["index_table_length"] = header.pop("record_table_length")
    except KeyError as exc:
        raise UpgradeError(f"missing v1 table field: {exc.args[0]}") from exc
    header["version"] = V2_VERSION
    encoded = json.dumps(header, indent=2, sort_keys=True).encode("utf-8")
    if len(encoded) > HEADER_BYTES - 16:
        raise UpgradeError(f"upgraded header JSON is too large: {len(encoded)} bytes")
    fixed = bytearray(HEADER_BYTES)
    fixed[:8] = V2_MAGIC
    struct.pack_into("<I", fixed, 8, len(encoded))
    struct.pack_into("<I", fixed, 12, V2_VERSION)
    fixed[16 : 16 + len(encoded)] = encoded
    return bytes(fixed)


def _write_fixed_header(path: Path, fixed: bytes) -> None:
    if len(fixed) != HEADER_BYTES:
        raise UpgradeError(f"fixed header must be {HEADER_BYTES} bytes")
    descriptor = os.open(path, os.O_RDWR)
    try:
        if hasattr(os, "pwrite"):
            written = os.pwrite(descriptor, fixed, 0)
        else:
            os.lseek(descriptor, 0, os.SEEK_SET)
            written = os.write(descriptor, fixed)
        if written != len(fixed):
            raise UpgradeError(f"short header write: expected={len(fixed)} got={written}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clone_file(source: Path, target: Path) -> None:
    if sys.platform != "darwin":
        raise UpgradeError("copy-on-write clone transaction requires macOS/APFS")
    libc = ctypes.CDLL(None, use_errno=True)
    clonefile = libc.clonefile
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clonefile.restype = ctypes.c_int
    result = clonefile(os.fsencode(source), os.fsencode(target), 0)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(source), str(target))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _verify_v2_clone(path: Path, expected_size: int, payload_type: str) -> None:
    from iatro.iac import PackReader, read_tables

    if path.stat().st_size != expected_size:
        raise UpgradeError(f"clone size changed: expected={expected_size} got={path.stat().st_size}")
    header, _, index_table = read_tables(path)
    if header.get("version") != V2_VERSION:
        raise UpgradeError(f"v2 reader returned version={header.get('version')}")
    if str(header.get("payload_type")) != payload_type:
        raise UpgradeError("payload_type changed during migration")
    if int(header["num_records"]) != len(index_table):
        raise UpgradeError("v2 reader record count mismatch")

    reader = PackReader(path)
    try:
        count = int(header["num_records"])
        if count == 0:
            return
        sample_rows = sorted({0, count - 1})
        if "offset" in index_table.column_names:
            for row in sample_rows:
                payload = reader.read_payload(row)
                if "crc32" in index_table.column_names:
                    expected_crc = int(index_table.column("crc32")[row].as_py())
                    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
                    if actual_crc != expected_crc:
                        raise UpgradeError(
                            f"payload CRC mismatch after migration: row={row} expected={expected_crc} got={actual_crc}"
                        )
        elif payload_type == "teacher_features":
            record_bytes = int(header["feature_record_bytes"])
            for row in sample_rows:
                payload = reader.read_data_span(row * record_bytes, record_bytes)
                if len(payload) != record_bytes:
                    raise UpgradeError(f"short fixed feature record after migration: row={row}")
    finally:
        reader.close()


def _assert_unchanged(inspection: Inspection) -> None:
    stat = inspection.path.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
        inspection.device,
        inspection.inode,
        inspection.size,
        inspection.mtime_ns,
    ):
        raise UpgradeError("source changed after preflight")
    fixed, _, _, _, _ = _read_fixed_header(inspection.path)
    if hashlib.sha256(fixed).hexdigest() != inspection.fixed_header_sha256:
        raise UpgradeError("source header changed after preflight")


def upgrade_one(inspection: Inspection, transaction: str) -> None:
    if inspection.state == "v2":
        return
    _assert_unchanged(inspection)
    upgraded_fixed = build_v2_fixed_header(inspection.header)

    if transaction == "in-place":
        old_fixed, _, _, _, _ = _read_fixed_header(inspection.path)
        try:
            _write_fixed_header(inspection.path, upgraded_fixed)
            _verify_v2_clone(inspection.path, inspection.size, inspection.payload_type)
        except BaseException:
            _write_fixed_header(inspection.path, old_fixed)
            raise
        return

    temp_path = inspection.path.with_name(
        f".{inspection.path.name}.iac-v2-upgrade-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        _clone_file(inspection.path, temp_path)
        _write_fixed_header(temp_path, upgraded_fixed)
        _verify_v2_clone(temp_path, inspection.size, inspection.payload_type)
        os.replace(temp_path, inspection.path)
        _fsync_directory(inspection.path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _format_bytes(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def _print_summary(inspections: list[Inspection]) -> None:
    states = Counter(item.state for item in inspections)
    payloads = Counter((item.state, item.payload_type) for item in inspections)
    total_bytes = sum(item.size for item in inspections)
    print(
        f"preflight_ok files={len(inspections)} bytes={total_bytes} size='{_format_bytes(total_bytes)}' "
        f"v1={states['v1']} v2={states['v2']}",
        flush=True,
    )
    for (state, payload_type), count in sorted(payloads.items()):
        print(f"preflight_group state={state} payload_type={payload_type} files={count}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="IAC file(s) or directory roots; symlink directories are not followed.")
    parser.add_argument("--apply", action="store_true", help="Perform conversion after a successful full preflight. Default is dry-run.")
    parser.add_argument(
        "--transaction",
        choices=("clone", "in-place"),
        default="clone",
        help="clone uses APFS copy-on-write plus atomic replace (default); in-place is intended only for tests.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers used only for read-only preflight.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print aggregate conversion progress every N files.")
    parser.add_argument("--show-paths", action="store_true", help="Include source paths in errors; disabled by default for privacy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_packages(args.roots)
    if not paths:
        raise SystemExit("no .iac packages found")
    print(f"discovered files={len(paths)}", flush=True)
    inspections = preflight(paths, args.workers, args.show_paths)
    _print_summary(inspections)
    if not args.apply:
        print("dry_run_complete no_files_changed=true", flush=True)
        return

    pending = [item for item in inspections if item.state == "v1"]
    print(f"conversion_start pending={len(pending)} transaction={args.transaction}", flush=True)
    completed = 0
    converted_bytes = 0
    for inspection in pending:
        try:
            upgrade_one(inspection, args.transaction)
        except Exception as exc:
            print(
                f"conversion_error completed={completed} {_display_path(inspection.path, args.show_paths)} "
                f"error={type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        completed += 1
        converted_bytes += inspection.size
        if completed == 1 or completed % max(1, args.progress_every) == 0 or completed == len(pending):
            print(
                f"conversion_progress completed={completed} total={len(pending)} "
                f"logical_bytes={converted_bytes} logical_size='{_format_bytes(converted_bytes)}'",
                flush=True,
            )
    print(f"conversion_complete converted={completed} already_v2={len(inspections) - len(pending)}", flush=True)


if __name__ == "__main__":
    main()
