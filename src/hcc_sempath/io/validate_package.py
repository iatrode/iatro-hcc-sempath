from __future__ import annotations

import argparse
from pathlib import Path
import zlib

import numpy as np
from tqdm import tqdm

from .iatrocache import read_payload, read_tables
from .tile_package import decode_jxl, read_package_metadata


def _require_columns(table, columns: tuple[str, ...], table_name: str) -> None:
    missing = [column for column in columns if column not in table.column_names]
    if missing:
        raise ValueError(f"{table_name} missing columns: {missing}")


def _validate_unique_tile_ids(record_table) -> None:
    if "tile_id" not in record_table.column_names:
        raise ValueError("record table missing required tile_id column")
    values = record_table.column("tile_id").to_pylist()
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        sample = ", ".join(str(value) for value in duplicates[:3])
        raise ValueError(f"duplicate tile_id values: count={len(duplicates)} sample={sample}")


def _sample_rows(num_records: int, limit: int) -> range:
    if limit == 0:
        return range(num_records)
    return range(min(limit, num_records))


def _validate_common(header: dict, slide_table, record_table) -> None:
    if header["num_records"] != len(record_table):
        raise ValueError(f"num_records mismatch: header={header['num_records']} table={len(record_table)}")
    if header["num_slides"] != len(slide_table):
        raise ValueError(f"num_slides mismatch: header={header['num_slides']} table={len(slide_table)}")
    _require_columns(slide_table, ("slide_idx", "slide_id", "patient_id"), "slide table")
    _require_columns(
        record_table,
        ("slide_idx", "tile_x", "tile_y", "tile_id", "flags"),
        "record table",
    )
    _validate_unique_tile_ids(record_table)

    slide_indices = set(slide_table.column("slide_idx").to_pylist())
    for row in range(len(record_table)):
        slide_idx = record_table.column("slide_idx")[row].as_py()
        if slide_idx not in slide_indices:
            raise ValueError(f"invalid slide_idx at row {row}: {slide_idx}")


def _validate_record_payload_spans(package_path: str, header: dict, record_table, max_crc: int) -> int:
    _require_columns(record_table, ("offset", "length", "crc32"), "record table")
    for row in range(len(record_table)):
        offset = record_table.column("offset")[row].as_py()
        length = record_table.column("length")[row].as_py()
        if offset < 0 or length < 0 or offset + length > header["data_length"]:
            raise ValueError(f"payload span outside data segment at row {row}: offset={offset} length={length}")

    crc_limit = len(record_table) if max_crc == 0 else min(max_crc, len(record_table))
    for row in range(crc_limit):
        offset = record_table.column("offset")[row].as_py()
        length = record_table.column("length")[row].as_py()
        expected = record_table.column("crc32")[row].as_py()
        payload = read_payload(package_path, header, offset, length)
        actual = zlib.crc32(payload) & 0xFFFFFFFF
        if actual != expected:
            raise ValueError(f"crc32 mismatch at row {row}: expected={expected} actual={actual}")
    return crc_limit


def _validate_image_tiles(package_path: str, header: dict, record_table, max_decode: int) -> int:
    if header.get("codec") != "jxl":
        raise ValueError(f"unsupported image tile codec: {header.get('codec')}")
    expected_size = (int(header["tile_width"]), int(header["tile_height"]))
    decoded = 0
    for row in _sample_rows(len(record_table), max_decode):
        offset = record_table.column("offset")[row].as_py()
        length = record_table.column("length")[row].as_py()
        payload = read_payload(package_path, header, offset, length)
        image = decode_jxl(payload)
        if image.size != expected_size:
            raise ValueError(f"expected {expected_size} tile at row {row}, got {image.size}")
        decoded += 1
    return decoded


def _validate_teacher_features(package_path: str, header: dict, record_table, max_payload: int) -> int:
    teacher = header.get("teacher")
    if not isinstance(teacher, str) or not teacher:
        raise ValueError("teacher_features header requires non-empty teacher")
    feature_dim = int(header.get("feature_dim", 0))
    if feature_dim <= 0:
        raise ValueError(f"invalid feature_dim: {header.get('feature_dim')}")
    try:
        dtype = np.dtype(header["dtype"])
    except Exception as exc:
        raise ValueError(f"invalid feature dtype: {header.get('dtype')}") from exc
    record_bytes = int(header.get("feature_record_bytes", 0))
    expected_record_bytes = feature_dim * dtype.itemsize
    if record_bytes != expected_record_bytes:
        raise ValueError(f"invalid feature_record_bytes: expected={expected_record_bytes} got={record_bytes}")
    expected_length = len(record_table) * record_bytes
    if int(header["data_length"]) != expected_length:
        raise ValueError(f"invalid feature data length: expected={expected_length} got={header['data_length']}")
    if max_payload == 0:
        rows = range(len(record_table))
    else:
        rows = range(min(max_payload, len(record_table)))
    for row in rows:
        payload = read_payload(package_path, header, row * record_bytes, record_bytes)
        feature = np.frombuffer(payload, dtype=dtype)
        if feature.shape != (feature_dim,):
            raise ValueError(f"invalid feature shape at row {row}: {feature.shape}")
    return len(record_table) if max_payload == 0 else min(max_payload, len(record_table))


def _discover_packages(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if root.is_dir():
        packages = sorted(candidate for candidate in root.rglob("*.iac") if candidate.is_file())
        if not packages:
            raise FileNotFoundError(f"no .iac packages found under {root}")
        return packages
    raise FileNotFoundError(f"package path does not exist: {root}")


def validate_package(package_path: str | Path, max_decode: int = 8, max_crc: int = 0) -> dict:
    metadata = read_package_metadata(package_path)
    header, slide_table, record_table = read_tables(package_path)
    _validate_common(header, slide_table, record_table)
    payload_type = metadata.get("payload_type")
    if payload_type == "image_tiles":
        crc_limit = _validate_record_payload_spans(str(package_path), header, record_table, max_crc)
        decoded = _validate_image_tiles(str(package_path), header, record_table, max_decode)
        return {
            "type": "image_tiles",
            "records": int(metadata["num_records"]),
            "codec": metadata["codec"],
            "decoded": decoded,
            "crc_checked": crc_limit,
        }
    if payload_type == "teacher_features":
        checked = _validate_teacher_features(str(package_path), header, record_table, max_decode)
        return {
            "type": "teacher_features",
            "records": int(metadata["num_records"]),
            "teacher": metadata["teacher"],
            "dim": metadata["feature_dim"],
            "dtype": metadata["dtype"],
            "rows_checked": checked,
        }
    raise ValueError(f"unsupported payload_type: {payload_type}")


def _format_valid_message(result: dict) -> str:
    if result["type"] == "image_tiles":
        return (
            f"package_valid type=image_tiles records={result['records']} "
            f"codec={result['codec']} decoded={result['decoded']} crc_checked={result['crc_checked']}"
        )
    return (
        f"package_valid type=teacher_features records={result['records']} "
        f"teacher={result['teacher']} dim={result['dim']} "
        f"dtype={result['dtype']} rows_checked={result['rows_checked']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one IatroCache package or a directory of .iac packages.")
    parser.add_argument("--input", required=True, help="Input .iac package file or directory scanned recursively for .iac packages.")
    parser.add_argument(
        "--max-decode",
        type=int,
        default=8,
        help="Maximum image-tile payloads to JXL-decode per image_tiles package; 0 decodes all tiles.",
    )
    parser.add_argument("--max-crc", type=int, default=0, help="Maximum image-tile records to CRC-check; 0 checks all records.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop directory validation on the first failed package.")
    args = parser.parse_args()
    packages = _discover_packages(args.input)
    if len(packages) == 1:
        result = validate_package(packages[0], max_decode=args.max_decode, max_crc=args.max_crc)
        print(_format_valid_message(result))
        return

    ok = 0
    failures: list[tuple[Path, str]] = []
    for package_path in tqdm(packages, desc="validating packages", unit="pkg"):
        try:
            result = validate_package(package_path, max_decode=args.max_decode, max_crc=args.max_crc)
            ok += 1
            tqdm.write(f"package_valid path={package_path} " + _format_valid_message(result))
        except Exception as exc:
            failures.append((package_path, str(exc)))
            tqdm.write(f"package_invalid path={package_path} error={exc}")
            if args.fail_fast:
                break

    print(f"validation_summary total={len(packages)} ok={ok} failed={len(failures)}")
    if failures:
        sample = "; ".join(f"{path}: {error}" for path, error in failures[:5])
        raise SystemExit(f"validation_failed count={len(failures)} sample={sample}")


if __name__ == "__main__":
    main()
