from __future__ import annotations

import argparse
import zlib

import numpy as np

from .iatrocache import read_payload, read_tables
from .feature_cache import _decompress_feature_matrix
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
    if header.get("feature_layout") != "matrix":
        raise ValueError(f"unsupported teacher feature layout: {header.get('feature_layout')}")
    feature_dim = int(header.get("feature_dim", 0))
    if feature_dim <= 0:
        raise ValueError(f"invalid feature_dim: {header.get('feature_dim')}")
    try:
        dtype = np.dtype(header["dtype"])
    except Exception as exc:
        raise ValueError(f"invalid feature dtype: {header.get('dtype')}") from exc
    compression = str(header.get("compression", "none")).lower()
    matrix_offset = int(header.get("matrix_offset", 0))
    matrix_length = int(header["matrix_length"])
    if matrix_offset < 0 or matrix_length < 0 or matrix_offset + matrix_length > int(header["data_length"]):
        raise ValueError(f"feature matrix span outside data segment: offset={matrix_offset} length={matrix_length}")
    payload = read_payload(package_path, header, matrix_offset, matrix_length)
    expected_crc = int(header["matrix_crc32"])
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"feature matrix crc32 mismatch: expected={expected_crc} actual={actual_crc}")
    raw = _decompress_feature_matrix(payload, compression)
    expected_length = len(record_table) * feature_dim * dtype.itemsize
    if len(raw) != expected_length:
        raise ValueError(f"invalid feature matrix byte length: expected={expected_length} got={len(raw)}")
    if int(header.get("matrix_uncompressed_length", expected_length)) != expected_length:
        raise ValueError(
            f"matrix_uncompressed_length mismatch: header={header.get('matrix_uncompressed_length')} "
            f"expected={expected_length}"
        )
    matrix = np.frombuffer(raw, dtype=dtype)
    if matrix.shape != (len(record_table) * feature_dim,):
        raise ValueError(f"invalid feature matrix flat shape: {matrix.shape}")
    matrix.reshape((len(record_table), feature_dim))
    if header.get("matrix_shape") != [len(record_table), feature_dim]:
        raise ValueError(f"matrix_shape mismatch: header={header.get('matrix_shape')} expected={[len(record_table), feature_dim]}")
    return len(record_table) if max_payload == 0 else min(max_payload, len(record_table))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an IatroCache package.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--max-decode", type=int, default=8)
    parser.add_argument("--max-crc", type=int, default=0, help="Maximum records to CRC-check; 0 checks all records.")
    args = parser.parse_args()
    metadata = read_package_metadata(args.package)
    header, slide_table, record_table = read_tables(args.package)
    _validate_common(header, slide_table, record_table)
    payload_type = metadata.get("payload_type")
    if payload_type == "image_tiles":
        crc_limit = _validate_record_payload_spans(args.package, header, record_table, args.max_crc)
        decoded = _validate_image_tiles(args.package, header, record_table, args.max_decode)
        print(
            f"package_valid type=image_tiles records={metadata['num_records']} "
            f"codec={metadata['codec']} decoded={decoded} crc_checked={crc_limit}"
        )
    elif payload_type == "teacher_features":
        checked = _validate_teacher_features(args.package, header, record_table, args.max_decode)
        print(
            f"package_valid type=teacher_features records={metadata['num_records']} "
            f"teacher={metadata['teacher']} dim={metadata['feature_dim']} "
            f"dtype={metadata['dtype']} compression={metadata['compression']} rows_checked={checked}"
        )
    else:
        raise ValueError(f"unsupported payload_type: {payload_type}")


if __name__ == "__main__":
    main()
