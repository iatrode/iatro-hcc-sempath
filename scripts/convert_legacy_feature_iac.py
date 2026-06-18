from __future__ import annotations

import argparse
import lzma
import zlib
from pathlib import Path

import imagecodecs
import numpy as np

from hcc_sempath.io.iatro_iac import build_pack_data_segment, read_payload, read_tables
from hcc_sempath.io.validate_package import validate_package


def _decompress(payload: bytes, compression: str) -> bytes:
    if compression == "none":
        return payload
    if compression == "zstd":
        return imagecodecs.zstd_decode(payload)
    if compression == "zlib":
        return zlib.decompress(payload)
    if compression == "lzma":
        return lzma.decompress(payload)
    raise ValueError(f"unsupported legacy feature compression: {compression}")


def convert_legacy_feature_iac(input_path: str | Path, output_path: str | Path, *, overwrite: bool = False) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    header, slide_table, record_table = read_tables(input_path)
    if header.get("payload_type") != "teacher_features":
        raise ValueError(f"not a teacher feature package: {input_path}")
    if header.get("feature_layout") != "matrix":
        raise ValueError(f"not a legacy compressed matrix feature package: {input_path}")

    dtype = np.dtype(header["dtype"])
    feature_dim = int(header["feature_dim"])
    num_records = int(header["num_records"])
    matrix_offset = int(header.get("matrix_offset", 0))
    matrix_length = int(header["matrix_length"])
    payload = read_payload(input_path, header, matrix_offset, matrix_length)
    expected_crc = int(header["matrix_crc32"])
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"legacy matrix crc32 mismatch: expected={expected_crc} actual={actual_crc}")
    raw = _decompress(payload, str(header.get("compression", "none")).lower())
    expected_length = num_records * feature_dim * dtype.itemsize
    if len(raw) != expected_length:
        raise ValueError(f"legacy matrix byte length mismatch: expected={expected_length} got={len(raw)}")

    new_header = {
        key: value
        for key, value in header.items()
        if not (
            key.startswith("matrix_")
            or key in {"feature_layout", "compression", "compression_level", "format", "version", "header_bytes"}
            or key.endswith("_offset")
            or key.endswith("_length")
            or key in {"data_offset", "data_length", "num_slides", "num_records"}
        )
    }
    new_header["payload_type"] = "teacher_features"
    new_header["teacher"] = str(header["teacher"])
    new_header["feature_dim"] = feature_dim
    new_header["dtype"] = dtype.name
    new_header["feature_record_bytes"] = feature_dim * dtype.itemsize
    build_pack_data_segment(output_path, new_header, slide_table, record_table, raw, overwrite=overwrite)


def _is_legacy_feature_package(path: Path) -> bool:
    try:
        header, _, _ = read_tables(path)
    except Exception:
        return False
    return header.get("payload_type") == "teacher_features" and header.get("feature_layout") == "matrix"


def _output_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    relative = input_path.relative_to(input_root)
    return output_root / relative


def convert_legacy_feature_iac_tree(
    input_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
    validate: bool = True,
) -> list[Path]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    if not input_root.is_dir():
        raise ValueError(f"input is not a directory: {input_root}")
    converted: list[Path] = []
    for input_path in sorted(input_root.rglob("*.iac")):
        if not _is_legacy_feature_package(input_path):
            continue
        output_path = _output_path_for(input_path, input_root, output_root)
        convert_legacy_feature_iac(input_path, output_path, overwrite=overwrite)
        if validate:
            validate_package(output_path, max_decode=8)
        converted.append(output_path)
        print(f"legacy_feature_iac_converted input={input_path} output={output_path}", flush=True)
    if not converted:
        raise FileNotFoundError(f"no legacy matrix teacher feature .iac packages found under {input_root}")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy compressed teacher feature IAC package(s) to fixed-record raw feature IAC.")
    parser.add_argument("--input", required=True, help="Legacy .features.iac file or directory recursively scanned for legacy feature packages.")
    parser.add_argument("--output", required=True, help="Output .iac file for file input, or output directory for directory input.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.is_dir():
        if output_path.suffix == ".iac":
            raise ValueError("--output must be a directory when --input is a directory")
        converted = convert_legacy_feature_iac_tree(
            input_path,
            output_path,
            overwrite=args.overwrite,
            validate=args.validate,
        )
        print(f"legacy_feature_iac_convert_summary input={input_path} output={output_path} converted={len(converted)}")
        return
    convert_legacy_feature_iac(input_path, output_path, overwrite=args.overwrite)
    if args.validate:
        validate_package(output_path, max_decode=8)
    print(f"legacy_feature_iac_converted input={input_path} output={output_path}")


if __name__ == "__main__":
    main()
