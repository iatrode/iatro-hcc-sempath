from __future__ import annotations

import argparse
import zlib

from .iatrocache import read_payload, read_tables
from .tile_package import iter_package_tiles, read_package_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an IatroCache tile package.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--max-decode", type=int, default=8)
    parser.add_argument("--max-crc", type=int, default=0, help="Maximum records to CRC-check; 0 checks all records.")
    args = parser.parse_args()
    metadata = read_package_metadata(args.package)
    header, slide_table, record_table = read_tables(args.package)
    if metadata["num_records"] != len(record_table):
        raise ValueError(f"num_records mismatch: header={metadata['num_records']} table={len(record_table)}")
    slide_indices = set(slide_table.column("slide_idx").to_pylist())
    for row in range(len(record_table)):
        slide_idx = record_table.column("slide_idx")[row].as_py()
        if slide_idx not in slide_indices:
            raise ValueError(f"invalid slide_idx at row {row}: {slide_idx}")
        offset = record_table.column("offset")[row].as_py()
        length = record_table.column("length")[row].as_py()
        if offset < 0 or length < 0 or offset + length > header["data_length"]:
            raise ValueError(f"payload span outside data segment at row {row}: offset={offset} length={length}")
    crc_limit = len(record_table) if args.max_crc == 0 else min(args.max_crc, len(record_table))
    for row in range(crc_limit):
        offset = record_table.column("offset")[row].as_py()
        length = record_table.column("length")[row].as_py()
        expected = record_table.column("crc32")[row].as_py()
        payload = read_payload(args.package, header, offset, length)
        actual = zlib.crc32(payload) & 0xFFFFFFFF
        if actual != expected:
            raise ValueError(f"crc32 mismatch at row {row}: expected={expected} actual={actual}")
    decoded = 0
    for _, image in iter_package_tiles(args.package):
        if decoded < args.max_decode:
            expected_size = (metadata["tile_width"], metadata["tile_height"])
            if image.size != expected_size:
                raise ValueError(f"expected {expected_size} tile, got {image.size}")
        decoded += 1
    print(
        f"package_valid tiles={decoded} declared={metadata['num_records']} "
        f"codec={metadata['codec']} crc_checked={crc_limit}"
    )


if __name__ == "__main__":
    main()
