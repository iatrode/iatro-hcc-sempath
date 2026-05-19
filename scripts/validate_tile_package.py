from __future__ import annotations

import argparse

from hcc_sempath.tile_package import iter_package_tiles, read_package_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an HCCSPK tile package.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--max-decode", type=int, default=8)
    args = parser.parse_args()
    metadata = read_package_metadata(args.package)
    decoded = 0
    for _, image in iter_package_tiles(args.package):
        if decoded < args.max_decode:
            if image.size != (224, 224):
                raise ValueError(f"expected 224x224 tile, got {image.size}")
        decoded += 1
    print(f"package_valid tiles={decoded} declared={metadata['tile_count']} codec={metadata['codec']}")


if __name__ == "__main__":
    main()

