from __future__ import annotations

import argparse

from hcc_sempath.tile_package import build_tile_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack tile manifest and JXL-compressed tiles into one HCCSPK file.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile-root", default=None)
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_tile_package(
        manifest_path=args.manifest,
        output_path=args.output,
        tile_root=args.tile_root,
        lossless=args.lossless,
        distance=None if args.lossless else args.distance,
        effort=args.effort,
        overwrite=args.overwrite,
    )
    print(f"package_ok output={args.output}")


if __name__ == "__main__":
    main()

