from __future__ import annotations

import argparse
from hcc_sempath.io.qc import render_tile_package_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a visual QC contact sheet from an image-tile IatroCache package.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tiles", type=int, default=36)
    parser.add_argument("--thumb-size", type=int, default=160)
    args = parser.parse_args()
    render_tile_package_qc(
        args.package,
        args.output,
        max_tiles=args.max_tiles,
        thumb_size=args.thumb_size,
    )
    print(f"tile_qc_ok output={args.output}")


if __name__ == "__main__":
    main()
