from __future__ import annotations

import argparse

from hcc_sempath.feature_cache import (
    build_teacher_feature_package_from_manifest,
    build_teacher_feature_package_from_tile_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack teacher feature .npy cache files into an IatroCache file.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--tile-package", default="")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-name", default="")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.tile_package):
        raise ValueError("provide exactly one of --manifest or --tile-package")
    if args.tile_package:
        build_teacher_feature_package_from_tile_package(
            args.tile_package,
            args.feature_dir,
            args.output,
            teacher_name=args.teacher_name,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )
    else:
        build_teacher_feature_package_from_manifest(
            args.manifest,
            args.feature_dir,
            args.output,
            teacher_name=args.teacher_name,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )
    print(f"feature_package_ok output={args.output}")


if __name__ == "__main__":
    main()
