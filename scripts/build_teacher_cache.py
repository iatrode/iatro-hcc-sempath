from __future__ import annotations

import argparse
from pathlib import Path

from hcc_sempath.teachers import TimmTeacherEncoder, cache_teacher_features, cache_teacher_features_from_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline teacher-cache builder for remote/high-performance machines.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--tile-package", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="hf_hub:bioptimus/H-optimus-1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.tile_package):
        raise ValueError("provide exactly one of --manifest or --tile-package")
    model = TimmTeacherEncoder(args.model_name, pretrained=args.pretrained)
    if args.tile_package:
        cache_teacher_features_from_package(
            model=model,
            package_path=args.tile_package,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
        )
        source = f"tile_package={args.tile_package}"
    else:
        cache_teacher_features(
            model=model,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=args.device,
        )
        source = f"manifest={args.manifest}"
    print(f"cache_ok {source} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
