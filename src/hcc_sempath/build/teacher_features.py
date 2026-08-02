"""Build the fixed four-teacher SemPath feature package."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from iatro.iac import read_header

from hcc_sempath.iac_naming import PATHOLOGY_FEATURE_SUFFIX, pathology_tile_stem
from hcc_sempath.teacher.cache import (
    TimmTeacherEncoder,
    _discover_tile_packages,
    _resolve_model_spec,
    _validate_feature_output,
    cache_teacher_features_from_packages,
)
from hcc_sempath.training.feature_pack_merge import (
    _build_merged_package,
    _validate_merged_against_sources,
    _validate_merged_package,
)


TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")


@dataclass(frozen=True)
class FeaturePlan:
    tile_package: Path
    relative_parent: Path
    stem: str
    output_path: Path


def _relative_parent(package: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path()
    try:
        return package.relative_to(input_path).parent
    except ValueError:
        return Path(package.parent.name)


def _plans(input_value: str, output_value: str) -> list[FeaturePlan]:
    input_path = Path(input_value).expanduser().resolve()
    output_path = Path(output_value).expanduser().resolve()
    packages = _discover_tile_packages(input_path)
    output_is_file = output_path.suffix == ".iac"
    if output_is_file and len(packages) != 1:
        raise ValueError("--output must be a directory when --input resolves to multiple tile packages")
    if output_is_file and not output_path.name.endswith(PATHOLOGY_FEATURE_SUFFIX):
        raise ValueError(f"feature output must end with {PATHOLOGY_FEATURE_SUFFIX}: {output_path}")
    plans = []
    for package in packages:
        stem = pathology_tile_stem(package)
        relative_parent = _relative_parent(package, input_path)
        final_path = output_path if output_is_file else output_path / relative_parent / f"{stem}{PATHOLOGY_FEATURE_SUFFIX}"
        plans.append(FeaturePlan(package, relative_parent, stem, final_path))
    by_output: dict[Path, list[Path]] = {}
    for plan in plans:
        by_output.setdefault(plan.output_path, []).append(plan.tile_package)
    duplicate = next((item for item in by_output.items() if len(item[1]) > 1), None)
    if duplicate is not None:
        raise ValueError(f"duplicate feature output {duplicate[0]} for {duplicate[1]}")
    return plans


def _staging_path(root: Path, teacher: str, plan: FeaturePlan) -> Path:
    return root / teacher / plan.relative_parent / f"{plan.stem}.{teacher}{PATHOLOGY_FEATURE_SUFFIX}"


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _valid_merged(plan: FeaturePlan) -> bool:
    try:
        header = read_header(plan.output_path)
        expected_dims = {str(name): int(value) for name, value in header["teacher_dims"].items()}
        _validate_merged_package(
            tile_path=plan.tile_package,
            merged_path=plan.output_path,
            teacher_names=list(TEACHERS),
            expected_dims=expected_dims,
        )
        return tuple(header.get("teachers", ())) == TEACHERS
    except Exception:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one merged four-teacher pathology feature package per tile package. "
            "Single-teacher packages are temporary and removed after verified merging."
        )
    )
    parser.add_argument("--input", required=True, help="Input .tile.path.iac file or directory.")
    parser.add_argument(
        "--output",
        required=True,
        help=f"Output {PATHOLOGY_FEATURE_SUFFIX} file for one input, or output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16", "auto"), default="bf16")
    parser.add_argument("--feature-dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size <= 0 or args.num_workers <= 0 or args.prefetch_factor <= 0:
        raise ValueError("batch size, workers, and prefetch factor must be positive")
    plans = _plans(args.input, args.output)
    invalid_existing = [
        plan.output_path
        for plan in plans
        if plan.output_path.exists() and not _valid_merged(plan)
    ]
    if invalid_existing and not args.overwrite:
        raise FileExistsError(
            "invalid feature output exists; inspect it or pass --overwrite: "
            f"{invalid_existing[0]}"
        )
    pending = [plan for plan in plans if args.overwrite or not _valid_merged(plan)]
    if not pending:
        print(f"teacher_features_ready packages={len(plans)} existing={len(plans)}", flush=True)
        return

    output_root = Path(args.output).expanduser().resolve()
    staging_parent = output_root.parent if output_root.suffix == ".iac" else output_root
    staging_root = staging_parent / ".hcc-sempath-feature-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources: dict[Path, dict[str, Path]] = {plan.tile_package: {} for plan in pending}
    try:
        for teacher in TEACHERS:
            model_spec = _resolve_model_spec(teacher)
            print(
                f"teacher_features_teacher_start teacher={teacher} model={model_spec['model_name']}",
                flush=True,
            )
            model = TimmTeacherEncoder(
                model_spec["model_name"],
                pretrained=args.pretrained,
                model_kwargs=model_spec["model_kwargs"],
                feature_mode=model_spec["feature_mode"],
            )
            try:
                for plan in pending:
                    stage_path = _staging_path(staging_root, teacher, plan)
                    stage_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_teacher_features_from_packages(
                        model=model,
                        package_paths=[plan.tile_package],
                        output=stage_path,
                        batch_size=args.batch_size,
                        device=args.device,
                        teacher_name=teacher,
                        num_workers=args.num_workers,
                        prefetch_factor=args.prefetch_factor,
                        overwrite=args.overwrite,
                        progress_manifest=stage_path.with_suffix(".progress.csv"),
                        precision=args.precision,
                        feature_dtype=args.feature_dtype,
                        compile_model=args.compile_model,
                        compile_mode=args.compile_mode,
                        validate_output=args.validate_output,
                    )
                    _validate_feature_output(
                        stage_path,
                        expected_teacher=teacher,
                        full=args.validate_output,
                    )
                    sources[plan.tile_package][teacher] = stage_path
            finally:
                del model
                _release_cuda()
            print(f"teacher_features_teacher_done teacher={teacher}", flush=True)

        for plan in pending:
            source_paths = sources[plan.tile_package]
            expected_dims = {
                teacher: int(read_header(source_paths[teacher])["feature_dim"])
                for teacher in TEACHERS
            }
            plan.output_path.parent.mkdir(parents=True, exist_ok=True)
            if plan.output_path.exists() and args.overwrite:
                plan.output_path.unlink()
            _build_merged_package(
                tile_path=plan.tile_package,
                source_paths=source_paths,
                merged_path=plan.output_path,
                expected_dims=expected_dims,
                dtype="source",
            )
            _validate_merged_package(
                tile_path=plan.tile_package,
                merged_path=plan.output_path,
                teacher_names=list(TEACHERS),
                expected_dims=expected_dims,
            )
            _validate_merged_against_sources(
                source_paths=source_paths,
                merged_path=plan.output_path,
                expected_dims=expected_dims,
            )
        records = []
        for plan in plans:
            header = read_header(plan.output_path)
            records.append(
                {
                    "tile_package": str(plan.tile_package),
                    "feature_package": str(plan.output_path),
                    "records": int(header["num_records"]),
                    "teacher_dims": {
                        str(name): int(value)
                        for name, value in header["teacher_dims"].items()
                    },
                }
            )
        manifest_root = output_root.parent if output_root.suffix == ".iac" else output_root
        manifest_root.mkdir(parents=True, exist_ok=True)
        (manifest_root / "feature_build_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "teachers": list(TEACHERS),
                    "packages": records,
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except BaseException:
        print(f"teacher_features_staging_retained path={staging_root}", flush=True)
        raise
    else:
        shutil.rmtree(staging_root)
    print(
        f"teacher_features_ready packages={len(plans)} built={len(pending)} "
        f"elapsed_seconds={time.monotonic() - started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
