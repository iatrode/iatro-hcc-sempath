"""Prepare the merged, row-shuffled teacher cache consumed by training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from iatro.iac import read_header
from hcc_sempath.iac_naming import (
    is_pathology_tile_name,
    pathology_feature_candidates,
    pathology_tile_stem,
)
from hcc_sempath.training.feature_pack_merge import (
    MERGED_FEATURE_SUFFIX,
    _build_merged_package,
    _delete_source_packages,
    _validate_merged_against_sources,
    _validate_merged_package,
)
from hcc_sempath.training.feature_pack_shuffle import _prepare_tile_feature_pair


DEFAULT_TEACHER_ROOTS = {
    "gigapath": "gigapath",
    "h_optimus_1": "h1",
    "uni2_h": "uni2",
    "virchow2": "virchow2",
}


@dataclass(frozen=True)
class PreparedPackage:
    tile_path: Path
    merged_path: Path
    status: str


def _discover_tile_packages(tile_root: Path) -> list[Path]:
    return sorted(
        path
        for path in tile_root.glob("*/*.iac")
        if path.is_file() and is_pathology_tile_name(path)
    )


def _parse_teacher_roots(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_TEACHER_ROOTS)
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"teacher mapping must be name=subdir: {value}")
        name, subdir = value.split("=", 1)
        name = name.strip()
        subdir = subdir.strip()
        if not name or not subdir:
            raise ValueError(f"teacher mapping must be name=subdir: {value}")
        result[name] = subdir
    if not result:
        raise ValueError("at least one teacher mapping is required")
    return result


def _merged_path(feature_root: Path, dataset: str, stem: str, teacher_roots: dict[str, str]) -> Path:
    return feature_root / dataset / f"{stem}{MERGED_FEATURE_SUFFIX}"


def _source_feature_path(feature_root: Path, dataset: str, stem: str, teacher: str, teacher_roots: dict[str, str]) -> Path:
    feature_dir = feature_root / teacher_roots[teacher] / dataset
    matches = pathology_feature_candidates(feature_dir, stem)
    if not matches:
        raise FileNotFoundError(f"missing_feature teacher={teacher} expected={feature_dir}/{stem}.feat.path.iac")
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous_feature teacher={teacher} matches={matches}")
    return matches[0]


def _existing_source_feature_paths(
    feature_root: Path,
    dataset: str,
    stem: str,
    teacher_roots: dict[str, str],
) -> dict[str, Path]:
    paths = {}
    for teacher in teacher_roots:
        try:
            paths[teacher] = _source_feature_path(feature_root, dataset, stem, teacher, teacher_roots)
        except FileNotFoundError:
            continue
    return paths


def _expected_dims_from_sources(source_paths: dict[str, Path]) -> dict[str, int]:
    dims = {}
    for teacher, path in source_paths.items():
        header = read_header(path)
        if header.get("payload_type") != "teacher_features":
            raise ValueError(f"not_teacher_features teacher={teacher} path={path}")
        dims[teacher] = int(header["feature_dim"])
    return dims


def _expected_dims_from_merged(path: Path) -> dict[str, int]:
    header = read_header(path)
    return {str(k): int(v) for k, v in header.get("teacher_dims", {}).items()}


def _prepare_one(
    tile_path: Path,
    feature_root: Path,
    teacher_roots: dict[str, str],
    seed: int,
    delete_sources: bool,
    dtype: str,
    *,
    validate_only: bool = False,
) -> PreparedPackage:
    dataset = tile_path.parent.name
    stem = pathology_tile_stem(tile_path)
    merged_path = _merged_path(feature_root, dataset, stem, teacher_roots)
    teacher_names = list(teacher_roots)
    if merged_path.exists():
        expected_dims = _expected_dims_from_merged(merged_path)
        _validate_merged_package(
            tile_path=tile_path,
            merged_path=merged_path,
            teacher_names=teacher_names,
            expected_dims=expected_dims,
        )
        if delete_sources:
            source_paths = _existing_source_feature_paths(feature_root, dataset, stem, teacher_roots)
            _delete_source_packages(source_paths, keep_path=merged_path)
        status = "existing_merged"
    elif validate_only:
        status = "missing_merged"
    else:
        source_paths = {
            teacher: _source_feature_path(feature_root, dataset, stem, teacher, teacher_roots)
            for teacher in teacher_names
        }
        expected_dims = _expected_dims_from_sources(source_paths)
        _build_merged_package(
            tile_path=tile_path,
            source_paths=source_paths,
            merged_path=merged_path,
            expected_dims=expected_dims,
            dtype=dtype,
        )
        _validate_merged_package(
            tile_path=tile_path,
            merged_path=merged_path,
            teacher_names=teacher_names,
            expected_dims=expected_dims,
        )
        _validate_merged_against_sources(
            source_paths=source_paths,
            merged_path=merged_path,
            expected_dims=expected_dims,
        )
        if delete_sources:
            _delete_source_packages(source_paths, keep_path=merged_path)
        status = "merged"
    if not validate_only and status != "missing_merged":
        _prepare_tile_feature_pair(tile_path=tile_path, feature_path=merged_path, seed=seed)
    return PreparedPackage(tile_path=tile_path, merged_path=merged_path, status=status)


def _progress(iterable, total: int):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        for item in iterable:
            yield item
        return
    with tqdm(total=total, desc="prepare_iac", unit="pkg", dynamic_ncols=True) as bar:
        for item in iterable:
            yield item
            bar.update(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate merged teacher features and align tile/feature row order."
    )
    parser.add_argument(
        "--tile-root",
        required=True,
        help="Root containing <dataset>/<stem>.tile.path.iac packages.",
    )
    parser.add_argument(
        "--feature-root",
        required=True,
        help="Root containing merged <dataset>/<stem>.feat.path.iac packages.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dtype", default="source", help="Merged feature dtype: source preserves each source package dtype.")
    parser.add_argument("--delete-source", action="store_true", help="Delete source teacher feature packages after validation.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing merged package headers without writing data.")
    parser.add_argument(
        "--teacher",
        action="append",
        default=None,
        help="Teacher mapping as logical_name=feature_subdir. Defaults to the four HCC-SemPath teachers.",
    )
    args = parser.parse_args()

    tile_root = Path(args.tile_root)
    feature_root = Path(args.feature_root)
    teacher_roots = _parse_teacher_roots(args.teacher)
    tile_paths = _discover_tile_packages(tile_root)
    if not tile_paths:
        raise SystemExit(f"no tile packages found: {tile_root}")

    failures: list[tuple[Path, str]] = []
    statuses = {"existing_merged": 0, "merged": 0, "missing_merged": 0}
    workers = max(1, min(int(args.workers), len(tile_paths)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _prepare_one,
                tile_path,
                feature_root,
                teacher_roots,
                int(args.seed),
                bool(args.delete_source),
                str(args.dtype),
                validate_only=bool(args.validate_only),
            ): tile_path
            for tile_path in tile_paths
        }
        for future in _progress(as_completed(futures), len(futures)):
            tile_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((tile_path, str(exc)))
                continue
            statuses[result.status] += 1
            if result.status == "existing_merged":
                print(f"existing_merged tile={result.tile_path} feature={result.merged_path}", flush=True)

    if failures:
        print(f"prepare_iac_failed count={len(failures)}", flush=True)
        for tile_path, message in failures[:50]:
            print(f"error tile={tile_path} {message}", flush=True)
        raise SystemExit(1)
    print(
        "prepare_iac_done "
        f"packages={len(tile_paths)} existing_merged={statuses['existing_merged']} "
        f"merged={statuses['merged']} missing_merged={statuses['missing_merged']} "
        f"seed={int(args.seed)} validate_only={bool(args.validate_only)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
