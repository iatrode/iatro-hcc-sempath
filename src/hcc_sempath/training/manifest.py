from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml

from iatro.iac.adapters.tiles import read_package_manifest


TILE_SUFFIX = ".tiles.iac"
FEATURE_SUFFIX_TEMPLATE = ".{teacher}.features.iac"


def package_stem(path: str | Path, tile_suffix: str = TILE_SUFFIX) -> str:
    name = Path(path).name
    if not name.endswith(tile_suffix):
        raise ValueError(f"tile package does not end with {tile_suffix}: {path}")
    return name[: -len(tile_suffix)]


def tile_package_path(tile_root: str | Path, stem: str, tile_suffix: str = TILE_SUFFIX) -> Path:
    return Path(tile_root) / f"{stem}{tile_suffix}"


def feature_package_path(
    feature_root: str | Path,
    teacher: str,
    stem: str,
    feature_suffix_template: str = FEATURE_SUFFIX_TEMPLATE,
) -> Path:
    return Path(feature_root) / teacher / f"{stem}{feature_suffix_template.format(teacher=teacher)}"


def _feature_package_path_for_tile(
    manifest: dict,
    tile_path: Path,
    teacher: str,
    feature_root: str | Path | None,
    feature_suffix_template: str = FEATURE_SUFFIX_TEMPLATE,
) -> Path:
    stem = package_stem(tile_path, str(manifest.get("tile_suffix", TILE_SUFFIX)))
    feature_roots = manifest.get("feature_roots")
    if isinstance(feature_roots, dict):
        teacher_root = feature_roots.get(teacher)
        if teacher_root is None:
            raise ValueError(f"training manifest feature_roots missing teacher={teacher}")
        feature_dir = Path(teacher_root) / tile_path.parent.name
        matches = sorted(feature_dir.glob(f"{stem}.*.features.iac"))
        if not matches:
            raise FileNotFoundError(
                f"missing feature package teacher={teacher} tile={tile_path} expected={feature_dir}/{stem}.*.features.iac"
            )
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous feature packages teacher={teacher} tile={tile_path} matches={matches}")
        return matches[0]
    if feature_root is None:
        raise ValueError("feature_root is required when manifest.feature_roots is not configured")
    return feature_package_path(feature_root, teacher, stem, feature_suffix_template)


def _discover_tile_packages(root: str | Path, tile_suffix: str = TILE_SUFFIX) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.glob(f"*{tile_suffix}") if path.is_file())


def _package_tile_count(path: Path) -> int:
    return len(read_package_manifest(path))


def _count_split_tiles(tile_root: str | Path, stems: list[str], tile_suffix: str) -> int:
    return sum(_package_tile_count(tile_package_path(tile_root, stem, tile_suffix)) for stem in stems)


def _split_key_for_package(path: Path, split_key: str) -> str:
    if split_key == "stem":
        return package_stem(path)
    if split_key not in {"patient_id", "slide_id"}:
        raise ValueError(f"unsupported split_key: {split_key}")
    records = read_package_manifest(path)
    if not records:
        raise ValueError(f"empty tile package: {path}")
    values = {getattr(record, split_key) for record in records}
    if len(values) != 1:
        sample = ", ".join(sorted(values)[:3])
        raise ValueError(f"package has multiple {split_key} values: {path} sample={sample}")
    return next(iter(values))


def _split_development_packages(
    packages: list[Path],
    split_key: str,
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in packages:
        groups[_split_key_for_package(path, split_key)].append(path)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    val_count = round(len(keys) * val_frac)
    if val_frac > 0 and keys:
        val_count = max(1, val_count)
    val_keys = set(keys[:val_count])
    train_paths = []
    val_paths = []
    for key in keys:
        target = val_paths if key in val_keys else train_paths
        target.extend(groups[key])
    return [package_stem(path) for path in sorted(train_paths)], [package_stem(path) for path in sorted(val_paths)]


def _split_public_packages(packages: list[Path], public_exval_n: int, seed: int) -> tuple[list[str], list[str]]:
    stems = [package_stem(path) for path in packages]
    rng = random.Random(seed)
    shuffled = sorted(stems)
    rng.shuffle(shuffled)
    if public_exval_n < 0:
        raise ValueError("public_exval_n must be non-negative")
    if public_exval_n > len(shuffled):
        raise ValueError(f"public_exval_n={public_exval_n} exceeds public package count={len(shuffled)}")
    exval = sorted(shuffled[:public_exval_n])
    train = sorted(shuffled[public_exval_n:])
    return train, exval


def build_training_manifest(
    dev_sources: dict[str, str | Path],
    public_source: tuple[str, str | Path] | None,
    public_exval_n: int,
    val_frac: float,
    split_key: str,
    seed: int,
    tile_suffix: str = TILE_SUFFIX,
) -> dict:
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be in [0, 1)")
    datasets = {}
    splits = {"train": {}, "val": {}, "exval": {}}
    summary = {"datasets": {}, "splits": {"train": {}, "val": {}, "exval": {}}}
    for name, root in dev_sources.items():
        packages = _discover_tile_packages(root, tile_suffix=tile_suffix)
        if not packages:
            raise ValueError(f"no tile packages found for dev source {name}: {root}")
        train_stems, val_stems = _split_development_packages(packages, split_key, val_frac, seed)
        datasets[name] = {"role": "development", "tile_root": str(root)}
        splits["train"][name] = train_stems
        splits["val"][name] = val_stems
        summary["datasets"][name] = {
            "role": "development",
            "package_count": len(packages),
            "tile_count": sum(_package_tile_count(path) for path in packages),
        }
        summary["splits"]["train"][name] = {
            "package_count": len(train_stems),
            "tile_count": _count_split_tiles(root, train_stems, tile_suffix),
        }
        summary["splits"]["val"][name] = {
            "package_count": len(val_stems),
            "tile_count": _count_split_tiles(root, val_stems, tile_suffix),
        }
    if public_source is not None:
        public_name, public_root = public_source
        packages = _discover_tile_packages(public_root, tile_suffix=tile_suffix)
        if not packages:
            raise ValueError(f"no tile packages found for public source {public_name}: {public_root}")
        train_stems, exval_stems = _split_public_packages(packages, public_exval_n, seed)
        datasets[public_name] = {"role": "public", "tile_root": str(public_root)}
        splits["train"][public_name] = train_stems
        splits["exval"][f"{public_name}_heldout"] = {"source": public_name, "stems": exval_stems}
        summary["datasets"][public_name] = {
            "role": "public",
            "package_count": len(packages),
            "tile_count": sum(_package_tile_count(path) for path in packages),
        }
        summary["splits"]["train"][public_name] = {
            "package_count": len(train_stems),
            "tile_count": _count_split_tiles(public_root, train_stems, tile_suffix),
        }
        summary["splits"]["exval"][f"{public_name}_heldout"] = {
            "source": public_name,
            "package_count": len(exval_stems),
            "tile_count": _count_split_tiles(public_root, exval_stems, tile_suffix),
        }
    return {
        "version": 1,
        "tile_suffix": tile_suffix,
        "split_key": split_key,
        "seed": seed,
        "datasets": datasets,
        "splits": splits,
        "summary": summary,
    }


def load_training_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError(f"unsupported training manifest: {path}")
    return manifest


def _auto_split_payload(manifest: dict, split: str) -> dict:
    auto_split = manifest.get("auto_split")
    if not isinstance(auto_split, dict) or not bool(auto_split.get("enabled", False)):
        return {}
    tile_suffix = str(manifest.get("tile_suffix", TILE_SUFFIX))
    split_key = str(auto_split.get("split_key", manifest.get("split_key", "stem")))
    development_val_frac = float(auto_split.get("development_val_fraction", auto_split.get("val_fraction", 0.15)))
    external_val_frac = float(auto_split.get("external_val_fraction", 0.50))
    seed = int(auto_split.get("seed", manifest.get("seed", 13)))
    payload = {"train": {}, "val": {}, "exval": {}}
    for name, dataset in manifest["datasets"].items():
        role = str(dataset.get("role", "development"))
        root = dataset["tile_root"]
        packages = _discover_tile_packages(root, tile_suffix)
        if role == "development":
            train_stems, val_stems = _split_development_packages(packages, split_key, development_val_frac, seed)
            payload["train"][name] = train_stems
            payload["val"][name] = val_stems
        elif role == "validation_external":
            exval_stems, val_stems = _split_development_packages(packages, split_key, external_val_frac, seed)
            payload["val"][name] = val_stems
            payload["exval"][f"{name}_heldout"] = {"source": name, "stems": exval_stems}
        elif role == "external":
            payload["exval"][f"{name}_heldout"] = {"source": name, "stems": [package_stem(path, tile_suffix) for path in packages]}
    return payload.get(split, {})


def manifest_tile_packages(manifest: dict, split: str) -> list[Path]:
    tile_suffix = str(manifest.get("tile_suffix", TILE_SUFFIX))
    datasets = manifest["datasets"]
    packages: list[Path] = []
    split_payload = manifest["splits"].get(split, {})
    if not split_payload:
        split_payload = _auto_split_payload(manifest, split)
    if split == "exval":
        for payload in split_payload.values():
            source = payload["source"]
            root = datasets[source]["tile_root"]
            packages.extend(tile_package_path(root, stem, tile_suffix) for stem in payload.get("stems", []))
        return packages
    for source, stems in split_payload.items():
        root = datasets[source]["tile_root"]
        packages.extend(tile_package_path(root, stem, tile_suffix) for stem in stems)
    return packages


def manifest_teacher_feature_packages(
    manifest: dict,
    split: str,
    teachers: list[str],
    feature_root: str | Path,
    feature_suffix_template: str = FEATURE_SUFFIX_TEMPLATE,
) -> dict[str, list[Path]]:
    packages_by_teacher = {teacher: [] for teacher in teachers}
    for tile_path in manifest_tile_packages(manifest, split):
        for teacher in teachers:
            packages_by_teacher[teacher].append(
                _feature_package_path_for_tile(manifest, tile_path, teacher, feature_root, feature_suffix_template)
            )
    return packages_by_teacher


def manifest_teacher_feature_packages_for_tiles(
    manifest: dict,
    tile_paths: list[str | Path],
    teachers: list[str],
    feature_root: str | Path | None = None,
    feature_suffix_template: str = FEATURE_SUFFIX_TEMPLATE,
) -> dict[str, list[Path]]:
    packages_by_teacher = {teacher: [] for teacher in teachers}
    for tile_path in [Path(path) for path in tile_paths]:
        for teacher in teachers:
            packages_by_teacher[teacher].append(
                _feature_package_path_for_tile(manifest, tile_path, teacher, feature_root, feature_suffix_template)
            )
    return packages_by_teacher


def validate_manifest_artifacts(
    manifest: dict,
    splits: list[str],
    teachers: list[str] | None = None,
    feature_root: str | Path | None = None,
    feature_suffix_template: str = FEATURE_SUFFIX_TEMPLATE,
) -> dict:
    missing_tiles = []
    for split in splits:
        for path in manifest_tile_packages(manifest, split):
            if not path.exists():
                missing_tiles.append(str(path))
    missing_features = []
    if teachers:
        if feature_root is None:
            raise ValueError("feature_root is required when teachers are provided")
        for split in splits:
            packages_by_teacher = manifest_teacher_feature_packages(
                manifest=manifest,
                split=split,
                teachers=teachers,
                feature_root=feature_root,
                feature_suffix_template=feature_suffix_template,
            )
            for teacher, paths in packages_by_teacher.items():
                for path in paths:
                    if not path.exists():
                        missing_features.append({"teacher": teacher, "path": str(path), "split": split})
    result = {
        "checked_splits": splits,
        "missing_tile_packages": missing_tiles,
        "missing_feature_packages": missing_features,
    }
    if missing_tiles or missing_features:
        missing_tile_sample = ", ".join(missing_tiles[:3])
        missing_feature_sample = ", ".join(item["path"] for item in missing_features[:3])
        raise FileNotFoundError(
            "manifest artifact check failed: "
            f"missing_tiles={len(missing_tiles)} sample=[{missing_tile_sample}] "
            f"missing_features={len(missing_features)} sample=[{missing_feature_sample}]"
        )
    return result


def _parse_source(values: list[str]) -> dict[str, str]:
    sources = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source must be name=path: {value}")
        name, path = value.split("=", 1)
        sources[name] = path
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training manifest from per-WSI tile IAC packages.")
    parser.add_argument("--dev-source", action="append", default=[], help="Development source as name=tile_root.")
    parser.add_argument("--public-source", default="", help="Public source as name=tile_root.")
    parser.add_argument("--public-exval-n", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--split-key", default="patient_id", choices=["patient_id", "slide_id", "stem"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--teacher", action="append", default=[], help="Teacher name to verify under --feature-root.")
    parser.add_argument("--feature-root", default="", help="Root containing <teacher>/<stem>.<teacher>.features.iac.")
    parser.add_argument("--feature-suffix-template", default=FEATURE_SUFFIX_TEMPLATE)
    parser.add_argument("--check-artifacts", action="store_true", help="Fail if tile or teacher feature packages are missing.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dev_sources = _parse_source(args.dev_source)
    public_source = None
    if args.public_source:
        public_items = _parse_source([args.public_source])
        if len(public_items) != 1:
            raise ValueError("--public-source must contain exactly one name=path")
        public_source = next(iter(public_items.items()))
    if not dev_sources and public_source is None:
        raise ValueError("at least one --dev-source or --public-source is required")
    manifest = build_training_manifest(
        dev_sources=dev_sources,
        public_source=public_source,
        public_exval_n=args.public_exval_n,
        val_frac=args.val_frac,
        split_key=args.split_key,
        seed=args.seed,
    )
    if args.check_artifacts:
        validate_manifest_artifacts(
            manifest=manifest,
            splits=["train", "val", "exval"],
            teachers=[str(teacher) for teacher in args.teacher] or None,
            feature_root=args.feature_root or None,
            feature_suffix_template=args.feature_suffix_template,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"train_manifest_ok output={output}")


if __name__ == "__main__":
    main()
