from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml

from ..io.tile_package import read_package_manifest


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


def _discover_tile_packages(root: str | Path, tile_suffix: str = TILE_SUFFIX) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.glob(f"*{tile_suffix}") if path.is_file())


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
    for name, root in dev_sources.items():
        packages = _discover_tile_packages(root, tile_suffix=tile_suffix)
        if not packages:
            raise ValueError(f"no tile packages found for dev source {name}: {root}")
        train_stems, val_stems = _split_development_packages(packages, split_key, val_frac, seed)
        datasets[name] = {"role": "development", "tile_root": str(root)}
        splits["train"][name] = train_stems
        splits["val"][name] = val_stems
    if public_source is not None:
        public_name, public_root = public_source
        packages = _discover_tile_packages(public_root, tile_suffix=tile_suffix)
        if not packages:
            raise ValueError(f"no tile packages found for public source {public_name}: {public_root}")
        train_stems, exval_stems = _split_public_packages(packages, public_exval_n, seed)
        datasets[public_name] = {"role": "public", "tile_root": str(public_root)}
        splits["train"][public_name] = train_stems
        splits["exval"][f"{public_name}_heldout"] = {"source": public_name, "stems": exval_stems}
    return {
        "version": 1,
        "tile_suffix": tile_suffix,
        "split_key": split_key,
        "seed": seed,
        "datasets": datasets,
        "splits": splits,
    }


def load_training_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError(f"unsupported training manifest: {path}")
    return manifest


def manifest_tile_packages(manifest: dict, split: str) -> list[Path]:
    tile_suffix = str(manifest.get("tile_suffix", TILE_SUFFIX))
    datasets = manifest["datasets"]
    packages: list[Path] = []
    split_payload = manifest["splits"].get(split, {})
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
    tile_suffix = str(manifest.get("tile_suffix", TILE_SUFFIX))
    packages_by_teacher = {teacher: [] for teacher in teachers}
    for tile_path in manifest_tile_packages(manifest, split):
        stem = package_stem(tile_path, tile_suffix)
        for teacher in teachers:
            packages_by_teacher[teacher].append(
                feature_package_path(feature_root, teacher, stem, feature_suffix_template)
            )
    return packages_by_teacher


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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"train_manifest_ok output={output}")


if __name__ == "__main__":
    main()
