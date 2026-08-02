"""Canonical IatroCache names used by the pathology pipeline."""

from __future__ import annotations

from pathlib import Path


PATHOLOGY_TILE_SUFFIX = ".tile.path.iac"
PATHOLOGY_PREDICTION_SUFFIX = ".pred.path.iac"
PATHOLOGY_FEATURE_SUFFIX = ".feat.path.iac"
PATHOLOGY_MERGED_FEATURE_SUFFIX = PATHOLOGY_FEATURE_SUFFIX
LEGACY_PATHOLOGY_TILE_SUFFIX = ".tiles.iac"
LEGACY_PATHOLOGY_FEATURE_SUFFIX = ".features.iac"
LEGACY_PATHOLOGY_MERGED_FEATURE_SUFFIX = ".merged.features.iac"


def pathology_tile_stem(path: str | Path) -> str:
    """Return the logical source name for canonical or legacy tile packages."""

    name = Path(path).name
    for suffix in (PATHOLOGY_TILE_SUFFIX, LEGACY_PATHOLOGY_TILE_SUFFIX):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if stem:
                return stem
    raise ValueError(
        "pathology tile package must end with "
        f"{PATHOLOGY_TILE_SUFFIX} (legacy {LEGACY_PATHOLOGY_TILE_SUFFIX} is read-only compatible): {path}"
    )


def is_pathology_tile_name(path: str | Path) -> bool:
    name = Path(path).name
    return name.endswith((PATHOLOGY_TILE_SUFFIX, LEGACY_PATHOLOGY_TILE_SUFFIX))


def pathology_tile_path(root: str | Path, name: str) -> Path:
    return Path(root) / f"{name}{PATHOLOGY_TILE_SUFFIX}"


def pathology_prediction_path(root: str | Path, name: str) -> Path:
    return Path(root) / f"{name}{PATHOLOGY_PREDICTION_SUFFIX}"


def pathology_feature_stem(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (PATHOLOGY_FEATURE_SUFFIX, LEGACY_PATHOLOGY_FEATURE_SUFFIX):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            if stem:
                return stem
    raise ValueError(f"pathology feature package has an unsupported name: {path}")


def pathology_feature_candidates(root: str | Path, stem: str) -> list[Path]:
    """Find canonical merged/single-teacher features, then legacy packages."""

    root = Path(root)
    candidates = [root / f"{stem}{PATHOLOGY_FEATURE_SUFFIX}"]
    candidates.extend(sorted(root.glob(f"{stem}.*{PATHOLOGY_FEATURE_SUFFIX}")))
    candidates.extend(sorted(root.glob(f"{stem}.*{LEGACY_PATHOLOGY_FEATURE_SUFFIX}")))
    return list(dict.fromkeys(path for path in candidates if path.is_file()))
