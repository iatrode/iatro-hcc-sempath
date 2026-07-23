from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..modeling.prototypes import PrototypeRegistry


DEFAULT_L1_CLASSES = (
    "HCC-tumor",
    "Background-liver",
    "Inflammatory-stromal",
    "Degenerative-material",
)


@dataclass(frozen=True)
class PrototypeLabel:
    tile_id: str
    level1: int
    source_split: str


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "adjudicated"}


def load_prototype_labels(
    manifest_path: str | Path | None,
    prototypes: PrototypeRegistry | list[str] | tuple[str, ...] | None,
    *,
    allowed_source_splits: set[str] | None = None,
    require_adjudicated: bool = True,
) -> dict[str, PrototypeLabel]:
    if manifest_path is None or prototypes is None:
        return {}
    if isinstance(prototypes, PrototypeRegistry):
        primary_names = [prototypes.names[idx] for idx in prototypes.primary_indices]
    else:
        primary_names = [str(name) for name in prototypes]
    primary_index = {name: idx for idx, name in enumerate(primary_names)}
    labels: dict[str, PrototypeLabel] = {}
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tile_id",
            "level1_label",
            "source_split",
            "adjudicated",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"prototype supervision manifest missing columns: {sorted(missing)}")
        for row in reader:
            source_split = str(row["source_split"]).strip()
            if allowed_source_splits is not None and source_split not in allowed_source_splits:
                continue
            if require_adjudicated and not _truthy(row.get("adjudicated")):
                continue
            level1_name = str(row["level1_label"]).strip()
            if level1_name not in primary_index:
                raise ValueError(f"unknown level1 prototype label: {level1_name}")
            tile_id = str(row["tile_id"]).strip()
            if tile_id in labels:
                raise ValueError(f"duplicate prototype supervision tile_id: {tile_id}")
            labels[tile_id] = PrototypeLabel(
                tile_id=tile_id,
                level1=int(primary_index[level1_name]),
                source_split=source_split,
            )
    return labels
