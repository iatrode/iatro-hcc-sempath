from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..modeling.prototypes import PrototypeRegistry


DEFAULT_CLASSIFICATION_CLASSES = (
    "HCC-tumor-well-differentiated",
    "HCC-tumor-moderately-differentiated",
    "HCC-tumor-poorly-differentiated",
    "Background-liver",
    "Inflammatory-stromal",
    "Degenerative-material",
)


@dataclass(frozen=True)
class PrototypeLabel:
    tile_id: str
    classification: int
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
        classification_names = list(prototypes.names)
    else:
        classification_names = [str(name) for name in prototypes]
    classification_index = {name: idx for idx, name in enumerate(classification_names)}
    labels: dict[str, PrototypeLabel] = {}
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tile_id",
            "classification_label",
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
            classification_name = str(row["classification_label"]).strip()
            if classification_name not in classification_index:
                raise ValueError(f"unknown classification prototype label: {classification_name}")
            tile_id = str(row["tile_id"]).strip()
            if tile_id in labels:
                raise ValueError(f"duplicate prototype supervision tile_id: {tile_id}")
            labels[tile_id] = PrototypeLabel(
                tile_id=tile_id,
                classification=int(classification_index[classification_name]),
                source_split=source_split,
            )
    return labels
