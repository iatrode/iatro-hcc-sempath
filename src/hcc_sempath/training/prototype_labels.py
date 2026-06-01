from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from ..modeling.prototypes import PrototypeRegistry


@dataclass(frozen=True)
class PrototypeLabel:
    tile_id: str
    level1: int
    level2: torch.Tensor
    source_split: str


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "adjudicated"}


def _split_level2_labels(value: str | None) -> list[str]:
    if value is None:
        return []
    normalized = str(value).replace("|", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def load_prototype_labels(
    manifest_path: str | Path | None,
    prototypes: PrototypeRegistry | None,
    *,
    allowed_source_splits: set[str] | None = None,
    require_adjudicated: bool = True,
) -> dict[str, PrototypeLabel]:
    if manifest_path is None or prototypes is None:
        return {}
    primary_names = [prototypes.names[idx] for idx in prototypes.primary_indices]
    attribute_names = [prototypes.names[idx] for idx in prototypes.attribute_indices]
    primary_index = {name: idx for idx, name in enumerate(primary_names)}
    attribute_index = {name: idx for idx, name in enumerate(attribute_names)}
    labels: dict[str, PrototypeLabel] = {}
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tile_id",
            "level1_label",
            "level2_labels",
            "source_split",
            "expert_a",
            "expert_b",
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
            level2 = torch.zeros(len(attribute_names), dtype=torch.float32)
            for name in _split_level2_labels(row.get("level2_labels")):
                if name not in attribute_index:
                    raise ValueError(f"unknown level2 prototype label: {name}")
                level2[attribute_index[name]] = 1.0
            tile_id = str(row["tile_id"]).strip()
            if tile_id in labels:
                raise ValueError(f"duplicate prototype supervision tile_id: {tile_id}")
            labels[tile_id] = PrototypeLabel(
                tile_id=tile_id,
                level1=int(primary_index[level1_name]),
                level2=level2,
                source_split=source_split,
            )
    return labels
