#!/usr/bin/env python
"""Merge finalized train/val spatial truth without carrying UI state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PORTABLE_TILE_FIELDS = (
    "tile_id",
    "dataset",
    "slide",
    "x",
    "y",
    "row",
    "roi_reviewed",
    "roi_complete_all",
    "roi_count_complete",
    "roi_measurement_complete",
)
PORTABLE_ROI_FIELDS = (
    "attribute",
    "state",
    "geometry",
    "review_complete",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"spatial annotation must be an object: {path}")
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(
            f"spatial annotation has no annotations object: {path}"
        )
    return payload


def _schema(payload: dict[str, Any]) -> tuple[Any, Any]:
    return (
        payload.get("spatial_prototypes"),
        payload.get("label_definitions", {}).get("spatial"),
    )


def _finalized_tile(
    raw: dict[str, Any],
    *,
    key: str,
    split: str,
    spatial_names: set[str],
) -> dict[str, Any]:
    tile_id = str(raw.get("tile_id") or "").strip()
    if not tile_id:
        raise ValueError(
            f"annotation has no tile_id: key={key!r}"
        )
    if not bool(raw.get("roi_reviewed", False)):
        raise ValueError(f"spatial tile is not finalized: {tile_id}")
    package = str(raw.get("iac") or raw.get("iac_path") or "").strip()
    row = raw.get("row")
    if not package or row in (None, "") or int(row) < 0:
        raise ValueError(
            f"spatial tile has no fixed IAC package/row: {tile_id}"
        )
    roi = raw.get("roi")
    if not isinstance(roi, list) or not roi:
        raise ValueError(f"spatial tile has no ROI truth: {tile_id}")
    clean_roi: list[dict[str, Any]] = []
    for index, item in enumerate(roi):
        if not isinstance(item, dict):
            raise ValueError(
                f"ROI record must be an object: tile={tile_id} index={index}"
            )
        attribute = str(item.get("attribute") or "")
        state = str(item.get("state", "positive")).lower()
        if attribute not in spatial_names or state not in {
            "positive",
            "negative",
        }:
            raise ValueError(
                "invalid finalized ROI label: "
                f"tile={tile_id} attribute={attribute!r} state={state!r}"
            )
        if item.get("geometry") is None and not (
            state == "negative"
            and bool(item.get("review_complete", False))
        ):
            raise ValueError(
                "geometry-free ROI must be a reviewed complete negative: "
                f"tile={tile_id} attribute={attribute}"
            )
        clean = {
            field: item[field]
            for field in PORTABLE_ROI_FIELDS
            if field in item
        }
        clean["attribute"] = attribute
        clean["state"] = state
        clean_roi.append(clean)
    clean_tile = {
        field: raw[field]
        for field in PORTABLE_TILE_FIELDS
        if field in raw
    }
    clean_tile.update(
        {
            "tile_id": tile_id,
            "iac": package,
            "row": int(row),
            "split": split,
            "roi_reviewed": True,
            "roi": clean_roi,
        }
    )
    return clean_tile


def _merge_split(
    destination: dict[str, dict[str, Any]],
    paths: list[Path],
    split: str,
    *,
    expected_schema: tuple[Any, Any],
) -> int:
    added = 0
    spatial_names = {
        str(value)
        for value in expected_schema[0]
    }
    for path in paths:
        payload = _load(path)
        if _schema(payload) != expected_schema:
            raise ValueError(
                f"spatial annotation schema differs: {path}"
            )
        for key, raw in payload["annotations"].items():
            if not isinstance(raw, dict):
                raise ValueError(
                    f"spatial annotation row must be an object: {path} {key}"
                )
            row = _finalized_tile(
                raw,
                key=str(key),
                split=split,
                spatial_names=spatial_names,
            )
            tile_id = str(row["tile_id"])
            existing = destination.get(tile_id)
            if existing is not None:
                if str(existing["split"]) != split:
                    raise ValueError(
                        "train/validation spatial tile overlap: "
                        f"{tile_id}"
                    )
                comparable_existing = {
                    name: value
                    for name, value in existing.items()
                    if name != "split"
                }
                comparable_row = {
                    name: value
                    for name, value in row.items()
                    if name != "split"
                }
                if comparable_existing != comparable_row:
                    raise ValueError(
                        "conflicting duplicate spatial tile: "
                        f"{tile_id}"
                    )
                continue
            destination[tile_id] = row
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge immutable A0 train/val spatial annotations."
    )
    parser.add_argument("--train-spatial", nargs="+", required=True)
    parser.add_argument("--val-spatial", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_paths = [
        Path(value).resolve()
        for value in args.train_spatial
    ]
    val_paths = [
        Path(value).resolve()
        for value in args.val_spatial
    ]
    seed_payload = _load(train_paths[0])
    schema = _schema(seed_payload)
    if not schema[0] or not schema[1]:
        raise ValueError("spatial annotation schema is incomplete")

    annotations: dict[str, dict[str, Any]] = {}
    train_count = _merge_split(
        annotations,
        train_paths,
        "train",
        expected_schema=schema,
    )
    val_count = _merge_split(
        annotations,
        val_paths,
        "val",
        expected_schema=schema,
    )
    train_ids = {
        tile_id
        for tile_id, row in annotations.items()
        if row["split"] == "train"
    }
    val_ids = {
        tile_id
        for tile_id, row in annotations.items()
        if row["split"] == "val"
    }
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            "train/validation spatial tile overlap: "
            f"{next(iter(sorted(overlap)))}"
        )
    if not train_ids or not val_ids:
        raise ValueError("both train and val spatial splits are required")

    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing merged truth: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "version": int(seed_payload.get("version", 1)),
        "spatial_prototypes": schema[0],
        "label_definitions": {
            "spatial": schema[1],
        },
        "annotations": {
            tile_id: annotations[tile_id]
            for tile_id in sorted(annotations)
        },
        "source": {
            "builder": "merge_a0_spatial_annotations.py",
            "train_files": [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in train_paths
            ],
            "val_files": [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in val_paths
            ],
            "train_tiles": len(train_ids),
            "val_tiles": len(val_ids),
        },
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        "a0_spatial_merge_ok "
        f"train={train_count} val={val_count} output={output}"
    )


if __name__ == "__main__":
    main()
