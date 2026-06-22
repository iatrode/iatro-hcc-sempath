from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class RoiTileTarget:
    target: torch.Tensor  # [K, H, W], float32
    valid: torch.Tensor  # [K, H, W], bool; unmarked partial annotations remain ignore
    consistency: torch.Tensor  # [K], bool


def empty_roi_target(attribute_count: int, grid_size: tuple[int, int]) -> RoiTileTarget:
    h, w = grid_size
    return RoiTileTarget(
        target=torch.zeros((attribute_count, h, w), dtype=torch.float32),
        valid=torch.zeros((attribute_count, h, w), dtype=torch.bool),
        consistency=torch.zeros((attribute_count,), dtype=torch.bool),
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        return payload["annotations"]
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), dict):
        flattened = []
        for tile in payload["annotations"].values():
            for roi in tile.get("roi", []):
                flattened.append(
                    {
                        "tile_id": tile["tile_id"],
                        "split": roi.get("split", tile.get("split", tile.get("source_split", "train"))),
                        **roi,
                    }
                )
        return flattened
    raise ValueError("ROI manifest must be a JSON list, JSONL records, or {'annotations': [...]}")


def _xy(value: Any, image_size: tuple[int, int], normalized: bool) -> tuple[float, float]:
    if isinstance(value, dict):
        x, y = float(value["x"]), float(value["y"])
    else:
        x, y = float(value[0]), float(value[1])
    image_h, image_w = image_size
    if normalized:
        x *= image_w
        y *= image_h
    return x, y


def _distance_to_segment(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _inside_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            cross_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < cross_x:
                inside = not inside
        j = i
    return inside


def geometry_token_mask(
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Rasterize point, brush/polyline, circle, rectangle, or polygon at patch centers."""
    image_h, image_w = image_size
    grid_h, grid_w = grid_size
    kind = str(geometry.get("type", "")).lower()
    normalized = str(geometry.get("coordinate_space", "pixel")).lower() == "normalized"
    patch_radius = 0.5 * min(image_w / grid_w, image_h / grid_h)
    mask = torch.zeros((grid_h, grid_w), dtype=torch.bool)

    point_center: tuple[float, float] | None = None
    if kind in {"point", "circle"}:
        center = _xy(geometry.get("point", geometry.get("center", geometry)), image_size, normalized)
        radius = float(geometry.get("radius", patch_radius))
        if normalized:
            radius *= min(image_w, image_h)
        predicate = lambda x, y: math.hypot(x - center[0], y - center[1]) <= max(radius, patch_radius * 0.5)
        if kind == "point":
            # A point must always mark the patch it lands in. With the DINOv2-S/14
            # grid a patch spans 14 px; a sub-patch radius would otherwise rasterize
            # to zero tokens whenever the click misses a patch center, silently
            # discarding the annotation.
            point_center = center
    elif kind in {"brush", "polyline"}:
        points = [_xy(item, image_size, normalized) for item in geometry.get("points", [])]
        if not points:
            raise ValueError("brush geometry requires at least one point")
        radius = float(geometry.get("radius", geometry.get("width", patch_radius * 2) / 2))
        if normalized:
            radius *= min(image_w, image_h)
        segments = list(zip(points, points[1:])) or [(points[0], points[0])]
        predicate = lambda x, y: min(_distance_to_segment(x, y, a, b) for a, b in segments) <= radius
    elif kind in {"polygon", "freehand"}:
        points = [_xy(item, image_size, normalized) for item in geometry.get("points", [])]
        if len(points) < 3:
            raise ValueError("polygon geometry requires at least three points")
        predicate = lambda x, y: _inside_polygon(x, y, points)
    elif kind in {"rectangle", "box"}:
        start = geometry["start"] if "start" in geometry else [geometry["x0"], geometry["y0"]]
        end = geometry["end"] if "end" in geometry else [geometry["x1"], geometry["y1"]]
        x0, y0 = _xy(start, image_size, normalized)
        x1, y1 = _xy(end, image_size, normalized)
        predicate = lambda x, y: min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1)
    else:
        raise ValueError(f"unsupported ROI geometry type: {kind!r}")

    for row in range(grid_h):
        y = (row + 0.5) * image_h / grid_h
        for col in range(grid_w):
            x = (col + 0.5) * image_w / grid_w
            mask[row, col] = bool(predicate(x, y))
    if point_center is not None and not mask.any():
        col = min(grid_w - 1, max(0, int(point_center[0] * grid_w / image_w)))
        row = min(grid_h - 1, max(0, int(point_center[1] * grid_h / image_h)))
        mask[row, col] = True
    return mask


def build_roi_targets(
    manifest_path: str | Path | None,
    *,
    attribute_names: list[str],
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    allowed_splits: set[str] | None = None,
) -> dict[str, RoiTileTarget]:
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    records = _load_records(path)
    positions = {name: idx for idx, name in enumerate(attribute_names)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        split = str(record.get("split", "train"))
        if allowed_splits is not None and split not in allowed_splits:
            continue
        tile_id = str(record["tile_id"])
        attribute = str(record["attribute"])
        if attribute not in positions:
            raise ValueError(f"unknown ROI attribute {attribute!r} for tile {tile_id}")
        grouped.setdefault(tile_id, []).append(record)

    result: dict[str, RoiTileTarget] = {}
    for tile_id, tile_records in grouped.items():
        tile = empty_roi_target(len(attribute_names), grid_size)
        target, valid = tile.target.clone(), tile.valid.clone()
        # Completeness defines reviewed background first; positive geometries
        # are then overlaid regardless of manifest record order.
        for record in tile_records:
            if bool(record.get("review_complete", False)):
                idx = positions[str(record["attribute"])]
                valid[idx].fill_(True)
                target[idx].zero_()
        for record in tile_records:
            idx = positions[str(record["attribute"])]
            geometry = record.get("geometry")
            if geometry is None:
                if not bool(record.get("review_complete", False)):
                    raise ValueError(f"ROI record requires geometry or review_complete=true: tile={tile_id}")
                continue
            region = geometry_token_mask(geometry, image_size=image_size, grid_size=grid_size)
            state = str(record.get("state", "positive")).lower()
            if state not in {"positive", "negative"}:
                raise ValueError(f"ROI state must be positive or negative, got {state!r}")
            valid[idx] |= region
            target[idx][region] = 1.0 if state == "positive" else 0.0
        consistency = valid.flatten(1).any(dim=1)
        result[tile_id] = RoiTileTarget(target=target, valid=valid, consistency=consistency)
    return result


def roi_payload(
    tile_id: str,
    roi_targets: dict[str, RoiTileTarget] | None,
    *,
    attribute_count: int,
    grid_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    item = (roi_targets or {}).get(tile_id)
    if item is None:
        item = empty_roi_target(attribute_count, grid_size)
    return {
        "roi_target": item.target,
        "roi_valid": item.valid,
        "roi_consistency": item.consistency,
    }
