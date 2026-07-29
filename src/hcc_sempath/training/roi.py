from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from hcc_sempath.spatial_schema import (
    CELL_INSTANCE_DENSITY,
    CONTINUOUS_AREA,
    DEFAULT_SPATIAL_COMPONENTS,
    PIGMENT_BURDEN,
    STRUCTURE_INSTANCE_AREA,
    spatial_component_specs,
)

POINT_GEOMETRIES = frozenset({"point"})
CIRCLE_GEOMETRIES = frozenset({"circle"})
BRUSH_GEOMETRIES = frozenset(
    {"brush", "polyline", "polygon", "freehand", "rectangle", "box"}
)


@dataclass(frozen=True)
class SpatialRoiTarget:
    """Weak spatial supervision for one tile.

    ``point_centers`` stores countable instance centres. Besides literal point
    marks, this includes the centre of a circle and one centroid per connected
    large-structure brush region. Multiple overlapping pen strokes therefore
    remain one biological instance. Continuous-area and pigment marks never
    enter this tensor.

    ``brush_bag_ids`` stores density-bag membership only for dense cell
    components. ``area_positive`` stores weak occupied-area support for
    continuous regions, pigment burden, and large discrete structures.
    ``instance_exclusion_support`` retains the extent of a circle or connected
    structure mark so the instance head cannot place duplicate centres inside
    one annotated object. It is not an occupied-area label.

    Explicit and implicit negatives remain separate so ordinary unmarked
    background can receive a much smaller weight than an annotator-confirmed
    absence.
    """

    point_centers: torch.Tensor  # [K, H, W], float32 click count per cell
    brush_bag_ids: torch.Tensor  # [K, H, W], int64; dense-cell bags only
    area_positive: torch.Tensor  # [K, H, W], bool; weak area/extent support
    explicit_negative: torch.Tensor  # [K, H, W], bool; strong negative
    implicit_negative: torch.Tensor  # [K, H, W], bool; weak background
    instance_exclusion_support: torch.Tensor | None = None

    @property
    def brush_mask(self) -> torch.Tensor:
        return self.brush_bag_ids > 0

    @property
    def supervised(self) -> torch.Tensor:
        """Per-component supervision presence, shape ``[K]``."""

        return (
            (self.point_centers > 0).flatten(1).any(dim=1)
            | (
                self.instance_exclusion_support.flatten(1).any(dim=1)
                if self.instance_exclusion_support is not None
                else torch.zeros(
                    self.point_centers.shape[0],
                    dtype=torch.bool,
                )
            )
            | self.brush_mask.flatten(1).any(dim=1)
            | self.area_positive.flatten(1).any(dim=1)
            | self.explicit_negative.flatten(1).any(dim=1)
            | self.implicit_negative.flatten(1).any(dim=1)
        )


@dataclass(frozen=True)
class SpatialValidationMetadata:
    """Per-component completeness and geometry provenance for calibration."""

    count_complete: torch.Tensor  # [K], bool
    measurement_complete: torch.Tensor  # [K], bool
    geometry_modes: tuple[tuple[str, ...], ...]  # [K], point/circle/brush/negative


@dataclass(frozen=True)
class _PreparedGeometrySupports:
    """Raster supports prepared before semantic target assembly.

    Worker threads only execute PIL/NumPy rasterization. All Torch resampling
    is deliberately batched on the calling thread so a Python thread pool
    cannot recursively enter Torch's intra-op thread pool.
    """

    grid: dict[int, torch.Tensor]
    pixel: dict[int, torch.Tensor]


def empty_spatial_roi_target(
    component_count: int,
    grid_size: tuple[int, int],
) -> SpatialRoiTarget:
    h, w = grid_size
    return SpatialRoiTarget(
        point_centers=torch.zeros((component_count, h, w), dtype=torch.float32),
        instance_exclusion_support=torch.zeros(
            (component_count, h, w),
            dtype=torch.bool,
        ),
        brush_bag_ids=torch.zeros((component_count, h, w), dtype=torch.long),
        area_positive=torch.zeros((component_count, h, w), dtype=torch.bool),
        explicit_negative=torch.zeros((component_count, h, w), dtype=torch.bool),
        implicit_negative=torch.zeros((component_count, h, w), dtype=torch.bool),
    )


def _load_records(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load ROI records and tile-level completion metadata."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        payload: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)

    records: list[dict[str, Any]] = []
    tile_metadata: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        for raw in payload:
            if not isinstance(raw, dict):
                raise ValueError("ROI record must be an object")
            record = dict(raw)
            tile_id = str(record["tile_id"])
            split = str(record.get("split", "train"))
            metadata = tile_metadata.setdefault(
                tile_id,
                {
                    "split": split,
                    "package_path": record.get("iac") or record.get("iac_path"),
                    "row": record.get("row"),
                    "complete_all": False,
                    "count_complete": None,
                    "measurement_complete": None,
                },
            )
            if str(metadata["split"]) != split:
                raise ValueError(f"inconsistent split for ROI tile {tile_id}")
            metadata["complete_all"] = bool(
                metadata["complete_all"]
                or record.get("tile_complete_all")
                or record.get("roi_complete_all")
            )
            metadata["count_complete"] = (
                record.get("tile_count_complete")
                or record.get("roi_count_complete")
                or metadata["count_complete"]
            )
            metadata["measurement_complete"] = (
                record.get("tile_measurement_complete")
                or record.get("roi_measurement_complete")
                or metadata["measurement_complete"]
            )
            records.append(record)
        return records, tile_metadata

    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if isinstance(annotations, list):
        return _load_records_from_annotation_list(annotations)
    if isinstance(annotations, dict):
        for tile in annotations.values():
            if not isinstance(tile, dict) or not tile.get("tile_id"):
                continue
            tile_id = str(tile["tile_id"])
            split = str(tile.get("split", tile.get("source_split", "train")))
            complete_all = bool(tile.get("roi_complete_all", False))
            tile_metadata[tile_id] = {
                "split": split,
                "package_path": tile.get("iac") or tile.get("iac_path"),
                "row": tile.get("row"),
                "complete_all": complete_all,
                "count_complete": tile.get("roi_count_complete"),
                "measurement_complete": tile.get(
                    "roi_measurement_complete"
                ),
            }
            for roi in tile.get("roi", []):
                if not isinstance(roi, dict):
                    raise ValueError(f"ROI record must be an object: tile={tile_id}")
                records.append(
                    {
                        "tile_id": tile_id,
                        "split": str(roi.get("split", split)),
                        "tile_complete_all": complete_all,
                        "tile_count_complete": tile.get(
                            "roi_count_complete"
                        ),
                        "tile_measurement_complete": tile.get(
                            "roi_measurement_complete"
                        ),
                        **roi,
                    }
                )
        return records, tile_metadata
    raise ValueError(
        "ROI manifest must be a JSON list, JSONL records, or "
        "{'annotations': [...]|{...}}"
    )


def _load_records_from_annotation_list(
    annotations: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    tile_metadata: dict[str, dict[str, Any]] = {}
    for raw in annotations:
        if not isinstance(raw, dict):
            raise ValueError("ROI annotation must be an object")
        record = dict(raw)
        tile_id = str(record["tile_id"])
        split = str(record.get("split", "train"))
        metadata = tile_metadata.setdefault(
            tile_id,
            {
                "split": split,
                "package_path": record.get("iac") or record.get("iac_path"),
                "row": record.get("row"),
                "complete_all": False,
                "count_complete": None,
                "measurement_complete": None,
            },
        )
        metadata["complete_all"] = bool(
            metadata["complete_all"]
            or record.get("tile_complete_all")
            or record.get("roi_complete_all")
        )
        metadata["count_complete"] = (
            record.get("tile_count_complete")
            or record.get("roi_count_complete")
            or metadata["count_complete"]
        )
        metadata["measurement_complete"] = (
            record.get("tile_measurement_complete")
            or record.get("roi_measurement_complete")
            or metadata["measurement_complete"]
        )
        records.append(record)
    return records, tile_metadata


def load_spatial_tile_locations(
    manifest_path: str | Path | None,
    *,
    allowed_splits: set[str] | None = None,
) -> dict[str, tuple[str, int]]:
    """Return fixed package/row provenance without opening any IAC package."""

    if manifest_path is None:
        return {}
    _, tile_metadata = _load_records(Path(manifest_path))
    locations: dict[str, tuple[str, int]] = {}
    for tile_id, metadata in tile_metadata.items():
        if (
            allowed_splits is not None
            and str(metadata["split"]) not in allowed_splits
        ):
            continue
        package_path = str(metadata.get("package_path") or "").strip()
        row = metadata.get("row")
        if not package_path or row in (None, ""):
            continue
        locations[str(tile_id)] = (package_path, int(row))
    return locations


def _completion_names(
    value: Any,
    component_names: list[str],
    *,
    field: str,
) -> set[str]:
    if value in (None, False):
        return set()
    if value is True:
        return set(component_names)
    if isinstance(value, str):
        selected = {value}
    elif isinstance(value, (list, tuple, set)):
        selected = {str(item) for item in value}
    elif isinstance(value, dict):
        selected = {
            str(name)
            for name, complete in value.items()
            if bool(complete)
        }
    else:
        raise ValueError(
            f"{field} must be bool, component name/list, or mapping"
        )
    unknown = sorted(selected.difference(component_names))
    if unknown:
        raise ValueError(f"{field} contains unknown components: {unknown}")
    return selected


def load_spatial_validation_metadata(
    manifest_path: str | Path,
    *,
    component_names: list[str],
    allowed_splits: set[str],
) -> dict[str, SpatialValidationMetadata]:
    """Load explicit calibration completeness without strengthening training labels."""

    records, tile_metadata = _load_records(Path(manifest_path))
    positions = {
        str(name): index
        for index, name in enumerate(component_names)
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        split = str(record.get("split", "train"))
        if split not in allowed_splits:
            continue
        tile_id = str(record["tile_id"])
        grouped.setdefault(tile_id, []).append(record)

    result: dict[str, SpatialValidationMetadata] = {}
    eligible_tile_ids = {
        tile_id
        for tile_id, metadata in tile_metadata.items()
        if str(metadata.get("split", "train")) in allowed_splits
    }
    eligible_tile_ids.update(grouped)
    for tile_id in sorted(eligible_tile_ids):
        metadata = tile_metadata.get(tile_id, {})
        count_names = _completion_names(
            metadata.get("count_complete"),
            component_names,
            field="roi_count_complete",
        )
        measurement_names = _completion_names(
            metadata.get("measurement_complete"),
            component_names,
            field="roi_measurement_complete",
        )
        geometry_modes = [set() for _ in component_names]
        for record in grouped.get(tile_id, []):
            component = str(record.get("attribute", ""))
            if component not in positions:
                raise ValueError(
                    f"unknown ROI component {component!r} for tile {tile_id}"
                )
            if bool(record.get("count_complete", False)):
                count_names.add(component)
            if bool(record.get("measurement_complete", False)):
                measurement_names.add(component)
            geometry = record.get("geometry")
            if geometry is None:
                if (
                    str(record.get("state", "positive")).lower()
                    == "negative"
                ):
                    geometry_modes[positions[component]].add("negative")
                continue
            kind = _geometry_kind(geometry)
            if kind in POINT_GEOMETRIES:
                normalized = "point"
            elif kind in CIRCLE_GEOMETRIES:
                normalized = "circle"
            else:
                normalized = "brush"
            geometry_modes[positions[component]].add(normalized)
        result[tile_id] = SpatialValidationMetadata(
            count_complete=torch.tensor(
                [name in count_names for name in component_names],
                dtype=torch.bool,
            ),
            measurement_complete=torch.tensor(
                [name in measurement_names for name in component_names],
                dtype=torch.bool,
            ),
            geometry_modes=tuple(
                tuple(sorted(values))
                for values in geometry_modes
            ),
        )
    return result


def spatial_component_names(manifest_path: str | Path) -> list[str]:
    """Read the active eleven-component spatial contract."""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    definitions = (
        payload.get("label_definitions", {}).get("spatial", [])
        if isinstance(payload, dict)
        else []
    )
    names = [
        str(item["id"])
        for item in definitions
        if isinstance(item, dict)
        and item.get("id")
        and bool(item.get("active", True))
    ]
    if not names and isinstance(payload, dict):
        names = [str(value) for value in payload.get("spatial_prototypes", [])]
    if not names:
        names = list(DEFAULT_SPATIAL_COMPONENTS)
    if tuple(names) != DEFAULT_SPATIAL_COMPONENTS:
        raise ValueError(
            "spatial component contract mismatch: "
            f"expected={list(DEFAULT_SPATIAL_COMPONENTS)} got={names}"
        )
    return names


def _xy(
    value: Any,
    image_size: tuple[int, int],
    normalized: bool,
) -> tuple[float, float]:
    if isinstance(value, dict):
        x, y = float(value["x"]), float(value["y"])
    else:
        x, y = float(value[0]), float(value[1])
    image_h, image_w = image_size
    if normalized:
        x *= image_w
        y *= image_h
    return x, y


def _geometry_kind(geometry: dict[str, Any]) -> str:
    kind = str(geometry.get("type", "")).lower()
    if kind not in POINT_GEOMETRIES | CIRCLE_GEOMETRIES | BRUSH_GEOMETRIES:
        raise ValueError(f"unsupported ROI geometry type: {kind!r}")
    return kind


def _radius_pixels(
    geometry: dict[str, Any],
    image_size: tuple[int, int],
    *,
    default: float,
) -> float:
    image_h, image_w = image_size
    normalized = (
        str(geometry.get("coordinate_space", "pixel")).lower() == "normalized"
    )
    radius = float(
        geometry.get("radius", geometry.get("width", default * 2.0) / 2.0)
    )
    if normalized:
        radius *= min(image_w, image_h)
    return max(0.5, radius)


def _geometry_pixel_support_numpy(
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Rasterize drawing support without interpreting it as occupied area."""

    image_h, image_w = image_size
    normalized = (
        str(geometry.get("coordinate_space", "pixel")).lower() == "normalized"
    )
    kind = _geometry_kind(geometry)
    if kind in POINT_GEOMETRIES:
        raise ValueError("point geometry does not define a brush bag")

    canvas = Image.new("L", (image_w, image_h), 0)
    draw = ImageDraw.Draw(canvas)
    if kind in {"brush", "polyline"}:
        points = [
            _xy(item, image_size, normalized)
            for item in geometry.get("points", [])
        ]
        if not points:
            raise ValueError("brush geometry requires at least one point")
        radius = _radius_pixels(geometry, image_size, default=7.0)
        width = max(1, int(round(radius * 2.0)))
        draw.line(points, fill=255, width=width, joint="curve")
        for x, y in (points[0], points[-1]):
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=255,
            )
    elif kind == "circle":
        center = _xy(
            geometry.get("point", geometry.get("center", geometry)),
            image_size,
            normalized,
        )
        radius = _radius_pixels(geometry, image_size, default=7.0)
        x, y = center
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=255,
        )
    elif kind in {"polygon", "freehand"}:
        points = [
            _xy(item, image_size, normalized)
            for item in geometry.get("points", [])
        ]
        if len(points) < 3:
            raise ValueError("polygon geometry requires at least three points")
        draw.polygon(points, fill=255)
    elif kind in {"rectangle", "box"}:
        start = (
            geometry["start"]
            if "start" in geometry
            else [geometry["x0"], geometry["y0"]]
        )
        end = (
            geometry["end"]
            if "end" in geometry
            else [geometry["x1"], geometry["y1"]]
        )
        draw.rectangle(
            (
                *_xy(start, image_size, normalized),
                *_xy(end, image_size, normalized),
            ),
            fill=255,
        )
    else:  # pragma: no cover - guarded by _geometry_kind
        raise ValueError(f"unsupported ROI geometry type: {kind!r}")
    return np.asarray(canvas, dtype=np.uint8) > 0


def _geometry_pixel_support(
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    return torch.from_numpy(
        _geometry_pixel_support_numpy(
            geometry,
            image_size=image_size,
        )
    )


def _prepare_geometry_supports(
    records: list[dict[str, Any]],
    *,
    positions: dict[str, int],
    specs: list,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> _PreparedGeometrySupports:
    """Rasterize in parallel, then reduce every support in batched Torch calls."""

    geometries: list[dict[str, Any]] = []
    retain_pixel: set[int] = set()
    seen: set[int] = set()
    for record in records:
        geometry = record.get("geometry")
        if not isinstance(geometry, dict) or _geometry_kind(geometry) == "point":
            continue
        key = id(geometry)
        if key not in seen:
            seen.add(key)
            geometries.append(geometry)
        component = str(record["attribute"])
        spec = specs[positions[component]]
        if (
            spec.mode == STRUCTURE_INSTANCE_AREA
            and _geometry_kind(geometry) in BRUSH_GEOMETRIES
        ):
            retain_pixel.add(key)

    if not geometries:
        return _PreparedGeometrySupports(grid={}, pixel={})

    workers = min(
        max(1, int(os.environ.get("HCC_ROI_RASTER_WORKERS", "32"))),
        len(geometries),
    )

    def rasterize(geometry: dict[str, Any]) -> np.ndarray:
        return _geometry_pixel_support_numpy(
            geometry,
            image_size=image_size,
        )

    grid_supports: dict[int, torch.Tensor] = {}
    pixel_supports: dict[int, torch.Tensor] = {}
    chunk_size = max(
        1,
        int(os.environ.get("HCC_ROI_TORCH_BATCH_SIZE", "128")),
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and os.environ.get("HCC_ROI_BUILD_DEVICE", "cuda").lower() == "cuda"
        else "cpu"
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rasterized = executor.map(rasterize, geometries)
        chunk_geometries: list[dict[str, Any]] = []
        chunk_masks: list[np.ndarray] = []
        for geometry, mask in zip(geometries, rasterized, strict=True):
            chunk_geometries.append(geometry)
            chunk_masks.append(mask)
            if len(chunk_masks) < chunk_size:
                continue
            _reduce_geometry_chunk(
                chunk_geometries,
                chunk_masks,
                grid_size=grid_size,
                device=device,
                retain_pixel=retain_pixel,
                grid_supports=grid_supports,
                pixel_supports=pixel_supports,
            )
            chunk_geometries.clear()
            chunk_masks.clear()
        if chunk_masks:
            _reduce_geometry_chunk(
                chunk_geometries,
                chunk_masks,
                grid_size=grid_size,
                device=device,
                retain_pixel=retain_pixel,
                grid_supports=grid_supports,
                pixel_supports=pixel_supports,
            )
    return _PreparedGeometrySupports(
        grid=grid_supports,
        pixel=pixel_supports,
    )


def _reduce_geometry_chunk(
    geometries: list[dict[str, Any]],
    masks: list[np.ndarray],
    *,
    grid_size: tuple[int, int],
    device: torch.device,
    retain_pixel: set[int],
    grid_supports: dict[int, torch.Tensor],
    pixel_supports: dict[int, torch.Tensor],
) -> None:
    pixel_batch = torch.from_numpy(np.stack(masks, axis=0))
    reduced = F.interpolate(
        pixel_batch[:, None].to(device=device, dtype=torch.float32),
        size=grid_size,
        mode="area",
    )[:, 0].gt(0).cpu()
    for index, geometry in enumerate(geometries):
        key = id(geometry)
        grid_supports[key] = reduced[index]
        if key in retain_pixel:
            pixel_supports[key] = pixel_batch[index].clone()


def geometry_token_mask(
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    prepared: _PreparedGeometrySupports | None = None,
) -> torch.Tensor:
    """Return grid cells touched by one annotation geometry."""

    kind = _geometry_kind(geometry)
    if kind == "point":
        image_h, image_w = image_size
        grid_h, grid_w = grid_size
        normalized = (
            str(geometry.get("coordinate_space", "pixel")).lower()
            == "normalized"
        )
        x, y = _xy(
            geometry.get("point", geometry.get("center", geometry)),
            image_size,
            normalized,
        )
        col = min(grid_w - 1, max(0, int(x * grid_w / image_w)))
        row = min(grid_h - 1, max(0, int(y * grid_h / image_h)))
        mask = torch.zeros(grid_size, dtype=torch.bool)
        mask[row, col] = True
        return mask

    if prepared is not None:
        support = prepared.grid.get(id(geometry))
        if support is not None:
            return support

    mask = _geometry_pixel_support(
        geometry,
        image_size=image_size,
    ).to(dtype=torch.float32)[None, None]
    support = F.interpolate(mask, size=grid_size, mode="area")[0, 0]
    return support > 0


def _add_point_center(
    point_centers: torch.Tensor,
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
    prepared: _PreparedGeometrySupports | None = None,
) -> None:
    support = geometry_token_mask(
        geometry,
        image_size=image_size,
        grid_size=tuple(point_centers.shape),
        prepared=prepared,
    )
    point_centers[support] += 1.0


def _add_geometry_center(
    point_centers: torch.Tensor,
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
    prepared: _PreparedGeometrySupports | None = None,
) -> None:
    support = geometry_token_mask(
        geometry,
        image_size=image_size,
        grid_size=tuple(point_centers.shape),
        prepared=prepared,
    )
    _add_support_center(point_centers, support)


def _add_support_center(
    point_centers: torch.Tensor,
    support: torch.Tensor,
) -> None:
    """Add one centre at the support cell nearest its centroid."""

    coordinates = support.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("instance geometry maps to zero spatial cells")
    centroid = coordinates.to(dtype=torch.float32).mean(dim=0)
    distance = (coordinates.to(dtype=torch.float32) - centroid).square().sum(dim=1)
    row, col = coordinates[int(distance.argmin().item())].tolist()
    point_centers[int(row), int(col)] += 1.0


def _add_connected_geometry_centers(
    point_centers: torch.Tensor,
    geometries: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    prepared: _PreparedGeometrySupports | None = None,
) -> None:
    """Count connected structure-brush support, not individual pen strokes.

    A large vessel, duct, or vacuole is often completed with several
    overlapping brush strokes. Those drawing events are annotation mechanics,
    not independent biological instances.
    """

    # Split instances before output-grid reduction. Reducing the union first
    # can make nearby but pixel-disjoint structures touch on the coarse grid
    # and incorrectly collapse them into one object.
    pixel_masks = [
        (
            prepared.pixel[id(geometry)]
            if prepared is not None and id(geometry) in prepared.pixel
            else _geometry_pixel_support(geometry, image_size=image_size)
        )
        for geometry in geometries
    ]
    stacked = torch.stack(pixel_masks, dim=0)
    dilated = F.max_pool2d(
        stacked[:, None].to(dtype=torch.float32),
        kernel_size=3,
        stride=1,
        padding=1,
    )[:, 0] > 0
    adjacency = (
        dilated[:, None] & stacked[None, :]
    ).flatten(2).any(dim=2)
    groups: list[list[int]] = []
    remaining = set(range(len(pixel_masks)))
    while remaining:
        pending = [remaining.pop()]
        group: list[int] = []
        while pending:
            current = pending.pop()
            group.append(current)
            neighbours = {
                other
                for other in remaining
                if bool(adjacency[current, other])
            }
            remaining.difference_update(neighbours)
            pending.extend(neighbours)
        groups.append(group)
    for group in groups:
        pixel_support = stacked[group].any(dim=0)
        grid_support = F.interpolate(
            pixel_support.to(dtype=torch.float32)[None, None],
            size=tuple(point_centers.shape),
            mode="area",
        )[0, 0] > 0
        _add_support_center(point_centers, grid_support)


def _add_area_support(
    area_positive: torch.Tensor,
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
    dilation_cells: int = 0,
    prepared: _PreparedGeometrySupports | None = None,
) -> None:
    support = geometry_token_mask(
        geometry,
        image_size=image_size,
        grid_size=tuple(area_positive.shape),
        prepared=prepared,
    )
    if not bool(support.any()):
        raise ValueError("area geometry maps to zero spatial cells")
    support = _dilate(support, dilation_cells)
    area_positive |= support


def _add_brush_bag(
    bag_ids: torch.Tensor,
    geometry: dict[str, Any],
    *,
    image_size: tuple[int, int],
    prepared: _PreparedGeometrySupports | None = None,
) -> None:
    support = geometry_token_mask(
        geometry,
        image_size=image_size,
        grid_size=tuple(bag_ids.shape),
        prepared=prepared,
    )
    if not bool(support.any()):
        raise ValueError("density-brush geometry maps to zero spatial cells")
    overlapping = torch.unique(bag_ids[support])
    overlapping = overlapping[overlapping > 0]
    if overlapping.numel() == 0:
        selected = int(bag_ids.max().item()) + 1
    else:
        selected = int(overlapping.min().item())
        for other in overlapping.tolist():
            if int(other) != selected:
                bag_ids[bag_ids == int(other)] = selected
    bag_ids[support] = selected


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    value = F.max_pool2d(
        mask.to(dtype=torch.float32)[None, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return value[0, 0] > 0


def _build_spatial_tile_target(
    tile_id: str,
    *,
    component_names: list[str],
    positions: dict[str, int],
    specs: list,
    tile_records: list[dict[str, Any]],
    tile_metadata: dict[str, Any],
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    point_tolerance_cells: int,
    prepared: _PreparedGeometrySupports | None = None,
) -> tuple[str, SpatialRoiTarget]:
    target = empty_spatial_roi_target(len(component_names), grid_size)
    point_centers = target.point_centers.clone()
    instance_exclusion_support = target.instance_exclusion_support.clone()
    brush_bag_ids = target.brush_bag_ids.clone()
    area_positive = target.area_positive.clone()
    explicit_negative = target.explicit_negative.clone()
    implicit_negative = target.implicit_negative.clone()

    by_component: dict[str, list[dict[str, Any]]] = {}
    for record in tile_records:
        by_component.setdefault(str(record["attribute"]), []).append(record)
    complete_all = bool(tile_metadata.get("complete_all", False))
    components_to_process = set(by_component)
    if complete_all:
        components_to_process.update(component_names)

    for component in components_to_process:
        idx = positions[component]
        spec = specs[idx]
        complete_negative = False
        structure_brush_geometries: list[dict[str, Any]] = []
        for record in by_component.get(component, []):
            geometry = record.get("geometry")
            state = str(record.get("state", "positive")).lower()
            if state not in {"positive", "negative"}:
                raise ValueError(
                    f"ROI state must be positive or negative, got {state!r}"
                )
            if geometry is None:
                if not bool(record.get("review_complete", False)):
                    raise ValueError(
                        "ROI record requires geometry or "
                        f"review_complete=true: tile={tile_id}"
                    )
                if state != "negative":
                    raise ValueError(
                        "geometry-free ROI completion must be negative: "
                        f"tile={tile_id} component={component}"
                    )
                complete_negative = True
                continue

            kind = _geometry_kind(geometry)
            if state == "positive":
                if spec.mode == CELL_INSTANCE_DENSITY:
                    if kind in POINT_GEOMETRIES:
                        _add_point_center(
                            point_centers[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                    elif kind in CIRCLE_GEOMETRIES:
                        _add_geometry_center(
                            point_centers[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                        instance_exclusion_support[idx] |= geometry_token_mask(
                            geometry,
                            image_size=image_size,
                            grid_size=grid_size,
                            prepared=prepared,
                        )
                        # A circle is both one resolved instance and the
                        # annotated full extent of that object. Keep its centre
                        # for the instance head and route its complete support
                        # to the measurement head.
                        _add_brush_bag(
                            brush_bag_ids[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                    else:
                        _add_brush_bag(
                            brush_bag_ids[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                elif spec.mode == STRUCTURE_INSTANCE_AREA:
                    if kind in POINT_GEOMETRIES:
                        _add_geometry_center(
                            point_centers[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                    elif kind in CIRCLE_GEOMETRIES:
                        _add_geometry_center(
                            point_centers[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                        _add_area_support(
                            area_positive[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                        instance_exclusion_support[idx] |= geometry_token_mask(
                            geometry,
                            image_size=image_size,
                            grid_size=grid_size,
                            prepared=prepared,
                        )
                    else:
                        structure_brush_geometries.append(geometry)
                        _add_area_support(
                            area_positive[idx],
                            geometry,
                            image_size=image_size,
                            prepared=prepared,
                        )
                        instance_exclusion_support[idx] |= geometry_token_mask(
                            geometry,
                            image_size=image_size,
                            grid_size=grid_size,
                            prepared=prepared,
                        )
                elif spec.mode == PIGMENT_BURDEN:
                    _add_area_support(
                        area_positive[idx],
                        geometry,
                        image_size=image_size,
                        dilation_cells=0,
                        prepared=prepared,
                    )
                elif spec.mode == CONTINUOUS_AREA:
                    _add_area_support(
                        area_positive[idx],
                        geometry,
                        image_size=image_size,
                        prepared=prepared,
                    )
                else:  # pragma: no cover - guarded by spatial schema
                    raise ValueError(
                        f"unsupported spatial mode: {spec.mode!r}"
                    )
            else:
                explicit_negative[idx] |= geometry_token_mask(
                    geometry,
                    image_size=image_size,
                    grid_size=grid_size,
                    prepared=prepared,
                )

        if structure_brush_geometries:
            _add_connected_geometry_centers(
                point_centers[idx],
                structure_brush_geometries,
                image_size=image_size,
                prepared=prepared,
            )

        positive_support = _dilate(
            point_centers[idx] > 0,
            point_tolerance_cells,
        ) | (brush_bag_ids[idx] > 0) | area_positive[idx]
        if complete_negative:
            if bool(positive_support.any()):
                raise ValueError(
                    "component cannot be both complete-negative and positive: "
                    f"tile={tile_id} component={component}"
                )
            explicit_negative[idx].fill_(True)
        overlap = (
            positive_support | instance_exclusion_support[idx]
        ) & explicit_negative[idx]
        if bool(overlap.any()):
            raise ValueError(
                "positive and explicit-negative ROI geometry overlap: "
                f"tile={tile_id} component={component} "
                f"cells={int(overlap.sum())}"
            )

    return tile_id, SpatialRoiTarget(
        point_centers=point_centers,
        instance_exclusion_support=instance_exclusion_support,
        brush_bag_ids=brush_bag_ids,
        area_positive=area_positive,
        explicit_negative=explicit_negative,
        implicit_negative=implicit_negative,
    )


def build_spatial_roi_targets(
    manifest_path: str | Path | None,
    *,
    component_names: list[str],
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    allowed_splits: set[str] | None = None,
    point_tolerance_cells: int = 1,
) -> dict[str, SpatialRoiTarget]:
    if not manifest_path:
        return {}
    if point_tolerance_cells < 0:
        raise ValueError(
            "point_tolerance_cells must be non-negative, "
            f"got {point_tolerance_cells}"
        )

    records, tile_metadata = _load_records(Path(manifest_path))
    positions = {name: idx for idx, name in enumerate(component_names)}
    specs = spatial_component_specs(
        component_names,
        unknown_mode=CELL_INSTANCE_DENSITY,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        split = str(record.get("split", "train"))
        if allowed_splits is not None and split not in allowed_splits:
            continue
        tile_id = str(record["tile_id"])
        component = str(record["attribute"])
        if component not in positions:
            raise ValueError(
                f"unknown ROI component {component!r} for tile {tile_id}"
            )
        grouped.setdefault(tile_id, []).append(record)

    eligible_metadata = {
        tile_id: metadata
        for tile_id, metadata in tile_metadata.items()
        if allowed_splits is None or str(metadata.get("split", "train")) in allowed_splits
    }
    tile_ids = set(grouped)
    tile_ids.update(
        tile_id
        for tile_id, metadata in eligible_metadata.items()
        if (
            bool(metadata.get("complete_all", False))
            or bool(metadata.get("count_complete"))
            or bool(metadata.get("measurement_complete"))
        )
    )

    prepared = _prepare_geometry_supports(
        [
            record
            for tile_id in sorted(tile_ids)
            for record in grouped.get(tile_id, [])
        ],
        positions=positions,
        specs=specs,
        image_size=image_size,
        grid_size=grid_size,
    )
    return dict(
        _build_spatial_tile_target(
            tile_id,
            component_names=component_names,
            positions=positions,
            specs=specs,
            tile_records=grouped.get(tile_id, []),
            tile_metadata=eligible_metadata.get(tile_id, {}),
            image_size=image_size,
            grid_size=grid_size,
            point_tolerance_cells=point_tolerance_cells,
            prepared=prepared,
        )
        for tile_id in sorted(tile_ids)
    )


def spatial_roi_payload(
    tile_id: str,
    roi_targets: dict[str, SpatialRoiTarget] | None,
    *,
    component_count: int,
    grid_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    item = (roi_targets or {}).get(tile_id)
    if item is None:
        item = empty_spatial_roi_target(component_count, grid_size)
    return {
        "spatial_point_centers": item.point_centers,
        "spatial_instance_exclusion_support": (
            item.instance_exclusion_support
            if item.instance_exclusion_support is not None
            else torch.zeros_like(
                item.area_positive,
                dtype=torch.bool,
            )
        ),
        "spatial_brush_bag_ids": item.brush_bag_ids,
        "spatial_area_positive": item.area_positive,
        "spatial_explicit_negative": item.explicit_negative,
        "spatial_implicit_negative": item.implicit_negative,
        "spatial_supervised": item.supervised,
    }
