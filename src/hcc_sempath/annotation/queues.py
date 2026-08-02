from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hcc_sempath.spatial_schema import DEFAULT_SPATIAL_COMPONENTS


# This bounds only the component-positive navigation pool. Information curves,
# not this planning cap, determine whether more annotation is required.
DEFAULT_PLANNING_COVERAGE = 100


def build_priority_manifest(annotation_paths: list[str | Path]) -> dict:
    by_tile_id: dict[str, dict] = {}
    for source in annotation_paths:
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotations = payload.get("annotations")
        if not isinstance(annotations, dict):
            raise ValueError(f"priority source requires an annotations object: {path}")
        for item in annotations.values():
            tile_id = str(item.get("tile_id") or "").strip()
            iac = str(item.get("iac") or item.get("iac_path") or "").strip()
            row = int(item.get("row", -1))
            if not tile_id or not iac or row < 0:
                continue
            candidate = {
                "tile_id": tile_id,
                "iac": iac,
                "row": row,
                "slide": str(item.get("slide") or item.get("slide_id") or ""),
            }
            previous = by_tile_id.get(tile_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"conflicting duplicate tile_id: {tile_id}")
            by_tile_id[tile_id] = candidate

    candidates = sorted(by_tile_id.values(), key=lambda item: (Path(item["iac"]).as_posix(), item["row"], item["tile_id"]))
    for rank, item in enumerate(candidates):
        item["rank"] = rank
    if not candidates:
        raise ValueError("priority sources did not contain any usable annotations")
    return {"version": 1, "candidate_count": len(candidates), "candidates": candidates}


def _records(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    annotations = payload.get("annotations")
    if isinstance(annotations, dict):
        return [dict(item) for item in annotations.values()]
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [dict(item) for item in candidates]
    raise ValueError(f"unsupported candidate source: {path}")


def build_roi_candidate_queue(
    annotation_paths: list[str | Path],
    *,
    planning_coverage: int = DEFAULT_PLANNING_COVERAGE,
) -> dict:
    if planning_coverage <= 0:
        raise ValueError("planning_coverage must be positive")
    attributes = list(DEFAULT_SPATIAL_COMPONENTS)
    by_tile: dict[str, dict] = {}
    for path in annotation_paths:
        for item in _records(path):
            tile_id = str(item.get("tile_id") or "").strip()
            source_spatial = sorted(
                set(item.get("source_spatial") or item.get("spatial") or [])
                & set(attributes)
            )
            if not tile_id or not source_spatial:
                continue
            candidate = {
                "tile_id": tile_id,
                "iac": str(item.get("iac") or item.get("iac_path") or ""),
                "row": int(item.get("row", -1)),
                "slide": str(item.get("slide") or item.get("slide_id") or ""),
                "source_spatial": source_spatial,
            }
            previous = by_tile.get(tile_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"conflicting duplicate tile_id: {tile_id}")
            by_tile[tile_id] = candidate

    available = Counter()
    for item in by_tile.values():
        available.update(item["source_spatial"])
    goals = {name: min(planning_coverage, available[name]) for name in attributes}
    selected: list[dict] = []
    counts = Counter()
    remaining = dict(by_tile)
    while True:
        useful = {name: max(0, goals[name] - counts[name]) for name in attributes}
        best = min(
            remaining.values(),
            key=lambda item: (
                -sum(
                    1 / max(1, available[name])
                    for name in item["source_spatial"]
                    if useful[name]
                ),
                -sum(useful[name] for name in item["source_spatial"]),
                item["slide"],
                item["tile_id"],
            ),
            default=None,
        )
        if best is None or not any(useful[name] for name in best["source_spatial"]):
            break
        remaining.pop(best["tile_id"])
        priority = [name for name in best["source_spatial"] if useful[name]]
        selected.append({**best, "priority_attributes": priority, "rank": len(selected)})
        counts.update(best["source_spatial"])
    unfilled = {
        name: max(0, planning_coverage - available[name])
        for name in attributes
    }
    return {
        "version": 2,
        "spatial_prototypes": attributes,
        "planning_coverage_per_attribute": {
            name: planning_coverage for name in attributes
        },
        "source_positive_inventory": {name: available[name] for name in attributes},
        "selected_source_coverage": {name: counts[name] for name in attributes},
        "unfilled_planning_coverage": unfilled,
        "candidate_count": len(selected),
        "candidates": selected,
    }
