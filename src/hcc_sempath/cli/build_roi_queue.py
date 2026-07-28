from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hcc_sempath.cli.annotate_prototypes import ROI_SPATIAL_PROTOTYPES


# This only bounds the size of the component-positive navigation pool. It is not
# an annotation target or a stopping rule; per-component information curves
# decide whether additional expert annotation is needed.
DEFAULT_PLANNING_COVERAGE = 100


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
    attributes = list(ROI_SPATIAL_PROTOTYPES)
    by_tile: dict[str, dict] = {}
    for path in annotation_paths:
        for item in _records(path):
            tile_id = str(item.get("tile_id") or "").strip()
            source_spatial = sorted(set(item.get("source_spatial") or item.get("spatial") or []) & set(attributes))
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
    goals = {
        name: min(planning_coverage, available[name])
        for name in attributes
    }
    selected: list[dict] = []
    counts = Counter()
    remaining = dict(by_tile)
    while True:
        useful = {
            name: max(0, goals[name] - counts[name])
            for name in attributes
        }
        best = min(
            remaining.values(),
            key=lambda item: (
                -sum(1 / max(1, available[name]) for name in item["source_spatial"] if useful[name]),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a component-presence navigation pool for ROI annotation."
    )
    parser.add_argument(
        "--annotations",
        action="append",
        required=True,
        help=(
            "Tile-level annotation JSON or supplemental candidate JSON; "
            "repeatable. Existing spatial labels are optional and only affect "
            "priority."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--planning-coverage",
        type=int,
        default=DEFAULT_PLANNING_COVERAGE,
        help=(
            "Maximum positive-tile coverage retained per component "
            "for navigation planning; this is not an annotation target."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace ROI candidate pool without --overwrite: {output}")
    payload = build_roi_candidate_queue(
        args.annotations,
        planning_coverage=args.planning_coverage,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "candidate_count",
                    "source_positive_inventory",
                    "unfilled_planning_coverage",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
