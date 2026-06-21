from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hcc_sempath.cli.annotate_prototypes import ROI_L2_PROTOTYPES, ROI_TARGET_PER_ATTRIBUTE


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
    target: int = ROI_TARGET_PER_ATTRIBUTE,
) -> dict:
    attributes = list(ROI_L2_PROTOTYPES)
    by_tile: dict[str, dict] = {}
    for path in annotation_paths:
        for item in _records(path):
            tile_id = str(item.get("tile_id") or "").strip()
            source_l2 = sorted(set(item.get("source_l2") or item.get("l2") or []) & set(attributes))
            if not tile_id or not source_l2:
                continue
            candidate = {
                "tile_id": tile_id,
                "iac": str(item.get("iac") or item.get("iac_path") or ""),
                "row": int(item.get("row", -1)),
                "slide": str(item.get("slide") or item.get("slide_id") or ""),
                "source_l2": source_l2,
            }
            previous = by_tile.get(tile_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"conflicting duplicate tile_id: {tile_id}")
            by_tile[tile_id] = candidate

    available = Counter()
    for item in by_tile.values():
        available.update(item["source_l2"])
    goals = {name: min(target, available[name]) for name in attributes}
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
                -sum(1 / max(1, available[name]) for name in item["source_l2"] if useful[name]),
                -sum(useful[name] for name in item["source_l2"]),
                item["slide"],
                item["tile_id"],
            ),
            default=None,
        )
        if best is None or not any(useful[name] for name in best["source_l2"]):
            break
        remaining.pop(best["tile_id"])
        priority = [name for name in best["source_l2"] if useful[name]]
        selected.append({**best, "priority_attributes": priority, "rank": len(selected)})
        counts.update(best["source_l2"])

    targets = {name: target for name in attributes}
    unfilled = {name: max(0, target - available[name]) for name in attributes}
    return {
        "version": 1,
        "frozen": not any(unfilled.values()),
        "l2_prototypes": attributes,
        "target_per_attribute": targets,
        "source_positive_inventory": {name: available[name] for name in attributes},
        "selected_source_coverage": {name: counts[name] for name in attributes},
        "unfilled_targets": unfilled,
        "candidate_count": len(selected),
        "candidates": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen quota-driven V2 ROI candidate queue.")
    parser.add_argument("--annotations", action="append", required=True, help="Tile-level annotation JSON or supplemental candidate JSON; repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=ROI_TARGET_PER_ATTRIBUTE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace frozen queue without --overwrite: {output}")
    payload = build_roi_candidate_queue(args.annotations, target=args.target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("candidate_count", "source_positive_inventory", "unfilled_targets")}, indent=2))


if __name__ == "__main__":
    main()
