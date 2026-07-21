from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shared L1/L2 tile-priority manifest.")
    parser.add_argument(
        "--annotations",
        action="append",
        required=True,
        help="Annotation state whose tile identities seed the shared boundary; repeatable.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to replace priority manifest without --overwrite: {output}")
    payload = build_priority_manifest(args.annotations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": payload["candidate_count"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
