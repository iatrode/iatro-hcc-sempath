#!/usr/bin/env python
"""Rebase frozen A0 expert truth onto an existing server tile-IAC row order."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from iatro.iac import read_tables


def _server_package(
    raw: dict[str, Any],
    *,
    tile_root: Path,
) -> Path:
    declared = Path(str(raw.get("iac") or raw.get("iac_path") or ""))
    if declared.is_absolute() and declared.is_file():
        return declared.resolve()
    dataset = str(raw.get("dataset") or declared.parent.name).strip()
    if not dataset:
        raise ValueError(
            f"missing dataset for tile={raw.get('tile_id')}"
        )
    candidate = tile_root / dataset / declared.name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"server tile package is missing: {candidate}"
        )
    return candidate.resolve()


def _load_classification(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    required = {"tile_id", "dataset", "iac", "row"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"classification CSV missing columns: {sorted(missing)}"
        )
    return fieldnames, rows


def _load_spatial(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("annotations"), dict):
        raise ValueError("spatial JSON has no annotations object")
    return payload


def _resolve_rows(
    records: list[dict[str, Any]],
    *,
    tile_root: Path,
) -> dict[tuple[Path, str], int]:
    requested: dict[Path, set[str]] = {}
    for raw in records:
        tile_id = str(raw.get("tile_id") or "").strip()
        if not tile_id:
            raise ValueError("expert truth contains an empty tile_id")
        package = _server_package(raw, tile_root=tile_root)
        requested.setdefault(package, set()).add(tile_id)

    resolved: dict[tuple[Path, str], int] = {}
    for package, wanted in sorted(
        requested.items(),
        key=lambda item: str(item[0]),
    ):
        _, _, table = read_tables(package)
        remaining = set(wanted)
        for row, scalar in enumerate(table.column("tile_id")):
            tile_id = str(scalar.as_py())
            if tile_id not in remaining:
                continue
            resolved[(package, tile_id)] = row
            remaining.remove(tile_id)
            if not remaining:
                break
        if remaining:
            raise ValueError(
                "expert tiles are absent from the declared server package: "
                f"package={package} count={len(remaining)}"
            )
    return resolved


def _atomic_classification(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve frozen classification/spatial tile IDs against the "
            "server's existing tile-IAC row order."
        )
    )
    parser.add_argument("--classification-csv", required=True)
    parser.add_argument("--spatial-json", required=True)
    parser.add_argument("--tile-root", required=True)
    parser.add_argument("--output-classification-csv", required=True)
    parser.add_argument("--output-spatial-json", required=True)
    args = parser.parse_args()

    classification_path = Path(args.classification_csv).resolve()
    spatial_path = Path(args.spatial_json).resolve()
    tile_root = Path(args.tile_root).resolve()
    classification_output = Path(
        args.output_classification_csv
    ).resolve()
    spatial_output = Path(args.output_spatial_json).resolve()
    for output in (classification_output, spatial_output):
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite rebased truth: {output}"
            )

    fieldnames, classification_rows = _load_classification(
        classification_path
    )
    spatial_payload = _load_spatial(spatial_path)
    spatial_rows = [
        raw
        for raw in spatial_payload["annotations"].values()
        if isinstance(raw, dict)
    ]
    all_rows: list[dict[str, Any]] = [
        *classification_rows,
        *spatial_rows,
    ]
    resolved = _resolve_rows(all_rows, tile_root=tile_root)

    for raw in all_rows:
        package = _server_package(raw, tile_root=tile_root)
        tile_id = str(raw["tile_id"]).strip()
        raw["row"] = resolved[(package, tile_id)]

    spatial_payload.setdefault("source", {})
    spatial_payload["source"]["server_row_rebase"] = {
        "tile_root": str(tile_root),
        "classification_tiles": len(classification_rows),
        "spatial_tiles": len(spatial_rows),
    }
    _atomic_classification(
        classification_output,
        fieldnames=fieldnames,
        rows=classification_rows,
    )
    _atomic_json(spatial_output, spatial_payload)
    print(
        "a0_expert_rows_rebased "
        f"classification={len(classification_rows)} "
        f"spatial={len(spatial_rows)} "
        f"packages={len({package for package, _ in resolved})}"
    )


if __name__ == "__main__":
    main()
