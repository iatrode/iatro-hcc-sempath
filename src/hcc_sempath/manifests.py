from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


TILE_COLUMNS = (
    "tile_id",
    "patient_id",
    "slide_id",
    "tile_path",
    "x",
    "y",
    "split",
)
REQUIRED_TILE_COLUMNS = set(TILE_COLUMNS)


@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    patient_id: str
    slide_id: str
    tile_path: Path
    x: int
    y: int
    split: str


def read_tile_manifest(path: str | Path) -> list[TileRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_TILE_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"tile manifest missing columns: {sorted(missing)}")
        records = []
        for row in reader:
            records.append(
                TileRecord(
                    tile_id=row["tile_id"],
                    patient_id=row["patient_id"],
                    slide_id=row["slide_id"],
                    tile_path=Path(row["tile_path"]),
                    x=int(row["x"]),
                    y=int(row["y"]),
                    split=row["split"],
                )
            )
    return records


def write_tile_manifest(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TILE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
