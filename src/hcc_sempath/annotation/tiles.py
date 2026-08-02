from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from iatro.iac import PackReader
from iatro.iac.adapters.tiles import decode_jxl


@dataclass(frozen=True)
class IacRecord:
    row: int
    slide_key: str
    slide_label: str
    tile_id: str
    grid_x: int
    grid_y: int
    display_x: int
    display_y: int
    split: str


def _table_value(table, column: str, row: int, default=None):
    if column not in table.column_names:
        return default
    return table.column(column)[row].as_py()


def _slide_labels(slide_table) -> dict[int, str]:
    labels = {}
    for row in range(len(slide_table)):
        index = int(_table_value(slide_table, "slide_idx", row, row))
        slide_id = str(_table_value(slide_table, "slide_id", row, f"slide_{index}"))
        patient_id = str(_table_value(slide_table, "patient_id", row, ""))
        labels[index] = f"{slide_id} ({patient_id})" if patient_id and patient_id != slide_id else slide_id
    return labels


class AnnotationTilePackageReader:
    """Image-tile package index used by the SemPath annotation workspace."""

    def __init__(self, package_path: str | Path) -> None:
        self.package_path = Path(package_path)
        self.reader = PackReader(self.package_path)
        self.header = self.reader.header
        self.slide_table = self.reader.slide_table
        self.record_table = self.reader.record_table
        self.payload_type = str(self.header.get("payload_type", "unknown"))
        if self.payload_type != "image_tiles":
            self.reader.close()
            raise ValueError(f"annotation requires image_tiles, got {self.payload_type}")
        self.stride_x = int(self.header.get("stride_x", 1))
        self.stride_y = int(self.header.get("stride_y", 1))
        self.records = self._load_records()
        self._by_row = {record.row: record for record in self.records}
        self._by_slide: dict[str, list[IacRecord]] = {}
        for record in self.records:
            self._by_slide.setdefault(record.slide_key, []).append(record)
        self._image_lookup = {
            (record.slide_key, record.grid_x, record.grid_y): record
            for record in self.records
        }

    def close(self) -> None:
        self.reader.close()

    def _load_records(self) -> list[IacRecord]:
        labels = _slide_labels(self.slide_table)
        coordinate_mode = self.header.get("coordinate_mode")
        records = []
        for row in range(len(self.record_table)):
            slide_index = int(_table_value(self.record_table, "slide_idx", row, 0))
            grid_x = int(_table_value(self.record_table, "tile_x", row, 0))
            grid_y = int(_table_value(self.record_table, "tile_y", row, 0))
            display_x = grid_x * self.stride_x if coordinate_mode == "tile_grid" else grid_x
            display_y = grid_y * self.stride_y if coordinate_mode == "tile_grid" else grid_y
            records.append(
                IacRecord(
                    row=row,
                    slide_key=str(slide_index),
                    slide_label=labels.get(slide_index, f"slide_{slide_index}"),
                    tile_id=str(_table_value(self.record_table, "tile_id", row, f"tile_{row}")),
                    grid_x=grid_x,
                    grid_y=grid_y,
                    display_x=display_x,
                    display_y=display_y,
                    split=str(_table_value(self.record_table, "split", row, "")),
                )
            )
        return records

    def nearest(self, slide_key: str, x: float, y: float) -> dict:
        records = self.records if slide_key == "__all__" else self._by_slide.get(slide_key, [])
        if not records:
            raise FileNotFoundError(f"slide has no records: {slide_key}")
        record = min(
            records,
            key=lambda item: (item.display_x - x) ** 2 + (item.display_y - y) ** 2,
        )
        return {"record": self._record_json(record)}

    def read_tile_png(self, row: int) -> bytes:
        image = decode_jxl(self.reader.read_payload(row))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _bounds(records: list[IacRecord]) -> tuple[int, int, int, int]:
        if not records:
            return (0, 1, 0, 1)
        xs = [record.display_x for record in records]
        ys = [record.display_y for record in records]
        return (min(xs), max(xs), min(ys), max(ys))

    @staticmethod
    def _record_json(record: IacRecord) -> dict:
        return {
            "row": record.row,
            "slide": record.slide_key,
            "slide_label": record.slide_label,
            "tile_id": record.tile_id,
            "grid_x": record.grid_x,
            "grid_y": record.grid_y,
            "x": record.display_x,
            "y": record.display_y,
            "split": record.split,
        }
