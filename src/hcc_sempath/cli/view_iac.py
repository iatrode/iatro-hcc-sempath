from __future__ import annotations

import argparse
import io
import json
import socket
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from iatro.iac import PackReader
from iatro.iac.adapters.tiles import decode_jxl


MAP_BINS = 256


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
    if len(slide_table) == 0:
        return labels
    for row in range(len(slide_table)):
        idx = int(_table_value(slide_table, "slide_idx", row, row))
        slide_id = str(_table_value(slide_table, "slide_id", row, f"slide_{idx}"))
        patient_id = str(_table_value(slide_table, "patient_id", row, ""))
        labels[idx] = f"{slide_id} ({patient_id})" if patient_id and patient_id != slide_id else slide_id
    return labels


def _display_coordinate(header: dict, value: int, stride: int) -> int:
    if header.get("coordinate_mode") == "tile_grid":
        return value * stride
    return value


class IacViewerData:
    def __init__(self, package_path: str | Path) -> None:
        self.package_path = Path(package_path)
        self.reader = PackReader(self.package_path)
        self.header = self.reader.header
        self.slide_table = self.reader.slide_table
        self.record_table = self.reader.record_table
        self.payload_type = str(self.header.get("payload_type", "unknown"))
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
            if self.payload_type == "image_tiles"
        }

    def close(self) -> None:
        self.reader.close()

    def _load_records(self) -> list[IacRecord]:
        if self.payload_type == "image_tiles":
            return self._load_image_records()
        if self.payload_type == "teacher_features":
            return self._load_feature_records()
        raise ValueError(f"unsupported IAC payload_type: {self.payload_type}")

    def _load_image_records(self) -> list[IacRecord]:
        labels = _slide_labels(self.slide_table)
        records = []
        for row in range(len(self.record_table)):
            slide_idx = int(_table_value(self.record_table, "slide_idx", row, 0))
            grid_x = int(_table_value(self.record_table, "tile_x", row, 0))
            grid_y = int(_table_value(self.record_table, "tile_y", row, 0))
            display_x = _display_coordinate(self.header, grid_x, self.stride_x)
            display_y = _display_coordinate(self.header, grid_y, self.stride_y)
            records.append(
                IacRecord(
                    row=row,
                    slide_key=str(slide_idx),
                    slide_label=labels.get(slide_idx, f"slide_{slide_idx}"),
                    tile_id=str(_table_value(self.record_table, "tile_id", row, f"tile_{row}")),
                    grid_x=grid_x,
                    grid_y=grid_y,
                    display_x=display_x,
                    display_y=display_y,
                    split=str(_table_value(self.record_table, "split", row, "")),
                )
            )
        return records

    def _load_feature_records(self) -> list[IacRecord]:
        labels = _slide_labels(self.slide_table)
        records = []
        for row in range(len(self.record_table)):
            slide_idx = int(_table_value(self.record_table, "slide_idx", row, 0))
            grid_x = int(_table_value(self.record_table, "tile_x", row, 0))
            grid_y = int(_table_value(self.record_table, "tile_y", row, 0))
            display_x = _display_coordinate(self.header, grid_x, self.stride_x)
            display_y = _display_coordinate(self.header, grid_y, self.stride_y)
            records.append(
                IacRecord(
                    row=row,
                    slide_key=str(slide_idx),
                    slide_label=labels.get(slide_idx, f"slide_{slide_idx}"),
                    tile_id=str(_table_value(self.record_table, "tile_id", row, f"tile_{row}")),
                    grid_x=grid_x,
                    grid_y=grid_y,
                    display_x=display_x,
                    display_y=display_y,
                    split="",
                )
            )
        return records

    def summary(self) -> dict:
        return {
            "package": str(self.package_path),
            "payload_type": self.payload_type,
            "num_records": len(self.records),
            "num_slides": len(self._by_slide),
            "header": {
                key: value
                for key, value in self.header.items()
                if key
                in {
                    "payload_type",
                    "codec",
                    "tile_width",
                    "tile_height",
                    "stride_x",
                    "stride_y",
                    "teacher",
                    "feature_dim",
                    "dtype",
                    "num_records",
                    "data_length",
                }
            },
            "slides": [
                {"key": key, "label": records[0].slide_label, "count": len(records)}
                for key, records in sorted(self._by_slide.items(), key=lambda item: item[0])
            ],
        }

    def map_payload(self, slide_key: str) -> dict:
        records = self._records_for_slide(slide_key)
        bounds = self._bounds(records)
        counts = [0] * (MAP_BINS * MAP_BINS)
        if records:
            min_x, max_x, min_y, max_y = bounds
            dx = max(1, max_x - min_x)
            dy = max(1, max_y - min_y)
            for record in records:
                bx = min(MAP_BINS - 1, max(0, int((record.display_x - min_x) / dx * (MAP_BINS - 1))))
                by = min(MAP_BINS - 1, max(0, int((record.display_y - min_y) / dy * (MAP_BINS - 1))))
                counts[by * MAP_BINS + bx] += 1
        return {
            "slide": slide_key,
            "payload_type": self.payload_type,
            "bins": MAP_BINS,
            "bounds": {
                "min_x": bounds[0],
                "max_x": bounds[1],
                "min_y": bounds[2],
                "max_y": bounds[3],
            },
            "count": len(records),
            "max_count": max(counts) if counts else 0,
            "counts": counts,
        }

    def nearest(self, slide_key: str, x: float, y: float) -> dict:
        records = self._records_for_slide(slide_key)
        if not records:
            raise FileNotFoundError(f"slide has no records: {slide_key}")
        nearest = min(records, key=lambda record: (record.display_x - x) ** 2 + (record.display_y - y) ** 2)
        return {"record": self._record_json(nearest)}

    def image_window_at(self, slide_key: str, x: float, y: float, radius: int = 2) -> dict:
        if self.payload_type != "image_tiles":
            raise ValueError("5x5 image window is only available for image tile IAC packages")
        center_x = int(round(x / max(1, self.stride_x)))
        center_y = int(round(y / max(1, self.stride_y)))
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                record = self._image_lookup.get((slide_key, center_x + dx, center_y + dy))
                cells.append(
                    {
                        "dx": dx,
                        "dy": dy,
                        "record": None if record is None else self._record_json(record),
                        "url": None if record is None else f"/api/tile?row={record.row}",
                    }
                )
        return {
            "center": {
                "slide": slide_key,
                "x": center_x * self.stride_x,
                "y": center_y * self.stride_y,
                "grid_x": center_x,
                "grid_y": center_y,
            },
            "radius": radius,
            "cells": cells,
        }

    def read_tile_png(self, row: int) -> bytes:
        if self.payload_type != "image_tiles":
            raise ValueError("tile image endpoint is only available for image tile IAC packages")
        image = decode_jxl(self.reader.read_payload(row))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _records_for_slide(self, slide_key: str) -> list[IacRecord]:
        if slide_key == "__all__":
            return self.records
        return self._by_slide.get(slide_key, [])

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


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IAC Viewer</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #191c20; }
    header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; border-bottom: 1px solid #d9dee5; background: #ffffff; }
    h1 { font-size: 16px; margin: 0; font-weight: 650; }
    main { display: grid; grid-template-columns: 340px minmax(380px, 1fr) 440px; gap: 14px; padding: 14px; height: calc(100vh - 85px); box-sizing: border-box; }
    section { background: #ffffff; border: 1px solid #d9dee5; border-radius: 8px; overflow: hidden; min-height: 0; }
    .panel-head { height: 42px; display: flex; align-items: center; justify-content: space-between; padding: 0 12px; border-bottom: 1px solid #e3e7ed; font-size: 13px; font-weight: 650; }
    .panel-body { padding: 12px; overflow: auto; height: calc(100% - 42px); box-sizing: border-box; }
    .meta-grid { display: grid; grid-template-columns: 120px 1fr; gap: 8px 10px; font-size: 12px; line-height: 1.35; }
    .key { color: #65717f; }
    select { width: 100%; height: 32px; border: 1px solid #cbd3dd; border-radius: 6px; padding: 0 8px; background: #fff; }
    canvas { width: 100%; aspect-ratio: 1 / 1; background: #0c1016; border-radius: 6px; display: block; cursor: crosshair; }
    .hint { color: #65717f; font-size: 12px; margin-top: 10px; line-height: 1.45; }
    .window-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; }
    .tile-cell { aspect-ratio: 1 / 1; background: #eef1f5; border: 1px solid #d7dde5; border-radius: 5px; display: flex; align-items: center; justify-content: center; overflow: hidden; font-size: 11px; color: #8894a2; }
    .tile-cell img { width: 100%; height: 100%; object-fit: cover; image-rendering: auto; }
    .tile-cell.center { outline: 2px solid #12805c; outline-offset: -2px; }
    pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; background: #f3f5f7; padding: 10px; border-radius: 6px; margin: 0; }
    .pill { display: inline-flex; align-items: center; height: 24px; padding: 0 8px; border-radius: 999px; background: #e9eef4; color: #27313d; font-size: 12px; }
    @media (max-width: 1100px) { main { grid-template-columns: 1fr; height: auto; } section { min-height: 360px; } }
  </style>
</head>
<body>
  <header>
    <h1>IAC Viewer</h1>
    <span id="type" class="pill">loading</span>
  </header>
  <main>
    <section>
      <div class="panel-head"><span>Package</span></div>
      <div class="panel-body">
        <div class="meta-grid" id="meta"></div>
        <div style="height:14px"></div>
        <label class="key" for="slide">Slide</label>
        <div style="height:6px"></div>
        <select id="slide"></select>
        <p class="hint">The map is built from record coordinates only. Feature caches are shown as a coordinate heatmap and do not decode feature payloads.</p>
      </div>
    </section>
    <section>
      <div class="panel-head"><span>Coordinate Map</span><span id="mapStatus" class="key"></span></div>
      <div class="panel-body">
        <canvas id="map" width="768" height="768"></canvas>
        <p class="hint" id="bounds"></p>
      </div>
    </section>
    <section>
      <div class="panel-head"><span id="detailTitle">Selection</span></div>
      <div class="panel-body">
        <div id="window" class="window-grid"></div>
        <div style="height:12px"></div>
        <pre id="record">Click the map.</pre>
      </div>
    </section>
  </main>
  <script>
    const state = { summary: null, map: null, selectedPoint: null };
    const canvas = document.getElementById('map');
    const ctx = canvas.getContext('2d');

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    function setMeta(summary) {
      document.getElementById('type').textContent = summary.payload_type;
      const meta = document.getElementById('meta');
      const rows = [
        ['Package', summary.package],
        ['Records', summary.num_records],
        ['Slides', summary.num_slides],
      ];
      for (const [key, value] of Object.entries(summary.header)) rows.push([key, value]);
      meta.innerHTML = rows.map(([k, v]) => `<div class="key">${k}</div><div>${String(v)}</div>`).join('');
      const select = document.getElementById('slide');
      select.innerHTML = summary.slides.map(s => `<option value="${s.key}">${s.label} (${s.count})</option>`).join('');
      select.addEventListener('change', () => loadMap());
    }

    function colorFor(value, maxValue, payloadType) {
      if (value <= 0) return [12, 16, 22, 255];
      const t = Math.min(1, Math.log1p(value) / Math.log1p(Math.max(1, maxValue)));
      if (payloadType === 'teacher_features') {
        return [Math.round(22 + 220 * t), Math.round(42 + 130 * (1 - t)), Math.round(150 + 70 * t), 255];
      }
      return [Math.round(20 + 230 * t), Math.round(72 + 150 * t), Math.round(74 + 40 * (1 - t)), 255];
    }

    function drawMap() {
      const map = state.map;
      const bins = map.bins;
      const image = ctx.createImageData(bins, bins);
      for (let i = 0; i < map.counts.length; i++) {
        const [r, g, b, a] = colorFor(map.counts[i], map.max_count, map.payload_type);
        const j = i * 4;
        image.data[j] = r; image.data[j + 1] = g; image.data[j + 2] = b; image.data[j + 3] = a;
      }
      const offscreen = document.createElement('canvas');
      offscreen.width = bins;
      offscreen.height = bins;
      offscreen.getContext('2d').putImageData(image, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
      if (state.selectedPoint) {
        const b = map.bounds;
        const dx = Math.max(1, b.max_x - b.min_x);
        const dy = Math.max(1, b.max_y - b.min_y);
        const px = (state.selectedPoint.x - b.min_x) / dx * canvas.width;
        const py = (state.selectedPoint.y - b.min_y) / dy * canvas.height;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    async function loadMap() {
      const slide = document.getElementById('slide').value;
      state.map = await getJson(`/api/map?slide=${encodeURIComponent(slide)}`);
      state.selectedPoint = null;
      drawMap();
      document.getElementById('mapStatus').textContent = `${state.map.count} records`;
      const b = state.map.bounds;
      document.getElementById('bounds').textContent = `x ${b.min_x}..${b.max_x}, y ${b.min_y}..${b.max_y}`;
      document.getElementById('window').innerHTML = '';
      document.getElementById('record').textContent = 'Click the map.';
    }

    async function selectAt(event) {
      const rect = canvas.getBoundingClientRect();
      const rx = (event.clientX - rect.left) / rect.width;
      const ry = (event.clientY - rect.top) / rect.height;
      const b = state.map.bounds;
      const x = b.min_x + rx * Math.max(1, b.max_x - b.min_x);
      const y = b.min_y + ry * Math.max(1, b.max_y - b.min_y);
      const slide = document.getElementById('slide').value;
      const nearest = await getJson(`/api/nearest?slide=${encodeURIComponent(slide)}&x=${x}&y=${y}`);
      state.selectedPoint = {x, y};
      document.getElementById('record').textContent = JSON.stringify({clicked_x: x, clicked_y: y, nearest_record: nearest.record}, null, 2);
      drawMap();
      if (state.summary.payload_type === 'image_tiles') {
        const win = await getJson(`/api/window?slide=${encodeURIComponent(slide)}&x=${x}&y=${y}`);
        document.getElementById('window').innerHTML = win.cells.map(c => {
          const cls = c.dx === 0 && c.dy === 0 ? 'tile-cell center' : 'tile-cell';
          return c.url ? `<div class="${cls}" title="${c.record.tile_id}"><img src="${c.url}" loading="lazy"></div>` : `<div class="${cls}">empty</div>`;
        }).join('');
        document.getElementById('detailTitle').textContent = '5x5 Tile Window';
      } else {
        document.getElementById('window').innerHTML = '';
        document.getElementById('detailTitle').textContent = 'Feature Cache Selection';
      }
    }

    canvas.addEventListener('click', selectAt);
    getJson('/api/summary').then(summary => {
      state.summary = summary;
      setMeta(summary);
      return loadMap();
    }).catch(err => {
      document.body.innerHTML = `<pre>${err.stack || err}</pre>`;
    });
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    data: IacViewerData

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/summary":
                self._send_json(self.data.summary())
            elif parsed.path == "/api/map":
                slide = query.get("slide", ["__all__"])[0]
                self._send_json(self.data.map_payload(slide))
            elif parsed.path == "/api/nearest":
                slide = query.get("slide", ["__all__"])[0]
                x = float(query.get("x", ["0"])[0])
                y = float(query.get("y", ["0"])[0])
                self._send_json(self.data.nearest(slide, x, y))
            elif parsed.path == "/api/window":
                slide = query.get("slide", ["__all__"])[0]
                x = float(query.get("x", ["0"])[0])
                y = float(query.get("y", ["0"])[0])
                self._send_json(self.data.image_window_at(slide, x, y))
            elif parsed.path == "/api/tile":
                row = int(query.get("row", ["0"])[0])
                self._send_bytes(self.data.read_tile_png(row), "image/png")
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"{type(exc).__name__}: {exc}".encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict) -> None:
        self._send_bytes(json.dumps(payload, separators=(",", ":")).encode("utf-8"), "application/json")

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _available_port(host: str, requested: int) -> int:
    if requested != 0:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a local browser UI for inspecting an IatroCache package.")
    parser.add_argument("--package", required=True, help="Input .iac package.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Use 0 to pick a free port.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the default browser automatically.")
    args = parser.parse_args()

    data = IacViewerData(args.package)
    port = _available_port(args.host, args.port)
    ViewerHandler.data = data
    server = ThreadingHTTPServer((args.host, port), ViewerHandler)
    url = f"http://{args.host}:{port}/"
    print(f"iac_viewer_start url={url} package={Path(args.package)} payload_type={data.payload_type}", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        data.close()
        server.server_close()


if __name__ == "__main__":
    main()
