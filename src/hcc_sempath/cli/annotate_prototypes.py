from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import socket
import tempfile
import webbrowser
from time import perf_counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

from hcc_sempath.cli.view_iac import IacRecord, IacViewerData
from hcc_sempath.io.iatrocache import read_header


LOG = logging.getLogger("hcc_sempath.annotate_prototypes")

L1_PROTOTYPES = [
    "HCC-trabecular",
    "HCC-solid",
    "HCC-pseudoglandular",
    "HCC-mixed-pattern",
    "Background-liver",
    "Fibrous-stromal",
    "Degenerative-material",
    "Indeterminate-region",
    "Artifact-non-tissue",
]

L2_PROTOTYPES = [
    "necrotic",
    "hemorrhagic-blood-rich",
    "bile-pigment-rich",
    "inflammatory-rich",
    "fibrotic",
    "steatotic-vacuolated",
    "interface-capsule",
]


@dataclass(frozen=True)
class AnnotationPackage:
    path: Path
    rel_path: str
    dataset: str
    total: int


def _find_free_port(host: str, preferred: int) -> int:
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def discover_iac_packages(input_path: str | Path) -> list[AnnotationPackage]:
    root = Path(input_path).resolve()
    LOG.info("discover_iac_start input=%s", root)
    paths = [root] if root.is_file() and root.suffix == ".iac" else sorted(root.rglob("*.iac"))
    packages = []
    for path in paths:
        header = read_header(path)
        if header.get("payload_type") != "image_tiles":
            if root.is_file():
                raise ValueError(f"annotation requires image-tile IAC package: {path}")
            LOG.info("discover_iac_skip path=%s payload_type=%s", path, header.get("payload_type"))
            continue
        total = int(header.get("num_records", 0))
        rel = path.name if root.is_file() else str(path.relative_to(root))
        dataset = ""
        if not root.is_file():
            parent = path.parent.relative_to(root)
            dataset = "" if str(parent) == "." else parent.parts[0]
        packages.append(AnnotationPackage(path=path, rel_path=rel, dataset=dataset, total=total))
        LOG.info("discover_iac_add rel_path=%s dataset=%s records=%d", rel, dataset or "-", total)
    if not packages:
        raise FileNotFoundError(f"no image-tile .iac packages found under: {root}")
    LOG.info("discover_iac_done input=%s packages=%d", root, len(packages))
    return packages


def _annotation_key(package: AnnotationPackage, record: IacRecord) -> str:
    return f"{package.rel_path}::{record.tile_id}::{record.display_x},{record.display_y}"


class AnnotationState:
    def __init__(self, state_path: str | Path, input_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.input_path = str(Path(input_path).resolve())
        self.annotations: dict[str, dict] = {}
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.input_path = str(Path(payload.get("input_path", self.input_path)).resolve())
            self.annotations = dict(payload.get("annotations", {}))
            LOG.info("state_load path=%s annotations=%d input=%s", self.state_path, len(self.annotations), self.input_path)
        else:
            LOG.info("state_new path=%s input=%s", self.state_path, self.input_path)

    @property
    def csv_path(self) -> Path:
        return self.state_path.with_suffix(".csv")

    def is_annotated(self, package: AnnotationPackage, record: IacRecord) -> bool:
        return _annotation_key(package, record) in self.annotations

    def counts_for_package(self, package: AnnotationPackage, records: list[IacRecord]) -> dict:
        annotated = sum(1 for record in records if self.is_annotated(package, record))
        return {"annotated": annotated, "total": len(records), "remaining": max(0, len(records) - annotated)}

    def annotations_for_package(self, package: AnnotationPackage) -> list[dict]:
        prefix = f"{package.rel_path}::"
        return [value for key, value in self.annotations.items() if key.startswith(prefix)]

    def lightweight_counts_for_package(self, package: AnnotationPackage) -> dict:
        annotated = len(self.annotations_for_package(package))
        return {"annotated": annotated, "total": package.total, "remaining": max(0, package.total - annotated)}

    def save_annotation(self, package: AnnotationPackage, record: IacRecord, l1: str, l2: list[str]) -> None:
        if l1 not in L1_PROTOTYPES:
            raise ValueError(f"unknown L1 prototype: {l1}")
        unknown_l2 = sorted(set(l2) - set(L2_PROTOTYPES))
        if unknown_l2:
            raise ValueError(f"unknown L2 prototype(s): {unknown_l2}")
        payload = {
            "dataset": package.dataset,
            "iac": package.rel_path,
            "iac_path": str(package.path),
            "tile_id": record.tile_id,
            "row": record.row,
            "slide": record.slide_label,
            "x": record.display_x,
            "y": record.display_y,
            "l1": l1,
            "l2": list(l2),
        }
        self.annotations[_annotation_key(package, record)] = payload
        self.flush()
        LOG.info(
            "annotation_save iac=%s row=%d tile_id=%s x=%d y=%d l1=%s l2=%s total_annotations=%d",
            package.rel_path,
            record.row,
            record.tile_id,
            record.display_x,
            record.display_y,
            l1,
            ",".join(l2) if l2 else "-",
            len(self.annotations),
        )

    def flush(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "input_path": self.input_path,
            "l1_prototypes": L1_PROTOTYPES,
            "l2_prototypes": L2_PROTOTYPES,
            "annotations": self.annotations,
        }
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.state_path.parent) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            tmp_path = Path(handle.name)
        tmp_path.replace(self.state_path)
        self._write_csv()
        LOG.info("state_flush json=%s csv=%s annotations=%d", self.state_path, self.csv_path, len(self.annotations))

    def _write_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", "iac", "iac_path", "tile_id", "row", "slide", "x", "y", "l1", "l2"],
            )
            writer.writeheader()
            for item in sorted(self.annotations.values(), key=lambda value: (value["iac"], value["row"])):
                row = dict(item)
                row["l2"] = ";".join(row["l2"])
                writer.writerow(row)


class AnnotationData:
    def __init__(self, input_path: str | Path, state_path: str | Path) -> None:
        self.packages = discover_iac_packages(input_path)
        self.state = AnnotationState(state_path, input_path)
        self._viewers: dict[int, IacViewerData] = {}

    def close(self) -> None:
        for viewer in self._viewers.values():
            viewer.close()

    def viewer(self, index: int) -> IacViewerData:
        if index not in self._viewers:
            package = self.packages[index]
            start = perf_counter()
            LOG.info("iac_open_start index=%d rel_path=%s path=%s", index, package.rel_path, package.path)
            viewer = IacViewerData(self.packages[index].path)
            if viewer.payload_type != "image_tiles":
                viewer.close()
                raise ValueError(f"annotation requires image-tile IAC package: {self.packages[index].path}")
            self._viewers[index] = viewer
            LOG.info(
                "iac_open_done index=%d rel_path=%s records=%d tile=%dx%d stride=%dx%d elapsed=%.3fs",
                index,
                package.rel_path,
                len(viewer.records),
                int(viewer.header.get("tile_width", 0)),
                int(viewer.header.get("tile_height", 0)),
                viewer.stride_x,
                viewer.stride_y,
                perf_counter() - start,
            )
        return self._viewers[index]

    def package_json(self) -> list[dict]:
        items = []
        for idx, package in enumerate(self.packages):
            counts = self.state.lightweight_counts_for_package(package)
            items.append(
                {
                    "index": idx,
                    "name": Path(package.rel_path).name,
                    "rel_path": package.rel_path,
                    "dataset": package.dataset,
                    "total": counts["total"],
                    "annotated": counts["annotated"],
                    "remaining": counts["remaining"],
                }
            )
        LOG.info("api_packages count=%d", len(items))
        return items

    def progress(self, index: int) -> dict:
        package = self.packages[index]
        viewer = self.viewer(index)
        counts = self.state.counts_for_package(package, viewer.records)
        l1_counts = {name: 0 for name in L1_PROTOTYPES}
        l2_counts = {name: 0 for name in L2_PROTOTYPES}
        for item in self.state.annotations_for_package(package):
            l1_counts[item["l1"]] = l1_counts.get(item["l1"], 0) + 1
            for label in item["l2"]:
                l2_counts[label] = l2_counts.get(label, 0) + 1
        LOG.info(
            "progress_read iac=%s annotated=%d total=%d remaining=%d",
            package.rel_path,
            counts["annotated"],
            counts["total"],
            counts["remaining"],
        )
        return {"package": counts, "l1": l1_counts, "l2": l2_counts}

    def random_record(self, index: int) -> dict:
        package = self.packages[index]
        viewer = self.viewer(index)
        remaining = [record for record in viewer.records if not self.state.is_annotated(package, record)]
        if not remaining:
            LOG.info("random_tile_empty iac=%s", package.rel_path)
            return {"record": None}
        record = random.choice(remaining)
        LOG.info("random_tile iac=%s row=%d tile_id=%s remaining=%d", package.rel_path, record.row, record.tile_id, len(remaining))
        return {"record": viewer._record_json(record)}

    def select_nearest(self, index: int, x: float, y: float) -> dict:
        viewer = self.viewer(index)
        result = viewer.nearest("__all__", x, y)
        record = result.get("record")
        if record:
            LOG.info("nearest_tile iac=%s x=%.1f y=%.1f row=%s tile_id=%s", self.packages[index].rel_path, x, y, record.get("row"), record.get("tile_id"))
        return result

    def annotated_rows(self, index: int) -> set[int]:
        package = self.packages[index]
        viewer = self.viewer(index)
        return {record.row for record in viewer.records if self.state.is_annotated(package, record)}

    def thumbnail_png(self, index: int, max_size: int = 1200) -> bytes:
        package = self.packages[index]
        viewer = self.viewer(index)
        start = perf_counter()
        records = viewer.records
        bounds = viewer._bounds(records)
        min_x, max_x, min_y, max_y = bounds
        footprint_w = max(1, viewer.stride_x)
        footprint_h = max(1, viewer.stride_y)
        width_span = max(1, max_x - min_x + footprint_w)
        height_span = max(1, max_y - min_y + footprint_h)
        scale = min(max_size / width_span, max_size / height_span)
        canvas_w = max(1, int(width_span * scale))
        canvas_h = max(1, int(height_span * scale))
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        annotated = self.state.annotations_for_package(package)
        tile_px_w = max(3, int(footprint_w * scale))
        tile_px_h = max(3, int(footprint_h * scale))
        sampled_records = records
        max_tiles = 6000
        if len(records) > max_tiles:
            sampled_records = random.Random(0).sample(records, max_tiles)
        for record in sampled_records:
            x = int((record.display_x - min_x) * scale)
            y = int((record.display_y - min_y) * scale)
            try:
                tile = Image.open(io.BytesIO(viewer.read_tile_png(record.row))).convert("RGB")
                tile = tile.resize((tile_px_w, tile_px_h), Image.Resampling.LANCZOS)
                canvas.paste(tile, (x, y))
            except Exception:
                LOG.exception("thumbnail_tile_decode_failed iac=%s row=%d", package.rel_path, record.row)
                pass
        draw = ImageDraw.Draw(canvas, "RGBA")
        for item in annotated:
            x = int((int(item["x"]) - min_x) * scale)
            y = int((int(item["y"]) - min_y) * scale)
            draw.rectangle([x, y, x + tile_px_w, y + tile_px_h], outline=(0, 140, 80, 220), width=2)
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        LOG.info(
            "thumbnail_render iac=%s mode=spatial records=%d sampled=%d annotated=%d bounds=(%d,%d,%d,%d) footprint=%dx%d canvas=%dx%d tile_px=%dx%d elapsed=%.3fs",
            package.rel_path,
            len(records),
            len(sampled_records),
            len(annotated),
            min_x,
            max_x,
            min_y,
            max_y,
            footprint_w,
            footprint_h,
            canvas_w,
            canvas_h,
            tile_px_w,
            tile_px_h,
            perf_counter() - start,
        )
        return buffer.getvalue()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HCC-SemPath Prototype Annotation</title>
<style>
body{margin:0;font:14px system-ui,-apple-system,Segoe UI,sans-serif;color:#202124;background:#f6f7f8}
.layout{display:grid;grid-template-columns:300px 1fr 320px;height:100vh}
aside,.right{overflow:auto;background:#fff;border-right:1px solid #d8dadd;padding:12px}.right{border-left:1px solid #d8dadd;border-right:0}
main{overflow:auto;padding:12px}.pkg{position:relative;padding:8px;border:1px solid #d8dadd;margin-bottom:6px;cursor:pointer;background:#fff;overflow:hidden}.pkg.active{border-color:#1a73e8;background:#eaf2ff}
.pkg>*{position:relative;z-index:1}.pkg::before{content:"";position:absolute;inset:0 auto 0 0;width:var(--pct,0%);background:#dff3eb;z-index:0}
.muted{color:#6b7280;font-size:12px}.thumbWrap{min-height:220px;border:1px solid #c7cbd1;background:white;display:flex;align-items:flex-start;justify-content:center}
.thumb{display:block;max-width:100%;background:white;cursor:crosshair}.loading{padding:18px;color:#6b7280;font-size:12px}
.tile{width:224px;height:224px;object-fit:contain;border:1px solid #c7cbd1;background:#fff}.chips{display:grid;grid-template-columns:1fr;gap:6px;margin:8px 0 16px}
button.chip{text-align:left;border:1px solid #c7cbd1;background:#fff;padding:8px;cursor:pointer}button.chip.selected{background:#1a73e8;color:white;border-color:#1a73e8}
.actions button{padding:8px 10px;margin-right:6px}.bar{height:8px;background:#e5e7eb;margin:6px 0 10px}.bar>div{height:8px;background:#18865b}
.stat{position:relative;border:1px solid #d8dadd;background:#fff;padding:7px 8px;margin:5px 0;overflow:hidden}.stat::before{content:"";position:absolute;inset:0 auto 0 0;width:var(--pct,0%);background:#eaf2ff;z-index:0}.stat>*{position:relative;z-index:1}
.stat .row{display:flex;justify-content:space-between;gap:8px}.stat .count{font-variant-numeric:tabular-nums;color:#374151}
pre{white-space:pre-wrap;font-size:12px;background:#f1f3f4;padding:8px}
</style>
</head>
<body><div class="layout"><aside><h3>IAC packages</h3><div id="packageSummary" class="muted"></div><div id="packages"></div></aside>
<main><h3 id="title">Prototype annotation</h3><div class="muted" id="recordMeta"></div>
<p><img id="tile" class="tile"></p><h3>Location overview</h3><div id="thumbWrap" class="thumbWrap"><div id="thumbLoading" class="loading">Loading overview...</div><img id="thumb" class="thumb"></div></main>
<section class="right"><h3>Progress</h3><div id="progress"></div><h3>L1 primary</h3><div id="l1" class="chips"></div>
<h3>L2 attributes</h3><div id="l2" class="chips"></div><div class="actions"><button onclick="save()">Save + next</button><button onclick="nextRandom()">Skip / random</button></div><pre id="status"></pre></section>
</div>
<script>
let packages=[], pkg=0, current=null, l1="", l2=new Set();
async function api(path, opts){const r=await fetch(path, opts); if(!r.ok) throw new Error(await r.text()); return r.headers.get('content-type')?.includes('json')?r.json():r.blob();}
function prototypeButton(name, selected, onClick){const b=document.createElement('button'); b.className='chip'+(selected?' selected':''); b.textContent=name; b.onclick=onClick; return b;}
function renderLabels(){const a=document.getElementById('l1'); a.innerHTML=''; L1.forEach(x=>a.appendChild(prototypeButton(x,l1===x,()=>{l1=x;renderLabels()}))); const b=document.getElementById('l2'); b.innerHTML=''; L2.forEach(x=>b.appendChild(prototypeButton(x,l2.has(x),()=>{l2.has(x)?l2.delete(x):l2.add(x);renderLabels()})));}
function statRow(name,count,maxCount){const pct=maxCount?Math.round(count/maxCount*100):0; return `<div class=stat style="--pct:${pct}%"><div class=row><span>${name}</span><span class=count>${count}</span></div></div>`;}
function statBlock(title, counts){const values=Object.values(counts); const maxCount=Math.max(1,...values); return `<b>${title}</b>`+Object.entries(counts).map(([name,count])=>statRow(name,count,maxCount)).join('');}
function setThumbnailSrc(){const img=document.getElementById('thumb'); const loading=document.getElementById('thumbLoading'); loading.style.display='block'; img.style.display='none'; img.onload=()=>{loading.style.display='none'; img.style.display='block';}; img.onerror=()=>{loading.textContent='Overview failed to load.';}; img.src='/api/thumbnail?package='+pkg+'&t='+Date.now();}
async function loadPackages(){packages=await api('/api/packages'); document.getElementById('packageSummary').textContent=`${packages.length} IAC package${packages.length===1?'':'s'}`; const box=document.getElementById('packages'); box.innerHTML=''; packages.forEach(p=>{const pct=p.total?Math.round(p.annotated/p.total*100):0; const d=document.createElement('div'); d.className='pkg'+(p.index===pkg?' active':''); d.style.setProperty('--pct',pct+'%'); d.innerHTML=`<b>${p.name}</b><div class=muted>${p.dataset||'no dataset'} · ${p.annotated}/${p.total} · ${pct}%</div>`; d.onclick=()=>{pkg=p.index; refreshPackage();}; box.appendChild(d)});}
async function refreshPackage(){await loadPackages(); setThumbnailSrc(); await progress(); await nextRandom();}
async function progress(){const p=await api('/api/progress?package='+pkg); const pct=p.package.total?Math.round(p.package.annotated/p.package.total*100):0; document.getElementById('progress').innerHTML=`<div>${p.package.annotated}/${p.package.total} (${pct}%)</div><div class=bar><div style="width:${pct}%"></div></div>${statBlock('L1',p.l1)}${statBlock('L2',p.l2)}`;}
async function showRecord(rec){current=rec; l1=""; l2=new Set(); renderLabels(); if(!rec){document.getElementById('recordMeta').textContent='All tiles in this IAC are annotated.'; document.getElementById('tile').removeAttribute('src'); return;} document.getElementById('recordMeta').textContent=`${packages[pkg].rel_path} · ${rec.tile_id} · x=${rec.x} y=${rec.y} row=${rec.row}`; document.getElementById('tile').src=`/api/tile?package=${pkg}&row=${rec.row}`;}
async function nextRandom(){const r=await api('/api/random?package='+pkg); await showRecord(r.record);}
async function save(){if(!current){await nextRandom(); return;} if(!l1){document.getElementById('status').textContent='Select one L1 primary prototype first.'; return;} await api('/api/annotation',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({package:pkg,row:current.row,l1,l2:[...l2]})}); document.getElementById('status').textContent='Saved.'; setThumbnailSrc(); await progress(); await loadPackages(); await nextRandom();}
document.getElementById('thumb').onclick=async ev=>{const img=ev.target, r=img.getBoundingClientRect(); const x=(ev.clientX-r.left)/r.width, y=(ev.clientY-r.top)/r.height; const rec=await api(`/api/nearest?package=${pkg}&rx=${x}&ry=${y}`); await showRecord(rec.record);}
const L1=%L1_JSON%; const L2=%L2_JSON%; renderLabels(); refreshPackage().catch(e=>document.getElementById('status').textContent=e);
</script></body></html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: dict | list) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(data: AnnotationData):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    LOG.info("ui_open path=/")
                    html = HTML.replace("%L1_JSON%", json.dumps(L1_PROTOTYPES)).replace("%L2_JSON%", json.dumps(L2_PROTOTYPES))
                    body = html.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/packages":
                    _json_response(self, data.package_json())
                    return
                index = int(qs.get("package", ["0"])[0])
                if parsed.path == "/api/progress":
                    _json_response(self, data.progress(index))
                    return
                if parsed.path == "/api/random":
                    _json_response(self, data.random_record(index))
                    return
                if parsed.path == "/api/nearest":
                    viewer = data.viewer(index)
                    rx = float(qs.get("rx", ["0"])[0])
                    ry = float(qs.get("ry", ["0"])[0])
                    bounds = viewer._bounds(viewer.records)
                    tile_w = max(1, int(viewer.header.get("tile_width", viewer.stride_x)))
                    tile_h = max(1, int(viewer.header.get("tile_height", viewer.stride_y)))
                    x = bounds[0] + rx * max(1, bounds[1] - bounds[0] + tile_w)
                    y = bounds[2] + ry * max(1, bounds[3] - bounds[2] + tile_h)
                    _json_response(self, data.select_nearest(index, x, y))
                    return
                if parsed.path == "/api/thumbnail":
                    body = data.thumbnail_png(index)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/tile":
                    row = int(qs["row"][0])
                    package = data.packages[index]
                    LOG.info("tile_read iac=%s row=%d", package.rel_path, row)
                    body = data.viewer(index).read_tile_png(row)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                LOG.exception("api_get_failed path=%s query=%s", parsed.path, parsed.query)
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                index = int(payload["package"])
                row = int(payload["row"])
                viewer = data.viewer(index)
                record = viewer._by_row[row]
                LOG.info("annotation_request iac=%s row=%d l1=%s l2=%s", data.packages[index].rel_path, row, payload.get("l1"), payload.get("l2", []))
                data.state.save_annotation(data.packages[index], record, str(payload["l1"]), list(payload.get("l2", [])))
                _json_response(self, {"ok": True})
            except Exception as exc:
                LOG.exception("api_post_failed path=%s", parsed.path)
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Open a browser UI for L1/L2 prototype tile annotation.")
    parser.add_argument("--input", required=True, help="IAC file or directory containing image-tile IAC packages.")
    parser.add_argument("--state", required=True, help="JSON annotation state file; CSV is exported next to it.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    data = AnnotationData(args.input, args.state)
    port = _find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(data))
    url = f"http://{args.host}:{port}/"
    LOG.info("server_start url=%s packages=%d state=%s csv=%s", url, len(data.packages), args.state, data.state.csv_path)
    print(f"annotation_ui url={url} packages={len(data.packages)} state={args.state} csv={data.state.csv_path}", flush=True)
    if not args.no_open:
        LOG.info("browser_open url=%s", url)
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
