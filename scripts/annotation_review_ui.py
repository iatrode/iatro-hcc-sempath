from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import tempfile
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.cli.annotate_prototypes import L1_PROTOTYPES
from hcc_sempath.cli.view_iac import IacViewerData


DEFAULT_UNSTABLE_L1 = {
    "Indeterminate-region",
    "Artifact-non-tissue",
    "Degenerative-material",
}


@dataclass(frozen=True)
class ReviewCandidate:
    key: str
    tile_id: str
    current_l1: str
    suggested_l1: str
    uncertainty: float
    score_summary: str


def _find_free_port(host: str, preferred: int) -> int:
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("annotations"), dict):
        raise ValueError(f"annotation JSON missing annotations object: {path}")
    return payload


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _package_keys(path: Path) -> list[str]:
    name = path.name
    suffixes = [
        ".prov-gigapath-local.features.iac",
        ".uni2_h.features.iac",
        ".virchow2.features.iac",
        ".features.iac",
    ]
    keys = []
    for suffix in suffixes:
        if name.endswith(suffix):
            keys.append(name[: -len(suffix)])
    keys.append(path.stem)
    return list(dict.fromkeys(key for key in keys if key))


def _tile_keys(tile_id: str) -> list[str]:
    keys = [tile_id]
    if "_" in tile_id:
        keys.append(tile_id.rsplit("_", 1)[0])
    return list(dict.fromkeys(keys))


def _discover_teacher_paths(root: Path, teachers: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for teacher in teachers:
        direct = root / teacher
        search_roots = [direct] if direct.exists() else [root]
        matches: list[Path] = []
        for search_root in search_roots:
            matches.extend(sorted(search_root.rglob("*.features.iac")))
            matches.extend(path for path in sorted(search_root.rglob("*features*.iac")) if path not in matches)
        teacher_matches = [path for path in matches if teacher in path.name or search_roots == [direct]]
        if teacher_matches:
            result[teacher] = teacher_matches
    return result


def _teacher_paths_from_arg(value: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for item in _parse_str_list(value):
        if "=" not in item:
            raise ValueError(f"teacher feature package entry must be teacher=path: {item}")
        teacher, paths = item.split("=", 1)
        parsed = [Path(path) for path in paths.split("|") if path]
        if parsed:
            result[teacher.strip()] = parsed
    return result


class FeatureL1Scorer:
    def __init__(self, payload: dict, *, teachers: list[str], teacher_paths: dict[str, list[Path]], l1_names: list[str]) -> None:
        from hcc_sempath.io.feature_cache import FeatureCacheReader

        self._reader_cls = FeatureCacheReader
        self._paths = teacher_paths
        self._path_index = {
            teacher: self._index_paths(paths)
            for teacher, paths in teacher_paths.items()
        }
        self._readers: dict[tuple[str, Path], object] = {}
        self._centers: dict[str, dict[str, np.ndarray]] = {}
        self._l1_names = [name for name in l1_names if name]
        self._build_centers(payload, teachers)

    @staticmethod
    def _index_paths(paths: list[Path]) -> dict[str, list[Path]]:
        result: dict[str, list[Path]] = {}
        for path in paths:
            for key in _package_keys(path):
                result.setdefault(key, []).append(path)
        return result

    def _reader(self, teacher: str, path: Path):
        key = (teacher, path)
        if key not in self._readers:
            self._readers[key] = self._reader_cls(path)
        return self._readers[key]

    def read(self, teacher: str, tile_id: str) -> np.ndarray | None:
        candidate_paths: list[Path] = []
        for key in _tile_keys(tile_id):
            candidate_paths.extend(self._path_index.get(teacher, {}).get(key, []))
        candidate_paths = list(dict.fromkeys([*candidate_paths, *self._paths.get(teacher, [])]))
        for path in candidate_paths:
            try:
                return self._reader(teacher, path).read_feature(tile_id)
            except FileNotFoundError:
                continue
        return None

    def _build_centers(self, payload: dict, teachers: list[str]) -> None:
        allowed = set(self._l1_names)
        for teacher in teachers:
            sums: dict[str, np.ndarray] = {}
            counts: dict[str, int] = {}
            for item in payload.get("annotations", {}).values():
                label = str(item.get("l1") or item.get("level1_label") or "")
                tile_id = str(item.get("tile_id") or "")
                if label not in allowed or not tile_id:
                    continue
                vector = self.read(teacher, tile_id)
                if vector is None:
                    continue
                vector = _normalize(vector)
                sums[label] = vector if label not in sums else sums[label] + vector
                counts[label] = counts.get(label, 0) + 1
            centers = {
                label: _normalize(total / max(1, counts[label]))
                for label, total in sums.items()
                if counts.get(label, 0) > 0
            }
            if centers:
                self._centers[teacher] = centers

    def suggest(self, item: dict) -> tuple[str, float, str]:
        current = str(item.get("l1") or item.get("level1_label") or "")
        tile_id = str(item.get("tile_id") or "")
        scores: dict[str, list[float]] = {}
        if not tile_id:
            return current, 0.0, ""
        for teacher, centers in self._centers.items():
            vector = self.read(teacher, tile_id)
            if vector is None:
                continue
            vector = _normalize(vector)
            for label, center in centers.items():
                scores.setdefault(label, []).append(float(np.dot(vector, center)))
        averaged = {
            label: sum(values) / len(values)
            for label, values in scores.items()
            if values and math.isfinite(sum(values) / len(values))
        }
        if not averaged:
            return current, 0.0, ""
        ranked = sorted(averaged.items(), key=lambda item: item[1], reverse=True)
        margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else ranked[0][1])
        uncertainty = max(0.0, 1.0 - margin)
        summary = "; ".join(f"{label}:{score:.3f}" for label, score in ranked[:3])
        return ranked[0][0], uncertainty, summary

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


def _suggest_l1(item: dict, mode: str, binary_a: str, binary_b: str, scorer: FeatureL1Scorer | None) -> tuple[str, float, str]:
    current = str(item.get("l1") or item.get("level1_label") or "")
    if mode == "binary":
        return current, 0.0, ""
    if scorer is None:
        return current, float(item.get("l1_uncertainty", item.get("uncertainty", 0.0)) or 0.0), ""
    return scorer.suggest(item)


def build_candidates(
    payload: dict,
    *,
    mode: str = "l1",
    binary_a: str = "Artifact-non-tissue",
    binary_b: str = "Degenerative-material",
    scorer: FeatureL1Scorer | None = None,
    unstable_l1: set[str] | None = None,
) -> list[ReviewCandidate]:
    unstable_l1 = unstable_l1 or DEFAULT_UNSTABLE_L1
    candidates: list[ReviewCandidate] = []
    for key, item in payload.get("annotations", {}).items():
        if bool(item.get("reviewed")):
            continue
        current = str(item.get("l1") or item.get("level1_label") or "")
        if mode == "binary" and current not in {binary_a, binary_b}:
            continue
        suggested, uncertainty, score_summary = _suggest_l1(item, mode, binary_a, binary_b, scorer)
        if current in unstable_l1:
            uncertainty += 1.0
        if suggested != current:
            uncertainty += 0.5
        candidates.append(
            ReviewCandidate(
                key=str(key),
                tile_id=str(item.get("tile_id", "")),
                current_l1=current,
                suggested_l1=suggested,
                uncertainty=uncertainty,
                score_summary=score_summary,
            )
        )
    return sorted(candidates, key=lambda item: (-item.uncertainty, item.tile_id, item.key))


class ReviewState:
    def __init__(
        self,
        annotation_json: Path,
        output_json: Path | None,
        *,
        mode: str,
        binary_a: str,
        binary_b: str,
        teachers: list[str] | None = None,
        teacher_feature_root: Path | None = None,
        teacher_feature_packages: str = "",
        unstable_l1: set[str] | None = None,
    ) -> None:
        self.annotation_json = annotation_json
        self.output_json = output_json or annotation_json
        self.payload = _load_payload(annotation_json)
        self.mode = mode
        self.binary_a = binary_a
        self.binary_b = binary_b
        self._viewers: dict[str, IacViewerData] = {}
        self.unstable_l1 = unstable_l1 or DEFAULT_UNSTABLE_L1
        self.scorer: FeatureL1Scorer | None = None
        teachers = teachers or []
        teacher_paths = _teacher_paths_from_arg(teacher_feature_packages)
        if teacher_feature_root and teachers:
            teacher_paths.update(_discover_teacher_paths(teacher_feature_root, teachers))
        if mode == "l1" and teachers and teacher_paths:
            l1_names = list(self.payload.get("l1_prototypes", L1_PROTOTYPES))
            self.scorer = FeatureL1Scorer(self.payload, teachers=teachers, teacher_paths=teacher_paths, l1_names=l1_names)

    @property
    def csv_path(self) -> Path:
        return self.output_json.with_suffix(".review.csv")

    def candidates(self) -> list[ReviewCandidate]:
        return build_candidates(
            self.payload,
            mode=self.mode,
            binary_a=self.binary_a,
            binary_b=self.binary_b,
            scorer=self.scorer,
            unstable_l1=self.unstable_l1,
        )

    def candidate_payload(self) -> dict:
        candidates = self.candidates()
        return {
            "remaining": len(candidates),
            "candidates": [candidate.__dict__ for candidate in candidates],
            "l1_prototypes": self.payload.get("l1_prototypes", L1_PROTOTYPES),
            "mode": self.mode,
            "binary_a": self.binary_a,
            "binary_b": self.binary_b,
        }

    def item(self, key: str) -> dict:
        return self.payload["annotations"][key]

    def viewer(self, iac_path: str) -> IacViewerData:
        if iac_path not in self._viewers:
            self._viewers[iac_path] = IacViewerData(iac_path)
        return self._viewers[iac_path]

    def tile_png(self, key: str) -> bytes:
        item = self.item(key)
        viewer = self.viewer(str(item["iac_path"]))
        row = int(item["row"])
        return viewer.read_tile_png(row)

    def review(self, key: str, decision: str, new_l1: str = "") -> dict:
        item = self.item(key)
        current = str(item.get("l1") or "")
        suggested = _suggest_l1(item, self.mode, self.binary_a, self.binary_b, self.scorer)[0]
        if decision == "accept":
            item["l1"] = suggested
        elif decision == "adjust":
            if not new_l1:
                raise ValueError("adjust requires new_l1")
            item["l1"] = new_l1
        elif decision != "reject":
            raise ValueError(f"unknown review decision: {decision}")
        item["reviewed"] = True
        item["review_decision"] = decision
        item["review_previous_l1"] = current
        item["review_suggested_l1"] = suggested
        self.flush()
        return item

    def flush(self) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.output_json.parent) as handle:
            json.dump(self.payload, handle, indent=2, sort_keys=True)
            tmp_path = Path(handle.name)
        tmp_path.replace(self.output_json)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["tile_id", "l1", "reviewed", "review_decision", "review_previous_l1", "review_suggested_l1"],
            )
            writer.writeheader()
            for item in self.payload.get("annotations", {}).values():
                writer.writerow({field: item.get(field, "") for field in writer.fieldnames})

    def close(self) -> None:
        for viewer in self._viewers.values():
            viewer.close()
        if self.scorer is not None:
            self.scorer.close()


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation Review</title>
<style>
body{margin:0;font:14px system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f8;color:#202124}
.layout{display:grid;grid-template-columns:320px 1fr 300px;height:100vh}.side{background:#fff;border-right:1px solid #d8dadd;overflow:auto;padding:12px}.right{border-left:1px solid #d8dadd;border-right:0}.main{padding:16px;overflow:auto}
.row{padding:8px;border-bottom:1px solid #e5e7eb;cursor:pointer}.row.active{background:#eaf2ff}.muted{color:#6b7280;font-size:12px}.tile{max-width:680px;width:100%;image-rendering:auto;background:#fff;border:1px solid #c7cbd1}
button{padding:8px 10px;margin:4px;border:1px solid #c7cbd1;background:#fff;cursor:pointer}button.primary{background:#1a73e8;color:white;border-color:#1a73e8}
select{width:100%;padding:8px;margin:8px 0}
</style></head><body><div class="layout"><div class="side"><h3>Review queue</h3><div id="count" class="muted"></div><div id="list"></div></div>
<div class="main"><h2 id="title">No tile selected</h2><div id="meta" class="muted"></div><p><img id="tile" class="tile"></p></div>
<div class="side right"><h3>Decision</h3><div id="decision"></div><select id="newL1"></select><button class="primary" onclick="act('adjust')">Adjust</button><pre id="status"></pre></div></div>
<script>
let queue=[], current=null, labels=[];
async function api(path, opts){const r=await fetch(path,opts); if(!r.ok) throw new Error(await r.text()); return r.headers.get('content-type')?.includes('json')?r.json():r.blob();}
let mode='l1', binaryA='', binaryB='';
async function load(){const p=await api('/api/candidates'); queue=p.candidates; labels=p.l1_prototypes; mode=p.mode; binaryA=p.binary_a; binaryB=p.binary_b; document.getElementById('count').textContent=`${p.remaining} remaining · ${p.mode}`; const sel=document.getElementById('newL1'); sel.innerHTML=''; labels.forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x;sel.appendChild(o)}); renderList(); if(queue.length) show(queue[0].key);}
function renderList(){const box=document.getElementById('list'); box.innerHTML=''; queue.forEach(c=>{const d=document.createElement('div');d.className='row'+(current&&current.key===c.key?' active':'');d.innerHTML=`<b>${c.tile_id}</b><div class=muted>${c.current_l1} -> ${c.suggested_l1} · ${c.uncertainty.toFixed(3)}</div>`;d.onclick=()=>show(c.key);box.appendChild(d);});}
async function show(key){const data=await api('/api/item?key='+encodeURIComponent(key)); current=data.candidate; document.getElementById('title').textContent=data.item.tile_id; document.getElementById('meta').textContent=`Current: ${current.current_l1} · Suggested: ${current.suggested_l1} · ${current.score_summary||''}`; document.getElementById('tile').src='/api/tile?key='+encodeURIComponent(key)+'&t='+Date.now(); document.getElementById('newL1').value=current.suggested_l1; if(mode==='binary'){document.getElementById('decision').innerHTML=`<button class=primary onclick="choose('${binaryA}')">${binaryA}</button><button class=primary onclick="choose('${binaryB}')">${binaryB}</button>`;}else{document.getElementById('decision').innerHTML=`<button class=primary onclick="act('accept')">Accept suggested</button><button onclick="act('reject')">Reject</button>`;} renderList();}
async function choose(label){document.getElementById('newL1').value=label; await act('adjust');}
async function act(decision){if(!current)return; const newL1=document.getElementById('newL1').value; await api('/api/review',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({key:current.key,decision,new_l1:newL1})}); document.getElementById('status').textContent='saved'; await load();}
load().catch(e=>document.getElementById('status').textContent=e);
</script></body></html>"""


def _json_response(handler: BaseHTTPRequestHandler, payload: dict | list) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/candidates":
                    _json_response(self, state.candidate_payload())
                    return
                if parsed.path == "/api/item":
                    key = qs["key"][0]
                    candidates = {candidate.key: candidate for candidate in state.candidates()}
                    _json_response(self, {"candidate": candidates[key].__dict__, "item": state.item(key)})
                    return
                if parsed.path == "/api/tile":
                    body = state.tile_png(qs["key"][0])
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                item = state.review(str(payload["key"]), str(payload["decision"]), str(payload.get("new_l1") or ""))
                _json_response(self, {"ok": True, "item": item})
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a focused annotation review UI.")
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--mode", choices=["l1", "binary"], default="l1")
    parser.add_argument("--class-a", "--binary-a", dest="binary_a", default="Artifact-non-tissue")
    parser.add_argument("--class-b", "--binary-b", dest="binary_b", default="Degenerative-material")
    parser.add_argument("--teachers", default="")
    parser.add_argument("--teacher-feature-root", default="")
    parser.add_argument("--teacher-feature-packages", default="")
    parser.add_argument("--unstable-l1", default="Indeterminate-region,Artifact-non-tissue,Degenerative-material")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    state = ReviewState(
        Path(args.annotation_json),
        Path(args.output_json) if args.output_json else None,
        mode=args.mode,
        binary_a=args.binary_a,
        binary_b=args.binary_b,
        teachers=_parse_str_list(args.teachers),
        teacher_feature_root=Path(args.teacher_feature_root) if args.teacher_feature_root else None,
        teacher_feature_packages=args.teacher_feature_packages,
        unstable_l1=set(_parse_str_list(args.unstable_l1)),
    )
    port = _find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(state))
    url = f"http://{args.host}:{port}/"
    print(f"annotation_review_ui url={url} state={state.output_json}", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.close()
        server.server_close()


if __name__ == "__main__":
    main()
